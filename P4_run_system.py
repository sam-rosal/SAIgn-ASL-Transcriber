import json
import operator
import os
import sqlite3
import unicodedata
from collections import deque, Counter
import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp
from scipy.interpolate import interp1d

# ==============================================================================
# PATHS & SYSTEM CONFIGURATION
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, 'saign_vision.db')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'asl_model.keras')
CLASSES_PATH = os.path.join(SCRIPT_DIR, 'classes.json')
MENU_ART_PATH = os.path.join(SCRIPT_DIR, 'saign_menu.png')  # main-menu artwork (falls back to a drawn menu)

SEQUENCE_LENGTH = 30
RAW_CAPTURE_WINDOW = 45  # ~1.5s rolling capture buffer before velocity segmentation + resampling
VELOCITY_THRESHOLD = 0.015
MIN_ACTIVE_FRAMES = 10
EXPECTED_FEATURE_DIM = 225  # Pose (99) + Left Hand (63) + Right Hand (63)
EXPRESSION_SEED_VERSION = 2  # Must match P4_setup_db.py. v2 = face-size-normalised ratio thresholds.

DEBOUNCE_WINDOW_SIZE = 8
DEBOUNCE_CONSENSUS_COUNT = 5
VOTE_CLEAR_NO_HAND_FRAMES = 5   # Flush stale vote history after ~0.17s without hands
NO_HAND_RESET_FRAMES = 15       # Reset active sign HUD after ~0.5s with no hands visible

# --- UI options ---
UI_W, UI_H = 1280, 720          # logical size of the live screen; the OpenCV window scales it
WAITING_TEXT = "Waiting for signs"
MIRROR_VIEW = True              # show the camera like a mirror (the model still sees the unmirrored frame)
START_FULLSCREEN = False        # open the live screen full screen (F toggles it either way)
DRAW_POSE = False               # also draw the body-pose skeleton (the notebook UI shows hands only)


# ==============================================================================
# HELPER FUNCTIONS: DATABASE, LABELS & SIGNAL PROCESSING
# ==============================================================================
def load_trained_classes():
    """
    Loads the frozen class index array saved during training (classes.json),
    falling back to alphabetical unique labels from SQLite if classes.json is missing.
    """
    if os.path.exists(CLASSES_PATH):
        try:
            with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
                classes = json.load(f)
                if isinstance(classes, list) and classes:
                    return [str(c) for c in classes]
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] Could not parse {CLASSES_PATH} ({e}). Falling back to SQLite...")

    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT label FROM training_dataset WHERE label IS NOT NULL ORDER BY label ASC;")
            rows = cursor.fetchall()
            if rows:
                return [row[0] for row in rows]
        except sqlite3.OperationalError:
            pass
        finally:
            conn.close()

    return []


def load_db_cache():
    """
    Loads display name mappings, confidence thresholds, and facial expression
    thresholds from SQLite.
    """
    trained_labels = load_trained_classes()
    gesture_cache = {}
    expression_thresholds = {}

    if not os.path.exists(DB_PATH):
        return trained_labels, gesture_cache, expression_thresholds

    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    try:
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT raw_label, display_name, min_score FROM gesture_mappings;")
            for raw_label, display_name, min_score in cursor.fetchall():
                if not raw_label:
                    continue
                try:
                    parsed_min_score = float(min_score) if min_score is not None else 0.40
                except (TypeError, ValueError):
                    parsed_min_score = 0.40
                gesture_cache[raw_label] = {
                    'display_name': display_name or raw_label.replace('_', ' ').title(),
                    'min_score': parsed_min_score
                }
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("SELECT blendshape_name, display_name, activation_threshold FROM expression_thresholds;")
            for blendshape, display_name, threshold in cursor.fetchall():
                if not blendshape or threshold is None:
                    continue
                try:
                    parsed_threshold = float(threshold)
                except (TypeError, ValueError):
                    continue
                expression_thresholds[blendshape] = {
                    'display_name': display_name or blendshape,
                    'threshold': parsed_threshold
                }
        except sqlite3.OperationalError:
            pass

        # Thresholds seeded by older P4_setup_db.py versions use raw image-fraction units that
        # are incompatible with the ratio-based metrics below, so don't apply them.
        try:
            row = cursor.execute("SELECT value FROM app_meta WHERE key = 'expression_seed_version';").fetchone()
            seed_version = int(row[0]) if row and str(row[0]).strip().isdigit() else 0
        except sqlite3.OperationalError:
            seed_version = 0
        if expression_thresholds and seed_version < EXPRESSION_SEED_VERSION:
            print("[WARN] expression_thresholds in the database use the old unit system. "
                  "Re-run P4_setup_db.py to install the updated values; expression detection is disabled until then.")
            expression_thresholds = {}
    finally:
        conn.close()

    if not trained_labels and gesture_cache:
        trained_labels = sorted(gesture_cache.keys())

    return trained_labels, gesture_cache, expression_thresholds


# Red -> orange -> yellow -> green, identical to the gradient in the Help popover (BGR).
CONFIDENCE_SCALE_STOPS = [
    (0.00, (47, 41, 253)),    # #fd292f
    (0.35, (44, 144, 255)),   # #ff902c
    (0.66, (44, 243, 255)),   # #fff32c
    (1.00, (0, 250, 54)),     # #36fa00
]


def get_confidence_color(score):
    """BGR colour for a 0..1 confidence, sampled from the Help popover's gradient."""
    score = max(0.0, min(1.0, float(score)))
    xs = [s for s, _ in CONFIDENCE_SCALE_STOPS]
    return tuple(int(round(np.interp(score, xs, [c[i] for _, c in CONFIDENCE_SCALE_STOPS]))) for i in range(3))


def extract_landmarks(results):
    """
    Flattens spatial (X, Y, Z) coordinates from MediaPipe Holistic output
    into a single 1D float32 feature vector of exactly 225 values.
    """
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark], dtype=np.float32).flatten() \
        if results.pose_landmarks else np.zeros(33 * 3, dtype=np.float32)

    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark], dtype=np.float32).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3, dtype=np.float32)

    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark], dtype=np.float32).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3, dtype=np.float32)

    return np.nan_to_num(np.concatenate([pose, lh, rh]), nan=0.0, posinf=0.0, neginf=0.0)


def prepare_live_sequence(raw_buffer, target_frames=SEQUENCE_LENGTH):
    """
    Mirrors P4_extract_features.py preprocessing on the live rolling window:
    filters static transition frames via L2 velocity and resamples active motion
    to target_frames (30), falling back to the latest 30 raw frames for static signs.
    """
    features = np.array(raw_buffer, dtype=np.float32)

    if len(features) >= 2:
        frame_differences = features[1:] - features[:-1]
        velocities = np.linalg.norm(frame_differences, axis=1)
        active_mask = np.ones(len(features), dtype=bool)
        active_mask[1:] = velocities >= VELOCITY_THRESHOLD
        filtered = features[active_mask]

        if len(filtered) >= MIN_ACTIVE_FRAMES:
            features = filtered
        else:
            features = features[-target_frames:]

    current_frames = len(features)
    if current_frames == 0:
        feat_cols = features.shape[1] if features.ndim == 2 else EXPECTED_FEATURE_DIM
        return np.zeros((target_frames, feat_cols), dtype=np.float32)

    if current_frames == target_frames:
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    if current_frames < 2:
        pad_amount = max(0, target_frames - current_frames)
        return np.pad(features, ((0, pad_amount), (0, 0)), mode='edge')[:target_frames]

    x_old = np.linspace(0, 1, num=current_frames)
    x_new = np.linspace(0, 1, num=target_frames)
    interpolator = interp1d(x_old, features, axis=0, kind='linear')
    return np.nan_to_num(interpolator(x_new).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def detect_expressions(results, thresholds, image_size=None):
    """
    Geometric facial-expression detection from the 468-point face mesh.

    Every metric is expressed as a RATIO of the face's own size (S = distance between the outer
    eye corners, landmarks 33 and 263), measured in PIXELS (image_size = (width, height)). That makes
    thresholds independent of camera distance and frame aspect ratio. Metric definitions
    (thresholds in the DB's expression_thresholds table are in these units):

      browSquint / browLowerer : |lm70.y  - lm159.y| / S     brow-to-eyelid gap          (smaller = lowered)
      browInnerUp              : |lm55.y  - lm6.y|   / S     inner brow-to-nose bridge   (larger  = raised)
      eyeWide* / eyeSquint*    : |lid_top.y - lid_bottom.y| / S  (159/145 and 386/374)   (larger = wider)
      mouthSmile*              : (lm13.y - corner.y) / S     corner LIFT above upper-lip centre (corners 61 / 291)
      mouthFrown*              : (corner.y - lm13.y) / S     corner DROP below upper-lip centre
      jawOpen                  : |lm14.y - lm13.y|   / S     inner-lip gap
      mouthPucker              : |lm291.x - lm61.x|  / S     mouth width                 (smaller = puckered)
      noseSneer*               : |lm197.y - lm0.y|   / S     nose bridge-to-upper-lip    (smaller = sneer)
    """
    active_emotions = []
    if not results.face_landmarks or not thresholds:
        return active_emotions

    landmarks = results.face_landmarks.landmark
    if len(landmarks) < 468:
        return active_emotions

    width, height = image_size if image_size else (1.0, 1.0)

    def px(i):
        return landmarks[i].x * width, landmarks[i].y * height

    (lx, ly), (rx, ry) = px(33), px(263)
    scale = float(np.hypot(rx - lx, ry - ly))
    if not np.isfinite(scale) or scale < 1e-6:
        return active_emotions

    def vdist(a, b):
        return abs(px(a)[1] - px(b)[1]) / scale

    def vdiff(a, b):  # signed: (a.y - b.y) / S   (image y grows downward)
        return (px(a)[1] - px(b)[1]) / scale

    def hdist(a, b):
        return abs(px(a)[0] - px(b)[0]) / scale

    def add(label):
        if label not in active_emotions:
            active_emotions.append(label)

    def pair_check(left_key, right_key, left_val, right_val, op, default_label):
        hit = any(
            key in thresholds and op(val, thresholds[key]['threshold'])
            for key, val in ((left_key, left_val), (right_key, right_val))
        )
        if hit:
            source = thresholds.get(left_key) or thresholds.get(right_key) or {}
            add(source.get('display_name', default_label))

    # Brow furrow / squint (stricter 'browSquint' checked before 'browLowerer')
    if 'browLowerer' in thresholds or 'browSquint' in thresholds:
        brow_gap = vdist(70, 159)
        if 'browSquint' in thresholds and brow_gap < thresholds['browSquint']['threshold']:
            add(thresholds['browSquint']['display_name'])
        elif 'browLowerer' in thresholds and brow_gap < thresholds['browLowerer']['threshold']:
            add(thresholds['browLowerer']['display_name'])

    # Inner brow raise
    if 'browInnerUp' in thresholds and vdist(55, 6) > thresholds['browInnerUp']['threshold']:
        add(thresholds['browInnerUp']['display_name'])

    # Eye aperture (shared between eyeWide and eyeSquint)
    left_eye_open = vdist(159, 145)
    right_eye_open = vdist(386, 374)
    pair_check('eyeWideLeft', 'eyeWideRight', left_eye_open, right_eye_open, operator.gt, 'Fearful / Startled')
    pair_check('eyeSquintLeft', 'eyeSquintRight', left_eye_open, right_eye_open, operator.lt, 'Suspicious / Wincing')

    # Smile (corners lifted above the upper-lip centre) / Frown (corners dropped below it)
    if 'mouthSmileLeft' in thresholds or 'mouthSmileRight' in thresholds:
        pair_check('mouthSmileLeft', 'mouthSmileRight',
                   vdiff(13, 61), vdiff(13, 291), operator.gt, 'Happy')

    if 'mouthFrownLeft' in thresholds or 'mouthFrownRight' in thresholds:
        pair_check('mouthFrownLeft', 'mouthFrownRight',
                   vdiff(61, 13), vdiff(291, 13), operator.gt, 'Sad / Concerned')

    # Jaw drop / open mouth
    if 'jawOpen' in thresholds and vdist(14, 13) > thresholds['jawOpen']['threshold']:
        add(thresholds['jawOpen']['display_name'])

    # Mouth pucker / horizontal narrowing
    if 'mouthPucker' in thresholds and hdist(291, 61) < thresholds['mouthPucker']['threshold']:
        add(thresholds['mouthPucker']['display_name'])

    # Nose sneer
    if 'noseSneerLeft' in thresholds or 'noseSneerRight' in thresholds:
        nose_dist = vdist(197, 0)
        pair_check('noseSneerLeft', 'noseSneerRight', nose_dist, nose_dist, operator.lt, 'Disgust')

    # Neutral fallback
    if not active_emotions and 'faceNeutral' in thresholds:
        add(thresholds['faceNeutral']['display_name'])

    return active_emotions


def is_window_closed(window_name):
    """Safely checks whether an OpenCV window has been closed by the user."""
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


# ==============================================================================
# UI THEME & DRAWING PRIMITIVES
# (OpenCV port of the SAIgn notebook's browser UI: same layout, colours and copy)
# ==============================================================================
# All colours are BGR tuples. The hex value of the original CSS colour is noted.
COLOR_YELLOW = (52, 203, 255)        # #ffcb34  brand yellow / borders
COLOR_YELLOW_TEXT = (49, 204, 255)   # #ffcc31  translation text
COLOR_GREEN_TEXT = (90, 248, 114)    # #72f85a  emotion text
COLOR_LIVE = (54, 250, 67)           # #43fa36  LIVE dot
COLOR_OFFLINE = (160, 145, 139)      # #8b91a0  OFFLINE dot
COLOR_CARD_BG = (15, 10, 8)          # #080a0f  overlay cards
COLOR_NAVY = (24, 13, 8)             # #080d18  menu background
COLOR_PANEL = (40, 28, 16)           # #101c28  menu panels
COLOR_INK = (32, 16, 6)              # #061020  dark text on yellow
COLOR_WHITE = (255, 255, 255)
COLOR_MUTED = (221, 221, 221)        # #dddddd
COLOR_SUBTLE = (185, 176, 173)       # #adb0b9
COLOR_HAND_LINE = (57, 225, 255)     # #ffe139
COLOR_HAND_DOT = (217, 239, 50)      # #32efd9
COLOR_HAND_RING = (255, 255, 240)    # #f0ffff

FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX

# Same 21-edge hand skeleton the notebook draws.
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
              (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]

CONFIDENCE_SCALE_LABELS = [("0-24%", "Low"), ("25-49%", "Fair"), ("50-74%", "Moderate"), ("75-100%", "High")]
CONFIDENCE_FOOTNOTE = ("Higher confidence means the model favours its prediction more strongly; "
                       "it does not guarantee accuracy. Emotion labels are estimates based on facial cues.")

# Menu artwork is 1120x720; click zones are the same percentages the notebook used.
MENU_W, MENU_H = 1120, 720


def _pct_rect(left, top, width, height):
    return (round(MENU_W * left / 100), round(MENU_H * top / 100),
            round(MENU_W * (left + width) / 100), round(MENU_H * (top + height) / 100))


MENU_START_RECT = _pct_rect(7.05, 62.78, 35.9, 9.9)
MENU_EXIT_RECT = _pct_rect(7.05, 75.14, 14.2, 5.6)


def _ascii(text):
    """OpenCV's Hershey fonts only render ASCII; map common punctuation and drop the rest."""
    text = str(text).replace('\u2013', '-').replace('\u2014', '-').replace('\u2019', "'")
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')


def text_size(text, scale, thickness=1, font=FONT_BOLD):
    (w, h), _ = cv2.getTextSize(_ascii(text), font, scale, thickness)
    return w, h


def put_text(img, text, org, scale, color, thickness=1, font=FONT_BOLD, align='left', shadow=False):
    """Anti-aliased text with optional left/center/right alignment and drop shadow. Returns the width."""
    text = _ascii(text)
    width, _ = text_size(text, scale, thickness, font)
    x, y = int(org[0]), int(org[1])
    if align == 'center':
        x -= width // 2
    elif align == 'right':
        x -= width
    if shadow:
        cv2.putText(img, text, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return width


def fit_text(text, max_width, scale, thickness=1, font=FONT_BOLD, min_ratio=0.65):
    """Shrinks text toward min_ratio*scale, then ellipsizes, so it fits max_width. Returns (text, scale)."""
    text = _ascii(text)
    s = scale
    while text_size(text, s, thickness, font)[0] > max_width and s > scale * min_ratio:
        s -= 0.03
    if text_size(text, s, thickness, font)[0] > max_width:
        while len(text) > 1 and text_size(text + '...', s, thickness, font)[0] > max_width:
            text = text[:-1]
        text = text.rstrip() + '...'
    return text, s


def wrap_text(text, max_width, scale, thickness=1, font=FONT_SMALL):
    lines, current = [], ''
    for word in _ascii(text).split():
        candidate = f"{current} {word}".strip()
        if current and text_size(candidate, scale, thickness, font)[0] > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _rounded_fill(img, x1, y1, x2, y2, r, color):
    if r <= 0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        return
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def _rounded_stroke(img, x1, y1, x2, y2, r, color, t):
    if r <= 0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, t, cv2.LINE_AA)
        return
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, t, cv2.LINE_AA)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, t, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, t, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, t, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 0, 180, 270, color, t, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 0, 270, 360, color, t, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, t, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 0, 90, 180, color, t, cv2.LINE_AA)


def draw_rounded_rect(img, p1, p2, radius, fill=None, fill_alpha=1.0,
                      border=None, border_px=2, border_alpha=1.0):
    """Rounded rectangle with optional translucent fill and border (the 'glass card' look)."""
    x1, y1, x2, y2 = int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])
    img_h, img_w = img.shape[:2]
    pad = border_px + 2
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(img_w, x2 + pad + 1), min(img_h, y2 + pad + 1)
    if rx2 <= rx1 or ry2 <= ry1:
        return
    roi = img[ry1:ry2, rx1:rx2]
    x1, y1, x2, y2 = x1 - rx1, y1 - ry1, x2 - rx1, y2 - ry1
    r = max(0, min(int(radius), (x2 - x1) // 2, (y2 - y1) // 2))

    for color, alpha, painter in (
        (fill, fill_alpha, lambda layer, c: _rounded_fill(layer, x1, y1, x2, y2, r, c)),
        (border, border_alpha, lambda layer, c: _rounded_stroke(layer, x1, y1, x2, y2, r, c, border_px)),
    ):
        if color is None:
            continue
        if alpha >= 1.0:
            painter(roi, color)
        else:
            layer = roi.copy()
            painter(layer, color)
            roi[:] = cv2.addWeighted(layer, alpha, roi, 1.0 - alpha, 0)


def _inside(rect, x, y):
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


# --- small icons (the notebook used unicode glyphs, which Hershey fonts can't render) ---
def _icon_translation(img, cx, cy, color):
    draw_rounded_rect(img, (cx - 13, cy - 11), (cx + 13, cy + 11), 4, border=color, border_px=2)
    for dy in (-5, 0, 5):
        cv2.line(img, (cx - 7, cy + dy), (cx + 7, cy + dy), color, 2, cv2.LINE_AA)


def _icon_emotion(img, cx, cy, color):
    cv2.circle(img, (cx, cy), 14, color, 2, cv2.LINE_AA)
    cv2.circle(img, (cx - 5, cy - 4), 2, color, -1, cv2.LINE_AA)
    cv2.circle(img, (cx + 5, cy - 4), 2, color, -1, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy + 1), (7, 6), 0, 20, 160, color, 2, cv2.LINE_AA)


def _icon_fullscreen(img, cx, cy, color, s=7):
    for sx in (-1, 1):
        for sy in (-1, 1):
            px, py = cx + sx * s, cy + sy * s
            cv2.line(img, (px, py), (px - sx * 4, py), color, 2, cv2.LINE_AA)
            cv2.line(img, (px, py), (px, py - sy * 4), color, 2, cv2.LINE_AA)


def _icon_back(img, cx, cy, color):
    cv2.arrowedLine(img, (cx + 7, cy), (cx - 7, cy), color, 2, cv2.LINE_AA, tipLength=0.5)


# ==============================================================================
# MAIN MENU
# ==============================================================================
def _build_fallback_menu():
    """Programmatic version of the menu, used only if saign_menu.png is missing."""
    img = np.full((MENU_H, MENU_W, 3), COLOR_NAVY, dtype=np.uint8)
    draw_rounded_rect(img, (50, 112), (1070, 600), 22, fill=COLOR_PANEL, border=COLOR_WHITE,
                      border_px=1, border_alpha=0.12)
    draw_rounded_rect(img, (50, 618), (1070, 686), 14, fill=COLOR_PANEL, border=COLOR_WHITE,
                      border_px=1, border_alpha=0.12)
    put_text(img, "ASL Transcriber", (MENU_W // 2, 662), 1.0, COLOR_WHITE, 2, align='center')

    draw_rounded_rect(img, (50, 37), (99, 86), 13, fill=COLOR_YELLOW)
    put_text(img, "S", (74, 72), 1.2, COLOR_INK, 3, align='center')
    put_text(img, "SAIgn", (113, 66), 1.0, COLOR_WHITE, 2)
    put_text(img, "VISION ENGINE", (114, 84), 0.42, COLOR_SUBTLE, 1, FONT_SMALL)

    put_text(img, "WELCOME TO SAIGN", (80, 182), 0.6, COLOR_YELLOW, 2)
    put_text(img, "Sign language,", (78, 265), 1.7, (250, 240, 235), 3)
    put_text(img, "made visible.", (78, 328), 1.7, COLOR_YELLOW, 3)
    put_text(img, "Real-time ASL recognition and transcription,", (80, 380), 0.62, COLOR_SUBTLE, 1, FONT_SMALL)
    put_text(img, "with expression awareness in one workspace.", (80, 408), 0.62, COLOR_SUBTLE, 1, FONT_SMALL)

    x1, y1, x2, y2 = MENU_START_RECT
    draw_rounded_rect(img, (x1, y1), (x2, y2), 14, fill=COLOR_YELLOW)
    put_text(img, "START RECOGNITION", (x1 + 26, (y1 + y2) // 2 + 9), 0.85, COLOR_INK, 2)
    draw_rounded_rect(img, (x2 - 88, y1 + 19), (x2 - 20, y2 - 19), 9, fill=(40, 175, 220))
    put_text(img, "ENTER", (x2 - 54, (y1 + y2) // 2 + 6), 0.5, COLOR_INK, 2, align='center')

    x1, y1, x2, y2 = MENU_EXIT_RECT
    draw_rounded_rect(img, (x1, y1), (x2, y2), 10, fill=(50, 36, 22), border=COLOR_WHITE,
                      border_px=1, border_alpha=0.15)
    put_text(img, "EXIT", (x1 + 28, y2 - 12), 0.7, COLOR_WHITE, 2)
    put_text(img, "ESC", (x1 + 100, y2 - 13), 0.5, COLOR_SUBTLE, 1, FONT_SMALL)

    draw_rounded_rect(img, (579, 135), (1055, 580), 20, border=COLOR_WHITE, border_px=1, border_alpha=0.12)
    put_text(img, "VISION TRACKING", (608, 176), 0.8, COLOR_WHITE, 2)
    put_text(img, "Camera starts after selection", (608, 560), 0.55, COLOR_SUBTLE, 1, FONT_SMALL)
    return img


def _load_menu_art():
    if os.path.exists(MENU_ART_PATH):
        art = cv2.imread(MENU_ART_PATH, cv2.IMREAD_COLOR)
        if art is not None:
            if art.shape[1] != MENU_W or art.shape[0] != MENU_H:
                art = cv2.resize(art, (MENU_W, MENU_H), interpolation=cv2.INTER_AREA)
            return art
        print(f"[WARN] Could not read {MENU_ART_PATH}. Using the built-in menu layout instead.")
    else:
        print(f"[INFO] {os.path.basename(MENU_ART_PATH)} not found next to this script. "
              "Using the built-in menu layout.")
    return _build_fallback_menu()


def show_main_menu(status_message=None):
    """
    Main menu. Start: click the button, or press ENTER / SPACE. Exit: click Exit, press ESC,
    or close the window. Returns 'START' or 'QUIT'. status_message (e.g. a camera error) is
    shown as a banner along the top.
    """
    window_name = 'SAIgn - Main Menu'
    art = _load_menu_art()
    state = {'mouse': (-1, -1), 'click': None}

    def on_mouse(event, x, y, flags, param):
        state['mouse'] = (x, y)
        if event == cv2.EVENT_LBUTTONUP:
            if _inside(MENU_START_RECT, x, y):
                state['click'] = 'START'
            elif _inside(MENU_EXIT_RECT, x, y):
                state['click'] = 'QUIT'

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, MENU_W, MENU_H)
    cv2.setMouseCallback(window_name, on_mouse)

    action = None
    while action is None:
        frame = art.copy()

        for rect in (MENU_START_RECT, MENU_EXIT_RECT):
            if _inside(rect, *state['mouse']):
                draw_rounded_rect(frame, rect[:2], rect[2:], 12, fill=COLOR_WHITE, fill_alpha=0.09,
                                  border=COLOR_WHITE, border_px=2, border_alpha=0.22)

        if status_message:
            lines = wrap_text(status_message, 540, 0.52, 1, FONT_SMALL)
            if len(lines) > 2:
                lines = [lines[0], lines[1].rstrip('. ')[:-3] + '...' if len(lines[1]) > 3 else lines[1]]
            box_h = 18 + 22 * len(lines)
            draw_rounded_rect(frame, (260, 28), (850, 28 + box_h), 12, fill=COLOR_CARD_BG, fill_alpha=0.9,
                              border=(44, 144, 255), border_px=2)
            for i, line in enumerate(lines):
                put_text(frame, line, (555, 28 + 17 + 22 * i + 8), 0.52, COLOR_WHITE, 1, FONT_SMALL, align='center')

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(30) & 0xFF

        if key == 27 or is_window_closed(window_name):
            action = 'QUIT'
        elif key in (13, 32):
            action = 'START'
        elif state['click']:
            action = state['click']

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return action


# ==============================================================================
# LIVE RECOGNITION SCREEN
# ==============================================================================
class LiveScreen:
    """
    Full-bleed live view: mirrored camera feed, hand skeleton, logo (top-right), Translation and
    Emotion cards (bottom-left), Help button + confidence scale (bottom-right), Full screen / Menu
    buttons. Drawn on a fixed UI_W x UI_H canvas; OpenCV scales the window, so the layout scales too.

    Controls: H = help, F = full screen, ESC = back to menu. Every button is also clickable.
    """

    def __init__(self, window_name):
        self.window_name = window_name
        self.help_open = False
        self.fullscreen = False
        self.mouse = (-1, -1)
        self._pending = set()
        self._gradient_cache = None
        self.layout = self._compute_layout()
        self._vignette = self._build_vignette()

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window_name, UI_W, UI_H)
        cv2.setMouseCallback(window_name, self._on_mouse)
        if START_FULLSCREEN:
            self._pending.add('fullscreen')

    # ---------- layout ----------
    def _compute_layout(self):
        W, H = UI_W, UI_H
        L = {}
        right = int(W * 0.97)
        top = int(H * 0.03)

        word_w, _ = text_size("SAIgn", 1.05, 2)
        L['logo_top'] = top
        L['logo_word_right'] = right
        L['logo_icon_x'] = right - word_w - 12 - 50

        def button_width(label):
            return 14 + 16 + 7 + text_size(label, 0.5, 1, FONT_SMALL)[0] + 14

        by1, bh = top + 50 + 14, 34
        menu_w, full_w = button_width("Menu"), button_width("Full screen")
        L['menu_btn'] = (right - menu_w, by1, right, by1 + bh)
        L['full_btn'] = (right - menu_w - 7 - full_w, by1, right - menu_w - 7, by1 + bh)

        card_x, card_w, bottom = int(W * 0.025), int(W * 0.68), H - int(H * 0.04)
        L['emotion_card'] = (card_x, bottom - 80, card_x + card_w, bottom)
        L['translation_card'] = (card_x, bottom - 80 - 14 - 98, card_x + card_w, bottom - 80 - 14)

        help_right = W - int(W * 0.025)
        help_w = 25 + 30 + 12 + text_size("Help", 0.85, 2)[0] + 25
        L['help_btn'] = (help_right - help_w, bottom - 58, help_right, bottom)

        pw, pad = 360, 18
        bar_y = pad + 54
        L['pop_offsets'] = {'title': pad + 18, 'sub': pad + 40, 'bar': bar_y,
                            'lab1': bar_y + 30 + 22, 'lab2': bar_y + 30 + 37, 'foot': bar_y + 30 + 63}
        L['pop_pad'] = pad
        L['foot_lines'] = wrap_text(CONFIDENCE_FOOTNOTE, pw - 2 * pad, 0.42, 1, FONT_SMALL)
        ph = L['pop_offsets']['foot'] + (len(L['foot_lines']) - 1) * 17 + 6 + pad
        L['help_popover'] = (help_right - pw, L['help_btn'][1] - 12 - ph, help_right, L['help_btn'][1] - 12)
        return L

    def _build_vignette(self):
        """Top/bottom darkening that keeps white overlay text readable on any background."""
        top_h, bot_start = int(UI_H * 0.31), int(UI_H * 0.55)
        top_alpha = np.linspace(0.69, 0.0, top_h)
        bot_alpha = np.linspace(0.0, 0.60, UI_H - bot_start)

        def factor(alpha):
            col = ((1.0 - alpha) * 255).astype(np.uint8)[:, None, None]
            return np.ascontiguousarray(np.broadcast_to(col, (len(alpha), UI_W, 3)))

        return {'top': (top_h, factor(top_alpha)), 'bottom': (bot_start, factor(bot_alpha))}

    # ---------- input ----------
    def _on_mouse(self, event, x, y, flags, param):
        self.mouse = (x, y)
        if event != cv2.EVENT_LBUTTONUP:
            return
        if _inside(self.layout['help_btn'], x, y):
            self._pending.add('help')
        elif _inside(self.layout['full_btn'], x, y):
            self._pending.add('fullscreen')
        elif _inside(self.layout['menu_btn'], x, y):
            self._pending.add('menu')

    def toggle_help(self):
        self.help_open = not self.help_open

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        try:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
        except cv2.error:
            self.fullscreen = False

    def _apply_pending(self):
        """Runs actions queued by the mouse callback. Returns True if 'back to menu' was requested."""
        pending, self._pending = self._pending, set()
        if 'help' in pending:
            self.toggle_help()
        if 'fullscreen' in pending:
            self.toggle_fullscreen()
        return 'menu' in pending

    def handle_key(self, key):
        """Processes a keypress plus any queued mouse clicks. Returns True to go back to the menu."""
        back = self._apply_pending()
        if key == 27:
            back = True
        elif key in (ord('h'), ord('H')):
            self.toggle_help()
        elif key in (ord('f'), ord('F')):
            self.toggle_fullscreen()
        return back

    # ---------- rendering ----------
    def render(self, frame, results, sign_text, confidence, expression_text, live=True):
        """
        frame: raw BGR camera frame (NOT mirrored; the model must keep seeing unmirrored landmarks).
        confidence: 0..1, or None while waiting for a sign.
        Returns the finished UI_H x UI_W BGR canvas.
        """
        self._apply_pending()

        canvas, transform = self._cover_resize(frame)
        if MIRROR_VIEW:
            canvas = cv2.flip(canvas, 1)

        self._draw_landmarks(canvas, results, transform)
        self._apply_vignette(canvas)
        self._draw_title(canvas, live)
        self._draw_logo_and_buttons(canvas)
        self._draw_info_cards(canvas, sign_text, confidence, expression_text)
        self._draw_help(canvas, confidence)
        cv2.rectangle(canvas, (0, 0), (UI_W - 1, UI_H - 1), COLOR_YELLOW, 10, cv2.LINE_AA)
        return canvas

    @staticmethod
    def _cover_resize(frame):
        """CSS 'object-fit: cover': scale to fill the canvas and centre-crop."""
        src_h, src_w = frame.shape[:2]
        scale = max(UI_W / src_w, UI_H / src_h)
        rw, rh = max(UI_W, round(src_w * scale)), max(UI_H, round(src_h * scale))
        resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
        x0, y0 = (rw - UI_W) // 2, (rh - UI_H) // 2
        canvas = np.ascontiguousarray(resized[y0:y0 + UI_H, x0:x0 + UI_W])
        return canvas, (rw, rh, x0, y0)

    @staticmethod
    def _map_point(transform, nx, ny):
        rw, rh, x0, y0 = transform
        x = nx * rw - x0
        if MIRROR_VIEW:
            x = UI_W - 1 - x
        return int(round(x)), int(round(ny * rh - y0))

    def _draw_landmarks(self, canvas, results, transform):
        for hand in (results.left_hand_landmarks, results.right_hand_landmarks):
            if hand is None:
                continue
            pts = [self._map_point(transform, lm.x, lm.y) for lm in hand.landmark]
            if len(pts) != 21:
                continue
            for a, b in HAND_EDGES:
                cv2.line(canvas, pts[a], pts[b], COLOR_HAND_LINE, 3, cv2.LINE_AA)
            for p in pts:
                cv2.circle(canvas, p, 5, COLOR_HAND_DOT, -1, cv2.LINE_AA)
                cv2.circle(canvas, p, 5, COLOR_HAND_RING, 1, cv2.LINE_AA)

        if DRAW_POSE and results.pose_landmarks:
            pose_pts = [self._map_point(transform, lm.x, lm.y) for lm in results.pose_landmarks.landmark]
            for a, b in mp.solutions.pose.POSE_CONNECTIONS:
                cv2.line(canvas, pose_pts[a], pose_pts[b], (200, 200, 200), 2, cv2.LINE_AA)

    def _apply_vignette(self, canvas):
        for start, factor in ((0, self._vignette['top'][1]), (self._vignette['bottom'][0], self._vignette['bottom'][1])):
            region = canvas[start:start + factor.shape[0]]
            region[:] = cv2.multiply(region, factor, scale=1.0 / 255.0)

    def _draw_title(self, canvas, live):
        x, y = int(UI_W * 0.03), int(UI_H * 0.032)
        title = "SAIgn Vision Engine"
        _, th = text_size(title, 1.1, 2)
        cv2.rectangle(canvas, (x, y), (x + 5, y + th + 16), COLOR_YELLOW, -1)
        put_text(canvas, title, (x + 18, y + th + 6), 1.1, COLOR_WHITE, 2, shadow=True)

        label = "LIVE" if live else "OFFLINE"
        label_w, label_h = text_size(label, 0.55, 2)
        bx, by = x, y + th + 16 + 20
        bw = 14 + 13 + 9 + label_w + 16
        draw_rounded_rect(canvas, (bx, by), (bx + bw, by + 34), 17, fill=(8, 8, 5), fill_alpha=0.86,
                          border=COLOR_WHITE, border_px=1, border_alpha=0.21)
        dot_c = (bx + 14 + 6, by + 17)
        cv2.circle(canvas, dot_c, 6, COLOR_LIVE if live else COLOR_OFFLINE, -1, cv2.LINE_AA)
        put_text(canvas, label, (bx + 14 + 13 + 9, by + 17 + label_h // 2), 0.55, COLOR_WHITE, 2)

    def _draw_logo_and_buttons(self, canvas):
        L = self.layout
        ix, iy = L['logo_icon_x'], L['logo_top']
        draw_rounded_rect(canvas, (ix, iy), (ix + 50, iy + 50), 13, fill=COLOR_YELLOW)
        put_text(canvas, "S", (ix + 25, iy + 37), 1.2, COLOR_INK, 3, align='center')

        word_x = ix + 50 + 12
        base_y = iy + 37
        w = put_text(canvas, "SAI", (word_x, base_y), 1.05, COLOR_WHITE, 2, shadow=True)
        put_text(canvas, "gn", (word_x + w, base_y), 1.05, COLOR_YELLOW, 2, shadow=True)

        for key, label, icon in (('full_btn', "Full screen", _icon_fullscreen), ('menu_btn', "Menu", _icon_back)):
            x1, y1, x2, y2 = L[key]
            hovered = _inside(L[key], *self.mouse)
            draw_rounded_rect(canvas, (x1, y1), (x2, y2), 9,
                              fill=(30, 22, 17) if hovered else (20, 14, 11), fill_alpha=0.85,
                              border=COLOR_YELLOW if hovered else COLOR_WHITE, border_px=1,
                              border_alpha=1.0 if hovered else 0.38)
            icon(canvas, x1 + 14 + 8, (y1 + y2) // 2, COLOR_WHITE)
            put_text(canvas, label, (x1 + 14 + 16 + 7, (y1 + y2) // 2 + 5), 0.5, COLOR_WHITE, 1, FONT_SMALL)

    def _draw_card(self, canvas, rect, icon, title, value, value_color, sub_text=None, sub_color=COLOR_MUTED):
        x1, y1, x2, y2 = rect
        cy = (y1 + y2) // 2
        draw_rounded_rect(canvas, (x1, y1), (x2, y2), 17, fill=COLOR_CARD_BG, fill_alpha=0.91,
                          border=COLOR_WHITE, border_px=2, border_alpha=0.35)
        icon(canvas, x1 + 24 + 23, cy, COLOR_YELLOW)
        rule_x = x1 + 24 + 46 + 20
        cv2.rectangle(canvas, (rule_x, cy - 24), (rule_x + 3, cy + 24), COLOR_YELLOW, -1)

        title_x = rule_x + 4 + 20
        _, title_h = text_size(title, 0.75, 1)
        put_text(canvas, title, (title_x, cy + title_h // 2), 0.75, COLOR_WHITE, 1)

        value_x = title_x + 154 + 20
        value_text, value_scale = fit_text(value, x2 - 24 - value_x, 1.0, 2)
        _, value_h = text_size(value_text, value_scale, 2)
        if sub_text:
            put_text(canvas, value_text, (value_x, cy - 2), value_scale, value_color, 2)
            put_text(canvas, sub_text, (value_x, cy + 26), 0.5, sub_color, 1, FONT_SMALL)
        else:
            put_text(canvas, value_text, (value_x, cy + value_h // 2), value_scale, value_color, 2)

    def _draw_info_cards(self, canvas, sign_text, confidence, expression_text):
        if confidence is None:
            sub_text, sub_color = "Confidence: --", COLOR_MUTED
        else:
            sub_text, sub_color = f"Confidence: {confidence * 100:.1f}%", get_confidence_color(confidence)
        self._draw_card(canvas, self.layout['translation_card'], _icon_translation, "Translation:",
                        sign_text, COLOR_YELLOW_TEXT, sub_text, sub_color)
        self._draw_card(canvas, self.layout['emotion_card'], _icon_emotion, "Emotion:",
                        expression_text, COLOR_GREEN_TEXT)

    def _confidence_gradient(self, width, height):
        if self._gradient_cache and self._gradient_cache[0] == (width, height):
            return self._gradient_cache[1], self._gradient_cache[2]
        xs = np.linspace(0.0, 1.0, width)
        stop_x = [s for s, _ in CONFIDENCE_SCALE_STOPS]
        row = np.stack([np.interp(xs, stop_x, [c[i] for _, c in CONFIDENCE_SCALE_STOPS])
                        for i in range(3)], axis=1).astype(np.float32)
        bar = np.repeat(row[None, :, :], height, axis=0)
        mask = np.zeros((height, width, 3), dtype=np.uint8)
        draw_rounded_rect(mask, (0, 0), (width - 1, height - 1), height // 2, fill=COLOR_WHITE)
        mask = mask.astype(np.float32) / 255.0
        self._gradient_cache = ((width, height), bar, mask)
        return bar, mask

    def _draw_help(self, canvas, confidence):
        L = self.layout
        x1, y1, x2, y2 = L['help_btn']
        hovered = _inside(L['help_btn'], *self.mouse)
        draw_rounded_rect(canvas, (x1, y1), (x2, y2), 16, fill=(28, 26, 18) if hovered else (15, 11, 10),
                          fill_alpha=0.93, border=COLOR_YELLOW, border_px=3)
        cy = (y1 + y2) // 2
        cv2.circle(canvas, (x1 + 25 + 15, cy), 15, COLOR_YELLOW, 2, cv2.LINE_AA)
        put_text(canvas, "i", (x1 + 25 + 15, cy + 7), 0.7, COLOR_YELLOW, 2, align='center')
        _, help_h = text_size("Help", 0.85, 2)
        put_text(canvas, "Help", (x1 + 25 + 30 + 12, cy + help_h // 2), 0.85, COLOR_WHITE, 2)

        if not self.help_open:
            return

        px1, py1, px2, py2 = L['help_popover']
        pad, off = L['pop_pad'], L['pop_offsets']
        draw_rounded_rect(canvas, (px1, py1), (px2, py2), 20, fill=(16, 11, 9), fill_alpha=0.97,
                          border=COLOR_YELLOW, border_px=2)
        center_x = (px1 + px2) // 2
        put_text(canvas, "Confidence Scale", (center_x, py1 + off['title']), 0.8, COLOR_WHITE, 2, align='center')
        put_text(canvas, "Model recognition confidence", (center_x, py1 + off['sub']), 0.45, COLOR_MUTED, 1,
                 FONT_SMALL, align='center')

        bar_x, bar_y, bar_w, bar_h = px1 + pad, py1 + off['bar'], (px2 - px1) - 2 * pad, 30
        bar, mask = self._confidence_gradient(bar_w, bar_h)
        roi = canvas[bar_y:bar_y + bar_h, bar_x:bar_x + bar_w]
        roi[:] = (bar * mask + roi.astype(np.float32) * (1.0 - mask)).astype(np.uint8)

        if confidence is not None:
            pointer_x = bar_x + int(round(max(0.0, min(1.0, confidence)) * (bar_w - 1)))
            cv2.line(canvas, (pointer_x, bar_y - 6), (pointer_x, bar_y + bar_h + 4), (0, 0, 0), 5, cv2.LINE_AA)
            cv2.line(canvas, (pointer_x, bar_y - 6), (pointer_x, bar_y + bar_h + 4), COLOR_WHITE, 3, cv2.LINE_AA)

        for i, (rng, name) in enumerate(CONFIDENCE_SCALE_LABELS):
            col_x = bar_x + int(bar_w * (i + 0.5) / len(CONFIDENCE_SCALE_LABELS))
            put_text(canvas, rng, (col_x, py1 + off['lab1']), 0.42, COLOR_WHITE, 1, FONT_SMALL, align='center')
            put_text(canvas, name, (col_x, py1 + off['lab2']), 0.42, COLOR_WHITE, 1, FONT_SMALL, align='center')

        for i, line in enumerate(L['foot_lines']):
            put_text(canvas, line, (bar_x, py1 + off['foot'] + i * 17), 0.42, COLOR_SUBTLE, 1, FONT_SMALL)


def run_recognition_engine(model, trained_labels, gesture_cache, expression_thresholds, sequence_length):
    window_name = 'SAIgn - Live Recognition Engine'
    labels = trained_labels if trained_labels else ['UNKNOWN']

    if not hasattr(mp, 'solutions'):
        raise ImportError(
            "This MediaPipe build does not expose the legacy 'mp.solutions' API. "
            "Install a version that does, e.g.: pip install mediapipe==0.10.14"
        )
    mp_holistic = mp.solutions.holistic

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam (camera index 0). "
            "Verify that no other app is using the camera and macOS Camera permissions are granted."
        )

    raw_buffer = deque(maxlen=max(RAW_CAPTURE_WINDOW, sequence_length))
    prediction_history = deque(maxlen=DEBOUNCE_WINDOW_SIZE)

    current_gesture = WAITING_TEXT
    display_confidence = 0.0
    no_hand_frames = 0
    consecutive_errors = 0

    try:
        ui = LiveScreen(window_name)
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[WARN] Failed to read frame from webcam. Ending capture session.")
                    break
                if frame.size == 0:
                    continue

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                try:
                    results = holistic.process(image)
                    consecutive_errors = 0
                except Exception as e:
                    consecutive_errors += 1
                    print(f"[WARN] MediaPipe processing error ({consecutive_errors}/5): {e}")
                    if consecutive_errors >= 5:
                        print("[ERROR] Too many consecutive MediaPipe errors. Ending capture session.")
                        break
                    try:
                        holistic.reset()
                    except Exception:
                        pass
                    continue

                hands_visible = bool(results.left_hand_landmarks or results.right_hand_landmarks)
                if hands_visible:
                    no_hand_frames = 0
                else:
                    no_hand_frames += 1

                keypoints = extract_landmarks(results)
                raw_buffer.append(keypoints)

                if no_hand_frames >= VOTE_CLEAR_NO_HAND_FRAMES:
                    prediction_history.clear()

                if no_hand_frames >= NO_HAND_RESET_FRAMES:
                    current_gesture = WAITING_TEXT
                    display_confidence = 0.0
                elif len(raw_buffer) >= sequence_length and hands_visible:
                    aligned_seq = prepare_live_sequence(raw_buffer, target_frames=sequence_length)
                    input_tensor = np.expand_dims(aligned_seq, axis=0)
                    res = np.asarray(model(input_tensor, training=False))[0]
                    prediction_idx = int(np.argmax(res))
                    frame_confidence = float(res[prediction_idx])

                    raw_label = labels[prediction_idx] if prediction_idx < len(labels) else "UNKNOWN"

                    cached_data = gesture_cache.get(raw_label, {
                        'display_name': raw_label.replace('_', ' ').title(),
                        'min_score': 0.40
                    })

                    if frame_confidence >= cached_data['min_score']:
                        frame_candidate = cached_data['display_name']
                    else:
                        frame_candidate = "Uncertain"

                    prediction_history.append((frame_candidate, frame_confidence))

                    vote_counts = Counter(cand for cand, _ in prediction_history)
                    most_common_gesture, vote_count = vote_counts.most_common(1)[0]

                    if vote_count >= DEBOUNCE_CONSENSUS_COUNT:
                        current_gesture = most_common_gesture
                        matching_scores = [
                            score for cand, score in prediction_history
                            if cand == most_common_gesture
                        ]
                        display_confidence = float(np.mean(matching_scores)) if matching_scores else frame_confidence

                if not results.face_landmarks:
                    expression_text = "Face not visible"
                elif not expression_thresholds:
                    expression_text = "Not configured"
                else:
                    active_expressions = detect_expressions(
                        results, expression_thresholds,
                        image_size=(frame.shape[1], frame.shape[0])
                    )
                    expression_text = ", ".join(active_expressions[:2]) if active_expressions else "Neutral"

                shown_confidence = None if current_gesture == WAITING_TEXT else display_confidence
                cv2.imshow(window_name, ui.render(frame, results, current_gesture, shown_confidence, expression_text))

                key = cv2.waitKey(10) & 0xFF
                if ui.handle_key(key) or is_window_closed(window_name):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


# ==============================================================================
# MAIN PROGRAM EXECUTION ENTRY POINT
# ==============================================================================
def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Please run your training script to generate the trained model."
        )

    model = load_model(MODEL_PATH, compile=False)  # inference only; avoids optimizer/loss deserialization issues

    input_shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
    output_shape = model.output_shape[0] if isinstance(model.output_shape, list) else model.output_shape

    seq_len = input_shape[1] or SEQUENCE_LENGTH
    feature_dim = input_shape[2]
    if feature_dim != EXPECTED_FEATURE_DIM:
        raise ValueError(
            f"Model feature dimension mismatch: '{MODEL_PATH}' expects {feature_dim} features per frame, "
            f"but Holistic extraction produces {EXPECTED_FEATURE_DIM}. Re-extract features and retrain the model."
        )

    model_num_classes = output_shape[-1]

    menu_message = None
    while True:
        trained_labels, gesture_cache, expression_thresholds = load_db_cache()
        if trained_labels and len(trained_labels) != model_num_classes:
            print(
                f"[WARN] Loaded {len(trained_labels)} class label(s), "
                f"but model outputs {model_num_classes} class(es). Retrain the model if labels changed."
            )

        action = show_main_menu(menu_message)
        menu_message = None

        if action == 'QUIT':
            print("Exiting SAIgn Engine cleanly. Goodbye!")
            break
        elif action == 'START':
            try:
                run_recognition_engine(model, trained_labels, gesture_cache, expression_thresholds, seq_len)
            except RuntimeError as e:
                # e.g. webcam unavailable: report and return to the menu instead of crashing
                print(f"[ERROR] {e}")
                menu_message = str(e)


if __name__ == "__main__":
    main()