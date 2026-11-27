import json
import operator
import os
import sqlite3
import sys
from collections import deque, Counter
import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp

# --- System & Path Configurations ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:  # make the sibling P4_extract_features.py importable from any working dir
    sys.path.insert(0, SCRIPT_DIR)

# Live inference MUST preprocess its rolling frame window exactly the way training preprocessed clips,
# or the model sees a systematically different encoding at inference time than it was trained on
# (this silently hurt accuracy on low-motion/static signs before). Importing the same functions
# training uses, instead of a separate hand-reimplementation, makes that drift impossible. The same goes for
# the per-frame landmark layout (extract_frame_landmarks).
from P4_extract_features import (  # noqa: E402
    apply_velocity_segmentation,
    extract_frame_landmarks,
    resample_sequence,
    TARGET_FRAMES,
    FEATURE_DIM,
)

# The user interface lives in its own module (P4_ui.py) so it can be restyled/previewed independently.
from P4_ui import (  # noqa: E402
    LiveScreen,
    UNCERTAIN_TEXT,
    WAITING_TEXT,
    is_window_closed,
    show_main_menu,
)

# ==============================================================================
# PATHS & SYSTEM CONFIGURATION
# ==============================================================================
DB_PATH = os.path.join(SCRIPT_DIR, 'saign_vision.db')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'asl_model.keras')
CLASSES_PATH = os.path.join(SCRIPT_DIR, 'classes.json')

SEQUENCE_LENGTH = TARGET_FRAMES          # kept in sync with P4_extract_features.py by import, not by copy
EXPECTED_FEATURE_DIM = FEATURE_DIM       # ditto
RAW_CAPTURE_WINDOW = 45  # ~1.5s rolling capture buffer before velocity segmentation + resampling
EXPRESSION_SEED_VERSION = 2  # Must match P4_setup_db.py. v2 = face-size-normalised ratio thresholds.

DEBOUNCE_WINDOW_SIZE = 8
DEBOUNCE_CONSENSUS_COUNT = 5
VOTE_CLEAR_NO_HAND_FRAMES = 5   # Flush stale vote history after ~0.17s without hands
NO_HAND_RESET_FRAMES = 15       # Reset active sign HUD after ~0.5s with no hands visible

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


def prepare_live_sequence(raw_buffer, target_frames=SEQUENCE_LENGTH):
    """
    Applies the EXACT SAME preprocessing training used (apply_velocity_segmentation then
    resample_sequence, both imported from P4_extract_features.py) to the live rolling window,
    so a static/low-motion sign is encoded the same way live as it was during training -- instead
    of a separately-written approximation that can silently drift out of sync with training.
    """
    features = np.array(raw_buffer, dtype=np.float32)
    segmented = apply_velocity_segmentation(features)
    resampled = resample_sequence(segmented, target_frames=target_frames)
    return np.nan_to_num(resampled, nan=0.0, posinf=0.0, neginf=0.0)


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

                keypoints = extract_frame_landmarks(results)
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
                        frame_candidate = UNCERTAIN_TEXT

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
                cv2.imshow(window_name, ui.render(frame, results, current_gesture, shown_confidence, expression_text,
                                                 hands_visible=hands_visible))

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

        system_info = {
            'signs': len(trained_labels) or model_num_classes,
            'expressions': bool(expression_thresholds),
            'mismatch': bool(trained_labels) and len(trained_labels) != model_num_classes,
        }
        action = show_main_menu(menu_message, system_info)
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