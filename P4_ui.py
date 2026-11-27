"""
P4_ui.py -- SAIgn user interface (OpenCV): animated main menu + full-bleed live recognition screen.

Depends only on OpenCV + NumPy (MediaPipe is imported lazily, and only for the optional pose overlay), so the
UI can be previewed and restyled without the model, database or webcam logic:

    python P4_ui.py          # opens the main menu on its own (Start/Exit just report what you clicked)

P4_run_system.py imports show_main_menu / LiveScreen / is_window_closed and the two label constants from here.
Edit the options below to change look and behaviour; the recognition logic lives in P4_run_system.py.
"""
import os
import time
import unicodedata
from collections import deque

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MENU_ART_PATH = os.path.join(SCRIPT_DIR, 'saign_menu.png')  # main-menu artwork (falls back to a drawn menu)

# --- UI options ---
UI_W, UI_H = 1280, 720          # logical size of the live screen; the OpenCV window scales it
WAITING_TEXT = "Waiting for signs"
MIRROR_VIEW = True              # show the camera like a mirror (the model still sees the unmirrored frame)
START_FULLSCREEN = False        # open the live screen full screen (F toggles it either way)
DRAW_POSE = False               # also draw the body-pose skeleton (the notebook UI shows hands only)
TRANSCRIPT_MAX = 12             # confirmed signs kept in the on-screen transcript
UNCERTAIN_TEXT = "Uncertain"     # label shown when the model is below a sign's min_score


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
COLOR_ORANGE = (44, 144, 255)        # #ff902c  warnings / uncertain
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

HEADER_KEY_HINTS = {'full_btn': 'F', 'menu_btn': 'Esc'}
SHORTCUTS_TEXT = "Keys: H help - F full screen - C clear transcript - Esc menu"
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


def _icon_transcript(img, cx, cy, color):
    draw_rounded_rect(img, (cx - 13, cy - 11), (cx + 13, cy + 7), 5, border=color, border_px=2)
    cv2.fillPoly(img, [np.array([[cx - 6, cy + 7], [cx - 10, cy + 14], [cx + 1, cy + 7]], np.int32)], color, cv2.LINE_AA)
    cv2.line(img, (cx - 7, cy - 5), (cx + 7, cy - 5), color, 2, cv2.LINE_AA)
    cv2.line(img, (cx - 7, cy), (cx + 2, cy), color, 2, cv2.LINE_AA)


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
    """Returns (image, is_custom_artwork). The animation overlays only make sense on the real artwork."""
    if os.path.exists(MENU_ART_PATH):
        art = cv2.imread(MENU_ART_PATH, cv2.IMREAD_COLOR)
        if art is not None:
            if art.shape[1] != MENU_W or art.shape[0] != MENU_H:
                art = cv2.resize(art, (MENU_W, MENU_H), interpolation=cv2.INTER_AREA)
            return art, True
        print(f"[WARN] Could not read {MENU_ART_PATH}. Using the built-in menu layout instead.")
    else:
        print(f"[INFO] {os.path.basename(MENU_ART_PATH)} not found next to this script. "
              "Using the built-in menu layout.")
    return _build_fallback_menu(), False


# Geometry of the hand illustration baked into saign_menu.png (used for the animated overlays).
PREVIEW_CENTER, PREVIEW_RADIUS = (815, 363), 133
FINGERTIPS = [(718, 349), (774, 281), (824, 264), (863, 291), (901, 331)]
MENU_ZONES = {'START': MENU_START_RECT, 'QUIT': MENU_EXIT_RECT}


def _draw_chip(img, right_x, cy, text, dot_color):
    """Small status pill, right-aligned at right_x. Returns its width."""
    text_w, _ = text_size(text, 0.5, 1, FONT_SMALL)
    width = 30 + text_w + 14
    x1 = right_x - width
    draw_rounded_rect(img, (x1, cy - 15), (right_x, cy + 15), 15, fill=COLOR_WHITE, fill_alpha=0.06,
                      border=COLOR_WHITE, border_px=1, border_alpha=0.16)
    cv2.circle(img, (x1 + 17, cy), 4, dot_color, -1, cv2.LINE_AA)
    put_text(img, text, (x1 + 30, cy + 5), 0.5, COLOR_WHITE, 1, FONT_SMALL)
    return width


def _paint_menu_footer(art, info):
    """Replaces the footer bar with 'ASL Transcriber' plus live system-status chips."""
    draw_rounded_rect(art, (50, 618), (1070, 686), 14, fill=(45, 29, 17), border=(72, 51, 38), border_px=1)
    put_text(art, "ASL Transcriber", (80, 661), 0.85, COLOR_WHITE, 2)

    chips = [("Model ready", COLOR_LIVE)]
    if info.get('signs'):
        chips.append((f"{info['signs']} signs", COLOR_YELLOW))
    chips.append(("Expressions on", COLOR_LIVE) if info.get('expressions') else ("Expressions off", COLOR_ORANGE))
    if info.get('mismatch'):
        chips.append(("Label count mismatch", COLOR_ORANGE))
    x = 1040
    for text, color in reversed(chips):
        x -= _draw_chip(art, x, 652, text, color) + 8


def _animate_menu_preview(frame, t):
    """Scan line sweeping the tracking circle + pulsing fingertip rings (cheap ROI blends)."""
    cx, cy = PREVIEW_CENTER
    r = PREVIEW_RADIUS
    phase = (t * 0.45) % 1.0
    for lag, alpha in ((0.0, 0.55), (0.03, 0.26), (0.06, 0.12)):
        y = int(cy - r + ((phase - lag) % 1.0) * 2 * r)
        half = int(np.sqrt(max(0.0, r * r - (y - cy) ** 2)))
        if half > 6:
            draw_rounded_rect(frame, (cx - half, y - 1), (cx + half, y + 1), 1, fill=(206, 235, 100), fill_alpha=alpha)
    for i, (px, py) in enumerate(FINGERTIPS):
        p = (t * 0.8 + i * 0.18) % 1.0
        rad = int(7 + 13 * p)
        draw_rounded_rect(frame, (px - rad, py - rad), (px + rad, py + rad), rad,
                          border=COLOR_YELLOW, border_px=2, border_alpha=0.55 * (1.0 - p))


def _render_menu_frame(art, custom_art, t, mouse=(-1, -1), pressed=None, status_message=None):
    """One frame of the menu at animation time t (seconds). Pure drawing, so it is easy to preview/test."""
    frame = art.copy()
    if custom_art:
        _animate_menu_preview(frame, t)

    x1, y1, x2, y2 = MENU_START_RECT
    glow = 0.18 + 0.16 * np.sin(t * 2.4)                       # 'breathing' call-to-action
    draw_rounded_rect(frame, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), 18, border=COLOR_YELLOW, border_px=3,
                      border_alpha=float(glow))

    for name, rect in MENU_ZONES.items():
        if _inside(rect, *mouse):
            held = pressed == name
            draw_rounded_rect(frame, rect[:2], rect[2:], 12, fill=(0, 0, 0) if held else COLOR_WHITE,
                              fill_alpha=0.20 if held else 0.09, border=COLOR_WHITE, border_px=2, border_alpha=0.22)

    if status_message:
        lines = wrap_text(status_message, 540, 0.52, 1, FONT_SMALL)
        if len(lines) > 2:
            lines = [lines[0], lines[1].rstrip('. ')[:-3] + '...' if len(lines[1]) > 3 else lines[1]]
        box_h = 18 + 22 * len(lines)
        draw_rounded_rect(frame, (260, 28), (850, 28 + box_h), 12, fill=COLOR_CARD_BG, fill_alpha=0.9,
                          border=COLOR_ORANGE, border_px=2)
        for i, line in enumerate(lines):
            put_text(frame, line, (555, 28 + 17 + 22 * i + 8), 0.52, COLOR_WHITE, 1, FONT_SMALL, align='center')

    if t < 0.35:                                                 # quick fade-in
        frame = (frame * (t / 0.35)).astype(np.uint8)
    return frame


def show_main_menu(status_message=None, system_info=None):
    """
    Main menu. Start: click the button or press ENTER / SPACE. Exit: click Exit, press ESC, or close
    the window. Returns 'START' or 'QUIT'.

    status_message : e.g. a camera error, shown as a banner along the top.
    system_info    : {'signs': int, 'expressions': bool, 'mismatch': bool} shown as footer status chips.
    """
    window_name = 'SAIgn - Main Menu'
    art, custom_art = _load_menu_art()
    _paint_menu_footer(art, system_info or {})
    state = {'mouse': (-1, -1), 'down': None, 'click': None}

    def zone_at(x, y):
        return next((name for name, rect in MENU_ZONES.items() if _inside(rect, x, y)), None)

    def on_mouse(event, x, y, flags, param):
        state['mouse'] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            state['down'] = zone_at(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            zone = zone_at(x, y)
            if zone and zone == state['down']:      # a click = press and release on the same button
                state['click'] = zone
            state['down'] = None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, MENU_W, MENU_H)
    cv2.setMouseCallback(window_name, on_mouse)

    action, t0 = None, time.time()
    while action is None:
        cv2.imshow(window_name, _render_menu_frame(art, custom_art, time.time() - t0, state['mouse'],
                                                   state['down'], status_message))
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
        self.transcript = deque(maxlen=TRANSCRIPT_MAX)
        self._last_logged = None
        self._current_sign = None
        self._t0 = time.time()
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

        def button_width(label, hint):
            return (14 + 16 + 7 + text_size(label, 0.5, 1, FONT_SMALL)[0] + 9
                    + text_size(hint, 0.4, 1, FONT_SMALL)[0] + 10 + 10)

        by1, bh = top + 50 + 14, 34
        menu_w, full_w = button_width("Menu", "Esc"), button_width("Full screen", "F")
        L['menu_btn'] = (right - menu_w, by1, right, by1 + bh)
        L['full_btn'] = (right - menu_w - 7 - full_w, by1, right - menu_w - 7, by1 + bh)

        card_x, card_w, bottom = int(W * 0.025), int(W * 0.68), H - int(H * 0.04)
        L['emotion_card'] = (card_x, bottom - 80, card_x + card_w, bottom)
        L['translation_card'] = (card_x, bottom - 80 - 14 - 98, card_x + card_w, bottom - 80 - 14)
        tr_top = L['translation_card'][1]
        L['transcript_card'] = (card_x, tr_top - 14 - 64, card_x + card_w, tr_top - 14)

        help_right = W - int(W * 0.025)
        help_w = 25 + 30 + 12 + text_size("Help", 0.85, 2)[0] + 25
        L['help_btn'] = (help_right - help_w, bottom - 58, help_right, bottom)

        pw, pad = 360, 18
        bar_y = pad + 54
        L['pop_offsets'] = {'title': pad + 18, 'sub': pad + 40, 'bar': bar_y,
                            'lab1': bar_y + 30 + 22, 'lab2': bar_y + 30 + 37, 'foot': bar_y + 30 + 63}
        L['pop_pad'] = pad
        L['foot_lines'] = wrap_text(CONFIDENCE_FOOTNOTE, pw - 2 * pad, 0.42, 1, FONT_SMALL)
        L['key_lines'] = wrap_text(SHORTCUTS_TEXT, pw - 2 * pad, 0.42, 1, FONT_SMALL)
        L['pop_offsets']['keys'] = L['pop_offsets']['foot'] + (len(L['foot_lines']) - 1) * 17 + 26
        ph = L['pop_offsets']['keys'] + (len(L['key_lines']) - 1) * 17 + 6 + pad
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
        elif key in (ord('c'), ord('C')):
            self.clear_transcript()
        return back

    def clear_transcript(self):
        """Empties the transcript. A sign that is still being held is NOT re-added until it changes or hands leave."""
        self.transcript.clear()
        held = self._current_sign
        self._last_logged = held if held not in (None, WAITING_TEXT, UNCERTAIN_TEXT) else None

    def log_sign(self, sign_text):
        """Adds newly confirmed signs to the transcript. Hands leaving the frame re-arms repeats."""
        self._current_sign = sign_text
        if sign_text == WAITING_TEXT:
            self._last_logged = None
        elif sign_text != UNCERTAIN_TEXT and sign_text != self._last_logged:
            self.transcript.append(sign_text)
            self._last_logged = sign_text

    # ---------- rendering ----------
    def render(self, frame, results, sign_text, confidence, expression_text, live=True, hands_visible=True):
        """
        frame: raw BGR camera frame (NOT mirrored; the model must keep seeing unmirrored landmarks).
        confidence: 0..1, or None while waiting for a sign.
        Returns the finished UI_H x UI_W BGR canvas.
        """
        self._apply_pending()
        self.log_sign(sign_text)

        canvas, transform = self._cover_resize(frame)
        if MIRROR_VIEW:
            canvas = cv2.flip(canvas, 1)

        self._draw_landmarks(canvas, results, transform)
        self._apply_vignette(canvas)
        self._draw_title(canvas, live)
        self._draw_logo_and_buttons(canvas)
        self._draw_info_cards(canvas, sign_text, confidence, expression_text)
        self._draw_help(canvas, confidence)
        if not hands_visible and sign_text == WAITING_TEXT:
            self._draw_hand_hint(canvas)
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
            import mediapipe as mp  # lazy: only the optional pose overlay needs it; keeps this module light
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
        if live:
            pulse = 0.5 + 0.5 * np.sin((time.time() - self._t0) * 3.0)
            halo = int(8 + 5 * pulse)
            draw_rounded_rect(canvas, (dot_c[0] - halo, dot_c[1] - halo), (dot_c[0] + halo, dot_c[1] + halo), halo,
                              fill=COLOR_LIVE, fill_alpha=float(0.05 + 0.30 * (1.0 - pulse)))
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
            hint = HEADER_KEY_HINTS[key]
            chip_x1 = x2 - 10 - (text_size(hint, 0.4, 1, FONT_SMALL)[0] + 10)
            draw_rounded_rect(canvas, (chip_x1, (y1 + y2) // 2 - 10), (x2 - 10, (y1 + y2) // 2 + 10), 5,
                              fill=COLOR_WHITE, fill_alpha=0.16)
            put_text(canvas, hint, (chip_x1 + 5, (y1 + y2) // 2 + 4), 0.4, COLOR_WHITE, 1, FONT_SMALL)

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
        sign_color = COLOR_ORANGE if sign_text == UNCERTAIN_TEXT else COLOR_YELLOW_TEXT

        self._draw_transcript_card(canvas)
        rect = self.layout['translation_card']
        self._draw_card(canvas, rect, _icon_translation, "Translation:", sign_text, sign_color, sub_text, sub_color)
        if confidence is not None:
            self._draw_confidence_meter(canvas, rect, confidence)
        self._draw_card(canvas, self.layout['emotion_card'], _icon_emotion, "Emotion:",
                        expression_text, COLOR_GREEN_TEXT)

    def _draw_confidence_meter(self, canvas, rect, confidence):
        """Thin bar beside the confidence text, coloured from the same scale as the Help gradient."""
        x1, y1, x2, y2 = rect
        mx, mw, my = x1 + 288 + 175, 230, (y1 + y2) // 2 + 21
        draw_rounded_rect(canvas, (mx, my), (mx + mw, my + 8), 4, fill=COLOR_WHITE, fill_alpha=0.16)
        fill_w = max(8, int(mw * max(0.0, min(1.0, confidence))))
        draw_rounded_rect(canvas, (mx, my), (mx + fill_w, my + 8), 4, fill=get_confidence_color(confidence))

    def _draw_transcript_card(self, canvas):
        x1, y1, x2, y2 = self.layout['transcript_card']
        cy = (y1 + y2) // 2
        draw_rounded_rect(canvas, (x1, y1), (x2, y2), 17, fill=COLOR_CARD_BG, fill_alpha=0.91,
                          border=COLOR_WHITE, border_px=2, border_alpha=0.35)
        _icon_transcript(canvas, x1 + 24 + 23, cy - 2, COLOR_YELLOW)
        rule_x = x1 + 24 + 46 + 20
        cv2.rectangle(canvas, (rule_x, cy - 18), (rule_x + 3, cy + 18), COLOR_YELLOW, -1)
        title_x = rule_x + 4 + 20
        _, title_h = text_size("Transcript:", 0.75, 1)
        put_text(canvas, "Transcript:", (title_x, cy + title_h // 2), 0.75, COLOR_WHITE, 1)

        text_x = title_x + 154 + 20
        avail = x2 - 24 - text_x
        words = list(self.transcript)
        if not words:
            put_text(canvas, "Confirmed signs appear here", (text_x, cy + 7), 0.6, COLOR_SUBTLE, 1, FONT_SMALL)
            return

        scale, sep = 0.85, "  /  "
        sep_w = text_size(sep, scale, 1)[0]
        reserve = text_size("...", scale, 1)[0] + 12          # room for the 'older words trimmed' ellipsis
        shown, used = [], 0
        for word in reversed(words):                           # keep the most recent words that fit
            need = text_size(word, scale, 1)[0] + (sep_w if shown else 0)
            if used + need > avail - reserve and shown:
                break
            shown.insert(0, word)
            used += need
        x, base = text_x, cy + 8
        if len(shown) < len(words):
            x += put_text(canvas, "...", (x, base), scale, COLOR_SUBTLE, 1) + 12
        for i, word in enumerate(shown):
            is_last = i == len(shown) - 1
            x += put_text(canvas, word, (x, base), scale, COLOR_YELLOW_TEXT if is_last else COLOR_WHITE,
                          2 if is_last else 1)
            if not is_last:
                x += put_text(canvas, sep, (x, base), scale, COLOR_SUBTLE, 1)

    def _draw_hand_hint(self, canvas):
        """Onboarding pill shown while nothing is being signed and no hands are visible."""
        text = "Show your hands to begin"
        text_w, text_h = text_size(text, 0.65, 1)
        width = text_w + 60
        x1, y1 = UI_W // 2 - width // 2, int(UI_H * 0.034)
        pulse = 0.5 + 0.5 * np.sin((time.time() - self._t0) * 2.6)
        draw_rounded_rect(canvas, (x1, y1), (x1 + width, y1 + 40), 20, fill=COLOR_CARD_BG, fill_alpha=0.86,
                          border=COLOR_YELLOW, border_px=2, border_alpha=float(0.45 + 0.45 * pulse))
        cv2.circle(canvas, (x1 + 22, y1 + 20), 5, COLOR_YELLOW, -1, cv2.LINE_AA)
        put_text(canvas, text, (x1 + 38, y1 + 20 + text_h // 2), 0.65, COLOR_WHITE, 1)

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
        cv2.line(canvas, (bar_x, py1 + off['keys'] - 14), (bar_x + bar_w, py1 + off['keys'] - 14), (80, 80, 80), 1)
        for i, line in enumerate(L['key_lines']):
            put_text(canvas, line, (bar_x, py1 + off['keys'] + i * 17), 0.42, COLOR_MUTED, 1, FONT_SMALL)


if __name__ == "__main__":
    choice = show_main_menu(system_info={'signs': 69, 'expressions': True, 'mismatch': False})
    print(f"Menu returned: {choice}")