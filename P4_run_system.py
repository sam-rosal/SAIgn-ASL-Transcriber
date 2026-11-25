import json
import operator
import os
import sqlite3
from collections import deque, Counter
import cv2
import numpy as np
try:
    from keras.models import load_model
except ImportError:  # TensorFlow-bundled Keras fallback
    from tensorflow.keras.models import load_model
import mediapipe as mp
from scipy.interpolate import interp1d

# ==============================================================================
# PATHS & SYSTEM CONFIGURATION
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, 'saign_vision.db')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'asl_model.keras')
CLASSES_PATH = os.path.join(SCRIPT_DIR, 'classes.json')

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


def get_confidence_color(score):
    score = max(0.0, min(1.0, float(score)))
    # BGR: low confidence = red, high confidence = green
    b = 0
    g = int(score * 255)
    r = int((1.0 - score) * 255)
    return (b, g, r)


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
# GUI SCREENS & STATE MANAGERS
# ==============================================================================
def show_main_menu():
    window_name = 'SAIgn - Main Menu'
    menu_bg = np.zeros((600, 800, 3), dtype=np.uint8)

    cv2.rectangle(menu_bg, (50, 50), (750, 550), (35, 35, 35), -1)
    cv2.rectangle(menu_bg, (50, 50), (750, 550), (0, 215, 255), 2)

    cv2.putText(menu_bg, "SAIgn Vision Engine", (160, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(menu_bg, "Automated ASL & Expression HUD", (230, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 1, cv2.LINE_AA)

    cv2.rectangle(menu_bg, (150, 280), (650, 350), (50, 50, 50), -1)
    cv2.putText(menu_bg, "[ SPACE / ENTER ] Start Recognition", (170, 325),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 127), 2, cv2.LINE_AA)

    cv2.rectangle(menu_bg, (150, 380), (650, 450), (50, 50, 50), -1)
    cv2.putText(menu_bg, "[ ESC ] Shut Down Engine", (240, 425),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 127, 255), 2, cv2.LINE_AA)

    cv2.imshow(window_name, menu_bg)

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == 27 or is_window_closed(window_name):
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            return 'QUIT'
        elif key in (13, 32):
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            return 'START'


def run_recognition_engine(model, trained_labels, gesture_cache, expression_thresholds, sequence_length):
    window_name = 'SAIgn - Live Recognition Engine'
    labels = trained_labels if trained_labels else ['UNKNOWN']

    if not hasattr(mp, 'solutions'):
        raise ImportError(
            "This MediaPipe build does not expose the legacy 'mp.solutions' API. "
            "Install a version that does, e.g.: pip install mediapipe==0.10.14"
        )
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam (camera index 0). "
            "Verify that no other app is using the camera and macOS Camera permissions are granted."
        )

    raw_buffer = deque(maxlen=max(RAW_CAPTURE_WINDOW, sequence_length))
    prediction_history = deque(maxlen=DEBOUNCE_WINDOW_SIZE)

    current_gesture = "Waiting..."
    display_confidence = 0.0
    no_hand_frames = 0
    consecutive_errors = 0

    try:
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

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

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
                    current_gesture = "Waiting..."
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

                active_expressions = detect_expressions(
                    results, expression_thresholds,
                    image_size=(image.shape[1], image.shape[0])
                )
                emotion_text = f"Emotion: {', '.join(active_expressions)}" if active_expressions else "Emotion: Neutral"

                hud_color = get_confidence_color(display_confidence)

                cv2.rectangle(image, (10, 10), (480, 110), (30, 30, 30), -1)
                cv2.rectangle(image, (10, 10), (480, 110), hud_color, 2)

                cv2.putText(image, f"Sign: {current_gesture} ({display_confidence*100:.1f}%)",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, hud_color, 2, cv2.LINE_AA)
                cv2.putText(image, emotion_text,
                            (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

                cv2.imshow(window_name, image)

                key = cv2.waitKey(10) & 0xFF
                if key == 27 or is_window_closed(window_name):
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

    while True:
        trained_labels, gesture_cache, expression_thresholds = load_db_cache()
        if trained_labels and len(trained_labels) != model_num_classes:
            print(
                f"[WARN] Loaded {len(trained_labels)} class label(s), "
                f"but model outputs {model_num_classes} class(es). Retrain the model if labels changed."
            )

        action = show_main_menu()

        if action == 'QUIT':
            print("Exiting SAIgn Engine cleanly. Goodbye!")
            break
        elif action == 'START':
            try:
                run_recognition_engine(model, trained_labels, gesture_cache, expression_thresholds, seq_len)
            except RuntimeError as e:
                # e.g. webcam unavailable: report and return to the menu instead of crashing
                print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()