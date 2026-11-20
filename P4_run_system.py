import json
import os
import sqlite3
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

SEQUENCE_LENGTH = 30
RAW_CAPTURE_WINDOW = 45  # ~1.5s rolling capture buffer before velocity segmentation + resampling
VELOCITY_THRESHOLD = 0.015
MIN_ACTIVE_FRAMES = 10
EXPECTED_FEATURE_DIM = 225  # Pose (99) + Left Hand (63) + Right Hand (63)

DEBOUNCE_WINDOW_SIZE = 8
DEBOUNCE_CONSENSUS_COUNT = 5


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
        conn = sqlite3.connect(DB_PATH)
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

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT raw_label, display_name, min_score FROM gesture_mappings;")
            for raw_label, display_name, min_score in cursor.fetchall():
                if not raw_label:
                    continue
                gesture_cache[raw_label] = {
                    'display_name': display_name or raw_label.replace('_', ' ').title(),
                    'min_score': float(min_score) if min_score is not None else 0.40
                }
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("SELECT blendshape_name, display_name, activation_threshold FROM expression_thresholds;")
            for blendshape, display_name, threshold in cursor.fetchall():
                if not blendshape or threshold is None:
                    continue
                expression_thresholds[blendshape] = {
                    'display_name': display_name or blendshape,
                    'threshold': float(threshold)
                }
        except sqlite3.OperationalError:
            pass
    finally:
        conn.close()

    if not trained_labels and gesture_cache:
        trained_labels = sorted(gesture_cache.keys())

    return trained_labels, gesture_cache, expression_thresholds


def get_confidence_color(score):
    score = max(0.0, min(1.0, float(score)))
    b = int((1.0 - score) * 255)
    r = int(score * 255)
    g = 0
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
    Mirrors P4_extract_features_4.py preprocessing on the live rolling window:
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
    if current_frames == target_frames:
        return features

    if current_frames < 2:
        pad_amount = max(0, target_frames - current_frames)
        return np.pad(features, ((0, pad_amount), (0, 0)), mode='edge')[:target_frames]

    x_old = np.linspace(0, 1, num=current_frames)
    x_new = np.linspace(0, 1, num=target_frames)
    interpolator = interp1d(x_old, features, axis=0, kind='linear')
    return interpolator(x_new).astype(np.float32)


def detect_expressions(results, thresholds):
    active_emotions = []
    if not results.face_landmarks or not thresholds:
        return active_emotions

    landmarks = results.face_landmarks.landmark

    # Brow furrow / squint (check stricter threshold 0.020 before 0.025)
    if 'browLowerer' in thresholds or 'browSquint' in thresholds:
        brow_dist = abs(landmarks[70].y - landmarks[159].y)
        if 'browSquint' in thresholds and brow_dist < thresholds['browSquint']['threshold']:
            active_emotions.append(thresholds['browSquint']['display_name'])
        elif 'browLowerer' in thresholds and brow_dist < thresholds['browLowerer']['threshold']:
            active_emotions.append(thresholds['browLowerer']['display_name'])

    # Inner brow raise
    if 'browInnerUp' in thresholds:
        inner_brow = abs(landmarks[55].y - landmarks[6].y)
        if inner_brow > thresholds['browInnerUp']['threshold']:
            active_emotions.append(thresholds['browInnerUp']['display_name'])

    # Eye aperture (shared between eyeWide and eyeSquint)
    left_eye_open = abs(landmarks[159].y - landmarks[145].y)
    right_eye_open = abs(landmarks[386].y - landmarks[374].y)

    if 'eyeWideLeft' in thresholds or 'eyeWideRight' in thresholds:
        left_wide = 'eyeWideLeft' in thresholds and left_eye_open > thresholds['eyeWideLeft']['threshold']
        right_wide = 'eyeWideRight' in thresholds and right_eye_open > thresholds['eyeWideRight']['threshold']
        if left_wide or right_wide:
            display_label = thresholds.get('eyeWideLeft', thresholds.get('eyeWideRight', {})).get('display_name', 'Fearful / Startled')
            if display_label not in active_emotions:
                active_emotions.append(display_label)

    if 'eyeSquintLeft' in thresholds or 'eyeSquintRight' in thresholds:
        left_squint = 'eyeSquintLeft' in thresholds and left_eye_open < thresholds['eyeSquintLeft']['threshold']
        right_squint = 'eyeSquintRight' in thresholds and right_eye_open < thresholds['eyeSquintRight']['threshold']
        if left_squint or right_squint:
            display_label = thresholds.get('eyeSquintLeft', thresholds.get('eyeSquintRight', {})).get('display_name', 'Suspicious / Wincing')
            if display_label not in active_emotions:
                active_emotions.append(display_label)

    # Smile
    if 'mouthSmileLeft' in thresholds or 'mouthSmileRight' in thresholds:
        left_smile = abs(landmarks[61].x - landmarks[13].x)
        right_smile = abs(landmarks[291].x - landmarks[13].x)

        left_active = 'mouthSmileLeft' in thresholds and left_smile > thresholds['mouthSmileLeft']['threshold']
        right_active = 'mouthSmileRight' in thresholds and right_smile > thresholds['mouthSmileRight']['threshold']

        if left_active or right_active:
            display_label = thresholds.get('mouthSmileLeft', thresholds.get('mouthSmileRight', {})).get('display_name', 'Happy')
            if display_label not in active_emotions:
                active_emotions.append(display_label)

    # Frown
    if 'mouthFrownLeft' in thresholds or 'mouthFrownRight' in thresholds:
        left_frown = landmarks[61].y - landmarks[17].y
        right_frown = landmarks[291].y - landmarks[17].y

        left_frown_active = 'mouthFrownLeft' in thresholds and left_frown > thresholds['mouthFrownLeft']['threshold']
        right_frown_active = 'mouthFrownRight' in thresholds and right_frown > thresholds['mouthFrownRight']['threshold']

        if left_frown_active or right_frown_active:
            display_label = thresholds.get('mouthFrownLeft', thresholds.get('mouthFrownRight', {})).get('display_name', 'Sad / Concerned')
            if display_label not in active_emotions:
                active_emotions.append(display_label)

    # Jaw drop / open mouth
    if 'jawOpen' in thresholds:
        mouth_open = abs(landmarks[14].y - landmarks[13].y)
        if mouth_open > thresholds['jawOpen']['threshold']:
            active_emotions.append(thresholds['jawOpen']['display_name'])

    # Mouth pucker / horizontal narrowing
    if 'mouthPucker' in thresholds:
        mouth_width = abs(landmarks[291].x - landmarks[61].x)
        if mouth_width < thresholds['mouthPucker']['threshold']:
            active_emotions.append(thresholds['mouthPucker']['display_name'])

    # Nose sneer
    if 'noseSneerLeft' in thresholds or 'noseSneerRight' in thresholds:
        nose_dist = abs(landmarks[197].y - landmarks[0].y)

        left_sneer_active = 'noseSneerLeft' in thresholds and nose_dist < thresholds['noseSneerLeft']['threshold']
        right_sneer_active = 'noseSneerRight' in thresholds and nose_dist < thresholds['noseSneerRight']['threshold']

        if left_sneer_active or right_sneer_active:
            display_label = thresholds.get('noseSneerLeft', thresholds.get('noseSneerRight', {})).get('display_name', 'Disgust')
            if display_label not in active_emotions:
                active_emotions.append(display_label)

    # Neutral fallback
    if not active_emotions and 'faceNeutral' in thresholds:
        active_emotions.append(thresholds['faceNeutral']['display_name'])

    return active_emotions


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
        if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
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
    confidence = 0.0

    try:
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[WARN] Failed to read frame from webcam. Ending capture session.")
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                keypoints = extract_landmarks(results)
                raw_buffer.append(keypoints)

                if len(raw_buffer) >= sequence_length:
                    aligned_seq = prepare_live_sequence(raw_buffer, target_frames=sequence_length)
                    input_tensor = np.expand_dims(aligned_seq, axis=0)
                    res = model.predict(input_tensor, verbose=0)[0]
                    prediction_idx = int(np.argmax(res))
                    confidence = float(res[prediction_idx])

                    raw_label = labels[prediction_idx] if prediction_idx < len(labels) else "UNKNOWN"

                    cached_data = gesture_cache.get(raw_label, {
                        'display_name': raw_label.replace('_', ' ').title(),
                        'min_score': 0.40
                    })

                    if confidence >= cached_data['min_score']:
                        frame_candidate = cached_data['display_name']
                    else:
                        frame_candidate = "Uncertain"

                    prediction_history.append(frame_candidate)

                    vote_counts = Counter(prediction_history)
                    most_common_gesture, vote_count = vote_counts.most_common(1)[0]

                    if vote_count >= DEBOUNCE_CONSENSUS_COUNT:
                        current_gesture = most_common_gesture

                active_expressions = detect_expressions(results, expression_thresholds)
                emotion_text = f"Emotion: {', '.join(active_expressions)}" if active_expressions else "Emotion: Neutral"

                hud_color = get_confidence_color(confidence)

                cv2.rectangle(image, (10, 10), (480, 110), (30, 30, 30), -1)
                cv2.rectangle(image, (10, 10), (480, 110), hud_color, 2)

                cv2.putText(image, f"Sign: {current_gesture} ({confidence*100:.1f}%)",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, hud_color, 2, cv2.LINE_AA)
                cv2.putText(image, emotion_text,
                            (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

                cv2.imshow(window_name, image)

                key = cv2.waitKey(10) & 0xFF
                if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
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
            "Please run P4_train_model_4.py to generate the trained model."
        )

    model = load_model(MODEL_PATH)

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    seq_len = input_shape[1] or SEQUENCE_LENGTH
    feature_dim = input_shape[2]
    if feature_dim != EXPECTED_FEATURE_DIM:
        raise ValueError(
            f"Model feature dimension mismatch: '{MODEL_PATH}' expects {feature_dim} features per frame, "
            f"but Holistic extraction produces {EXPECTED_FEATURE_DIM}. Re-extract features and retrain the model."
        )

    model_num_classes = model.output_shape[-1]

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
            run_recognition_engine(model, trained_labels, gesture_cache, expression_thresholds, seq_len)


if __name__ == "__main__":
    main()