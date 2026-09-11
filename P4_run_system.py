import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# --- PROTOBUF 4/5 COMPATIBILITY PATCHES FOR MEDIAPIPE HOLISTIC (NECESSARY & IMPORTANT) ---
from google.protobuf import descriptor as _descriptor
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import message_factory as _message_factory

# 1. Patch missing .label descriptor attribute
if not hasattr(_descriptor.FieldDescriptor, 'label'):
    _descriptor.FieldDescriptor.label = property(lambda self: getattr(self, '_label', None))

# 2. Patch legacy GetPrototype method to use modern GetMessageClass for compatibility
if not hasattr(_symbol_database.SymbolDatabase, 'GetPrototype'):
    _symbol_database.SymbolDatabase.GetPrototype = lambda self, descriptor: _message_factory.GetMessageClass(descriptor)

if not hasattr(_message_factory.MessageFactory, 'GetPrototype'):
    _message_factory.MessageFactory.GetPrototype = lambda self, descriptor: _message_factory.GetMessageClass(descriptor)
# -----------------------------------------------------------------

import sqlite3
import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.python.solutions import holistic as mp_holistic
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical

# ==============================================================================
# 1. SETUP MODEL ASSETS & DIRECTORY PATHS
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Database path fallbacks
DB_PATH = os.path.join(SCRIPT_DIR, 'saign_vision.db')
if not os.path.exists(DB_PATH):
    DB_PATH = os.path.join(SCRIPT_DIR, 'saign_vision.db')

MODEL_PATH = os.path.join(SCRIPT_DIR, 'asl_model.keras')

MAX_SEQUENCE_LENGTH = 30  # Frame buffer length matching LSTM input shape
CONFIDENCE_THRESHOLD = 0.5

# --- REAL-TIME SIGN SEGMENTATION & DEBOUNCE CONFIGURATION ---
VELOCITY_THRESHOLD = 0.015   # Minimum movement delta to qualify as active signing
NEUTRAL_Y_THRESHOLD = 0.50    # Y-coordinate boundary (0 top, 1 bottom) representing resting area
DEBOUNCE_COOLDOWN = 0.5       # Minimum time gap (seconds) required before allowing repeated words
MIN_SIGN_FRAMES = 10          # Minimum collected frames required before triggering inference

# ==============================================================================
# 2. MANUAL SKELETON MAP (21 HAND LANDMARKS)
# ==============================================================================
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
]

# ==============================================================================
# 3. DATABASE CONFIGURATION & RAM CACHING
# ==============================================================================
def load_db_configurations():
    """Connects to SQLite once and caches active mappings into RAM dictionaries."""
    gesture_cache = {}
    expression_cache = {}
    
    if not os.path.exists(DB_PATH):
        print(f"Warning: Database at {DB_PATH} not found. Running with default fallbacks.")
        # Default expression fallback entries if DB missing
        expression_cache = {
            'browInnerUp': {'display_name': 'Surprised / Question', 'threshold': 0.035},
            'mouthSmileLeft': {'display_name': 'Happy', 'threshold': 0.12},
            'browLowerer': {'display_name': 'Angry', 'threshold': 0.025},
            'browSquint': {'display_name': 'Confused', 'threshold': 0.020}
        }
        return gesture_cache, expression_cache

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT raw_label, display_name, min_score FROM gesture_mappings")
    for raw, display, min_score in cursor.fetchall():
        gesture_cache[raw] = {'display_name': display, 'min_score': min_score}
        
    cursor.execute("SELECT blendshape_name, display_name, activation_threshold FROM expression_thresholds")
    for blendshape, display, threshold in cursor.fetchall():
        expression_cache[blendshape] = {'display_name': display, 'threshold': threshold}
        
    conn.close()
    
    # Ensure fallbacks exist for new emotion blendshapes if DB lacks them
    if 'browLowerer' not in expression_cache:
        expression_cache['browLowerer'] = {'display_name': 'Angry', 'threshold': 0.025}
    if 'browSquint' not in expression_cache:
        expression_cache['browSquint'] = {'display_name': 'Confused', 'threshold': 0.020}

    return gesture_cache, expression_cache

# ==============================================================================
# 4. FEATURE EXTRACTION, GEOMETRY & MOTION HELPERS
# ==============================================================================
def extract_frame_landmarks(results):
    """Flattens pose and dual hand landmarks into a unified 225-element vector."""
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])

def extract_hand_coordinates_only(results):
    """Extracts raw 3D hand landmarks for velocity and neutral space detection."""
    hands_combined = []
    if results.left_hand_landmarks:
        hands_combined.extend([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
    if results.right_hand_landmarks:
        hands_combined.extend([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
    
    return np.array(hands_combined) if len(hands_combined) > 0 else None

def calculate_hand_velocity(prev_coords, curr_coords):
    """Computes frame-to-frame mean landmark movement displacement (Velocity Thresholding)."""
    if prev_coords is None or curr_coords is None or prev_coords.shape != curr_coords.shape:
        return 0.0
    return float(np.mean(np.linalg.norm(curr_coords - prev_coords, axis=1)))

def is_hand_in_neutral_space(results):
    """Checks if hands drop into lower torso/resting space (Neutral Space Detection)."""
    if results.left_hand_landmarks and results.left_hand_landmarks.landmark[0].y > NEUTRAL_Y_THRESHOLD:
        return True
    if results.right_hand_landmarks and results.right_hand_landmarks.landmark[0].y > NEUTRAL_Y_THRESHOLD:
        return True
    return False

def detect_expressions(results, expression_cache):
    """Computes facial geometry thresholds for non-manual markers (Happy, Surprised, Angry, Confused)."""
    active_expressions = {}
    if not results.face_landmarks:
        return active_expressions

    face = results.face_landmarks.landmark

    left_eyebrow = face[70].y
    right_eyebrow = face[300].y
    left_eye = face[159].y
    right_eye = face[386].y
    left_eye_bottom = face[145].y
    
    forehead_height = abs(left_eye - left_eyebrow)
    eye_aperture = abs(left_eye_bottom - left_eye)

    # 1. Eyebrow raise detection (Surprised / Question)
    if 'browInnerUp' in expression_cache:
        rule = expression_cache['browInnerUp']
        if forehead_height > rule['threshold']:
            strength = min(1.0, (forehead_height - rule['threshold']) / 0.05 + 0.5)
            active_expressions[rule['display_name']] = strength

    # 2. Smile detection (Happy)
    mouth_left = face[61].x
    mouth_right = face[291].x
    mouth_top = face[13].y
    mouth_bottom = face[14].y
    mouth_width = abs(mouth_right - mouth_left)
    mouth_height = abs(mouth_bottom - mouth_top)

    if 'mouthSmileLeft' in expression_cache:
        rule = expression_cache['mouthSmileLeft']
        if mouth_width > rule['threshold']:
            strength = min(1.0, (mouth_width - rule['threshold']) / 0.1 + 0.5)
            active_expressions[rule['display_name']] = strength

    # 3. Brow Lowerer / Furrow detection (Angry)
    eyebrow_drop = (left_eye - left_eyebrow) + (right_eye - right_eyebrow)
    if 'browLowerer' in expression_cache:
        rule = expression_cache['browLowerer']
        if forehead_height < rule['threshold'] and mouth_height < 0.03:
            strength = min(1.0, (rule['threshold'] - forehead_height) / 0.02 + 0.5)
            active_expressions[rule['display_name']] = strength

    # 4. Asymmetric Brow / Squint detection (Confused)
    brow_asymmetry = abs(left_eyebrow - right_eyebrow)
    if 'browSquint' in expression_cache:
        rule = expression_cache['browSquint']
        if (brow_asymmetry > rule['threshold'] or eye_aperture < 0.012) and 'Angry' not in active_expressions:
            strength = min(1.0, (brow_asymmetry / 0.03) + 0.4)
            active_expressions[rule['display_name']] = strength

    return active_expressions

# ==============================================================================
# 5. UI, COLOR GRADIENT & SKELETON RENDERING FUNCTIONS
# ==============================================================================
def get_confidence_color(score):
    """Maps confidence score (0.0 - 1.0) to a BGR gradient from Light Blue to Red."""
    factor = max(0.0, min(1.0, float(score)))
    # Light Blue (BGR): (255, 200, 100) -> Pure Red (BGR): (0, 0, 255)
    b = int(255 * (1.0 - factor) + 0 * factor)
    g = int(200 * (1.0 - factor) + 0 * factor)
    r = int(100 * (1.0 - factor) + 255 * factor)
    return (b, g, r)

def draw_start_menu(frame):
    """Renders dark background overlay start menu before starting the execution thread."""
    overlay = frame.copy()
    h, w, _ = frame.shape
    
    cv2.rectangle(overlay, (w//8, h//6), (7*w//8, 5*h//6), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    cv2.rectangle(frame, (w//8, h//6), (7*w//8, h//6 + 10), (0, 255, 0), -1)

    cv2.putText(frame, "SAIgn | AI ASL TRANSCRIBER", (w//2 - 190, h//3 - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.putText(frame, "Real-Time Sequence Recognition Engine", (w//2 - 165, h//3 + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, "CONTROLS:", (w//8 + 40, h//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "• PRESS 'SPACEBAR' TO START LIVE TRANSLATION", (w//8 + 50, h//2 + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "• SENTENCE AUTO-CLEARS WHEN DISPLAY IS FULL", (w//8 + 50, h//2 + 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "• PRESS 'Q' OR 'ESC' TO TERMINATE", (w//8 + 50, h//2 + 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.putText(frame, "READY - PRESS SPACE TO BEGIN", (w//2 - 145, 5*h//6 - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

def draw_manual_landmarks(image, hand_landmarks, color=(0, 255, 0)):
    """Renders custom 21-point hand topology connections directly on the frame."""
    if not hand_landmarks:
        return
    
    h, w, _ = image.shape
    landmarks = hand_landmarks.landmark

    for connection in HAND_CONNECTIONS:
        pt1 = (int(landmarks[connection[0]].x * w), int(landmarks[connection[0]].y * h))
        pt2 = (int(landmarks[connection[1]].x * w), int(landmarks[connection[1]].y * h))
        cv2.line(image, pt1, pt2, (200, 200, 200), 1)

    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 3, color, -1)

def draw_ui_panel(frame, expressions, sentence=[]):
    """Renders dark banner HUD overlay with colored sentence text and current emotion line (No progress bars)."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 75), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Title header
    cv2.putText(frame, "SAIgn | CUSTOM LSTM VISION ENGINE", (15, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # 1. Output dynamic colored sentence (Gradient Light Blue -> Red based on score)
    x_curr = 15
    y_sentence = 40
    prefix = "Sentence: "
    cv2.putText(frame, prefix, (x_curr, y_sentence), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    x_curr += cv2.getTextSize(prefix, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]

    if not sentence:
        cv2.putText(frame, "Waiting for gestures...", (x_curr, y_sentence), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    else:
        for word, score in sentence:
            color = get_confidence_color(score)
            word_str = f"{word} "
            cv2.putText(frame, word_str, (x_curr, y_sentence), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            x_curr += cv2.getTextSize(word_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]

    # 2. Output current emotion line in brackets with confidence gradient color
    y_emotion = 60
    emo_prefix = "Emotion: "
    cv2.putText(frame, emo_prefix, (15, y_emotion), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    x_emo = 15 + cv2.getTextSize(emo_prefix, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]

    if expressions:
        top_emo = max(expressions.items(), key=lambda item: item[1])
        emo_name, emo_score = top_emo[0], top_emo[1]
        emo_text = f"({emo_name})"
        emo_color = get_confidence_color(emo_score)
    else:
        emo_text = "(Neutral)"
        emo_color = get_confidence_color(0.0)

    cv2.putText(frame, emo_text, (x_emo, y_emotion), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, emo_color, 1, cv2.LINE_AA)

# ==============================================================================
# 6. MAIN PIPELINE EXECUTION THREAD
# ==============================================================================
def run_system():
    # Load database rule cache into RAM
    gesture_cache, expression_cache = load_db_configurations()
    print(f"Loaded {len(gesture_cache)} Gestures and {len(expression_cache)} Expressions from DB Cache.")

    # Load custom Keras LSTM model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model file at {MODEL_PATH}. Please run P4_train_model.py first.")

    print(f"Loading custom Keras model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    classes = sorted(list(gesture_cache.keys())) if gesture_cache else []

    sequence = []
    sentence = []  # Stores tuples of (predicted_label, confidence_score)
    prev_hand_coords = None
    last_prediction = None
    last_pred_time = 0

    cap = cv2.VideoCapture(0)

    in_start_menu = True
    print("\nStarting camera feed. Displaying Start Menu overlay.")

    while in_start_menu and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        draw_start_menu(frame)
        cv2.imshow('SAIgn Database-Driven Vision Frame', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            in_start_menu = False
        elif key == 27 or key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    print("Exited Start Menu. Live ASL Translation active. Press 'q' or 'ESC' to terminate.")

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)

            landmarks = extract_frame_landmarks(results)
            curr_hand_coords = extract_hand_coordinates_only(results)

            velocity = calculate_hand_velocity(prev_hand_coords, curr_hand_coords)
            is_moving = velocity > VELOCITY_THRESHOLD
            in_neutral = is_hand_in_neutral_space(results)
            prev_hand_coords = curr_hand_coords

            if is_moving and not in_neutral:
                sequence.append(landmarks)
            else:
                if len(sequence) >= MIN_SIGN_FRAMES and classes:
                    padded_sequence = sequence[-MAX_SEQUENCE_LENGTH:]
                    if len(padded_sequence) < MAX_SEQUENCE_LENGTH:
                        padding = [np.zeros_like(landmarks)] * (MAX_SEQUENCE_LENGTH - len(padded_sequence))
                        padded_sequence = padding + padded_sequence

                    input_data = np.expand_dims(padded_sequence, axis=0)
                    predictions = model.predict(input_data, verbose=0)[0]
                    best_idx = np.argmax(predictions)
                    raw_name = classes[best_idx]
                    score = float(predictions[best_idx])

                    config = gesture_cache.get(raw_name, {'display_name': raw_name, 'min_score': CONFIDENCE_THRESHOLD})
                    if score >= config['min_score']:
                        predicted_label = config['display_name']

                        current_time = time.time()
                        is_duplicate = (predicted_label == last_prediction)
                        time_elapsed = current_time - last_pred_time

                        if not is_duplicate or time_elapsed > DEBOUNCE_COOLDOWN:
                            test_sentence = sentence + [(predicted_label, score)]
                            test_str = "Sentence: " + " ".join([item[0] for item in test_sentence])
                            text_width, _ = cv2.getTextSize(test_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

                            max_display_width = frame.shape[1] - 30
                            if text_width > max_display_width:
                                sentence.clear()

                            sentence.append((predicted_label, score))
                            last_prediction = predicted_label
                            last_pred_time = current_time

                    sequence.clear()

            active_expressions = detect_expressions(results, expression_cache)

            if results.left_hand_landmarks:
                draw_manual_landmarks(frame, results.left_hand_landmarks, color=(0, 255, 0))
            if results.right_hand_landmarks:
                draw_manual_landmarks(frame, results.right_hand_landmarks, color=(0, 255, 255))

            draw_ui_panel(frame, active_expressions, sentence)

            cv2.imshow('SAIgn Database-Driven Vision Frame', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_system()