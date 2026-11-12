import os
import sqlite3
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import mediapipe as mp
from collections import deque, Counter

# ==============================================================================
# PATHS & SYSTEM CONFIGURATION
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, 'saign_vision.db')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'asl_model.keras')

SEQUENCE_LENGTH = 30 
DEBOUNCE_WINDOW_SIZE = 8
DEBOUNCE_CONSENSUS_COUNT = 5


# ==============================================================================
# HELPER FUNCTIONS: DATABASE & DATA PROCESSING
# ==============================================================================
def load_db_cache():
    gesture_cache = {}
    expression_thresholds = {}

    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT raw_label, display_name, min_score FROM gesture_mappings;")
        for raw_label, display_name, min_score in cursor.fetchall():
            gesture_cache[raw_label] = {'display_name': display_name, 'min_score': min_score}

        cursor.execute("SELECT blendshape_name, display_name, activation_threshold FROM expression_thresholds;")
        for blendshape, display_name, threshold in cursor.fetchall():
            expression_thresholds[blendshape] = {'display_name': display_name, 'threshold': threshold}
        conn.close()

    return gesture_cache, expression_thresholds


def get_confidence_color(score):
    score = max(0.0, min(1.0, float(score)))
    b = int((1.0 - score) * 255)
    r = int(score * 255)
    g = 0
    return (b, g, r)


def extract_landmarks(results):
    """
    Flattens spatial (X, Y, Z) coordinates from MediaPipe Holistic output 
    into a single 1D feature vector of exactly 225 values.
    """
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 3)
    
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)
    
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)
    
    # 99 + 63 + 63 = 225
    return np.concatenate([pose, lh, rh])


def detect_expressions(results, thresholds):
    active_emotions = []
    if not results.face_landmarks or not thresholds:
        return active_emotions

    landmarks = results.face_landmarks.landmark

    if 'browLowerer' in thresholds or 'browSquint' in thresholds:
        brow_dist = abs(landmarks[70].y - landmarks[159].y)
        if 'browLowerer' in thresholds and brow_dist < thresholds['browLowerer']['threshold']:
            active_emotions.append(thresholds['browLowerer']['display_name'])
        elif 'browSquint' in thresholds and brow_dist < thresholds['browSquint']['threshold']:
            active_emotions.append(thresholds['browSquint']['display_name'])

    if 'browInnerUp' in thresholds:
        inner_brow = abs(landmarks[55].y - landmarks[6].y)
        if inner_brow > thresholds['browInnerUp']['threshold']:
            active_emotions.append(thresholds['browInnerUp']['display_name'])

    if 'mouthSmileLeft' in thresholds or 'mouthSmileRight' in thresholds:
        left_smile = abs(landmarks[61].x - landmarks[13].x)
        right_smile = abs(landmarks[291].x - landmarks[13].x)
        
        left_active = 'mouthSmileLeft' in thresholds and left_smile > thresholds['mouthSmileLeft']['threshold']
        right_active = 'mouthSmileRight' in thresholds and right_smile > thresholds['mouthSmileRight']['threshold']
        
        if left_active or right_active:
            display_label = thresholds.get('mouthSmileLeft', thresholds.get('mouthSmileRight', {})).get('display_name', 'Happy')
            if display_label not in active_emotions:
                active_emotions.append(display_label)

    if 'mouthFrownLeft' in thresholds or 'mouthFrownRight' in thresholds:
        left_frown = landmarks[61].y - landmarks[17].y
        right_frown = landmarks[291].y - landmarks[17].y
        
        left_frown_active = 'mouthFrownLeft' in thresholds and left_frown > thresholds['mouthFrownLeft']['threshold']
        right_frown_active = 'mouthFrownRight' in thresholds and right_frown > thresholds['mouthFrownRight']['threshold']
        
        if left_frown_active or right_frown_active:
            display_label = thresholds.get('mouthFrownLeft', thresholds.get('mouthFrownRight', {})).get('display_name', 'Sad / Concerned')
            if display_label not in active_emotions:
                active_emotions.append(display_label)

    if 'noseSneerLeft' in thresholds or 'noseSneerRight' in thresholds:
        nose_dist = abs(landmarks[197].y - landmarks[0].y)
        
        left_sneer_active = 'noseSneerLeft' in thresholds and nose_dist < thresholds['noseSneerLeft']['threshold']
        right_sneer_active = 'noseSneerRight' in thresholds and nose_dist < thresholds['noseSneerRight']['threshold']
        
        if left_sneer_active or right_sneer_active:
            display_label = thresholds.get('noseSneerLeft', thresholds.get('noseSneerRight', {})).get('display_name', 'Disgust')
            if display_label not in active_emotions:
                active_emotions.append(display_label)

    if not active_emotions and 'faceNeutral' in thresholds:
        active_emotions.append(thresholds['faceNeutral']['display_name'])

    return active_emotions


# ==============================================================================
# GUI SCREENS & STATE MANAGERS
# ==============================================================================
def show_main_menu():
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

    cv2.imshow('SAIgn - Main Menu', menu_bg)

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  
            cv2.destroyAllWindows()
            return 'QUIT'
        elif key in (13, 32):  
            cv2.destroyWindow('SAIgn - Main Menu')
            return 'START'


def run_recognition_engine(model, gesture_cache, expression_thresholds):
    labels = list(gesture_cache.keys()) if gesture_cache else ['UNKNOWN']
    
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    sequence_buffer = []  
    prediction_history = deque(maxlen=DEBOUNCE_WINDOW_SIZE)
    
    current_gesture = "Waiting..."
    confidence = 0.0

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
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
            sequence_buffer.append(keypoints)
            sequence_buffer = sequence_buffer[-SEQUENCE_LENGTH:]  

            if len(sequence_buffer) == SEQUENCE_LENGTH:
                res = model.predict(np.expand_dims(sequence_buffer, axis=0), verbose=0)[0]
                prediction_idx = np.argmax(res)
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

            cv2.imshow('SAIgn - Live Recognition Engine', image)

            if cv2.waitKey(10) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyWindow('SAIgn - Live Recognition Engine')


# ==============================================================================
# MAIN PROGRAM EXECUTION ENTRY POINT
# ==============================================================================
def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Please run P4_train_model.py to generate the trained model."
        )

    model = load_model(MODEL_PATH)

    while True:
        gesture_cache, expression_thresholds = load_db_cache()
        action = show_main_menu()

        if action == 'QUIT':
            print("Exiting SAIgn Engine cleanly. Goodbye!")
            break
        elif action == 'START':
            run_recognition_engine(model, gesture_cache, expression_thresholds)


if __name__ == "__main__":
    main()