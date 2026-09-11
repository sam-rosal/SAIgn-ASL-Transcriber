import cv2
import mediapipe as mp
import os
import sqlite3
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GESTURE_MODEL = os.path.join(SCRIPT_DIR, 'gesture_recognizer.task')
FACE_MODEL = os.path.join(SCRIPT_DIR, 'face_landmarker.task')
DB_PATH = os.path.join(SCRIPT_DIR, 'sai_vision.db')

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
]

def load_db_configurations():
    """
    Connects to the database and populates active lookups into RAM dictionaries 
    to avoid slow, blocking SQL transactions inside the OpenCV camera loop.
    """
    gesture_cache = {}
    expression_cache = {}
    
    if not os.path.exists(DB_PATH):
        print(f"Warning: Database at {DB_PATH} not found. Running with unmapped fallbacks.")
        return gesture_cache, expression_cache

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Load dynamic gesture mappings
    cursor.execute("SELECT raw_label, display_name, min_score FROM gesture_mappings")
    for raw, display, min_score in cursor.fetchall():
        gesture_cache[raw] = {'display_name': display, 'min_score': min_score}
        
    # Load dynamic expression thresholds
    cursor.execute("SELECT blendshape_name, display_name, activation_threshold FROM expression_thresholds")
    for blendshape, display, threshold in cursor.fetchall():
        expression_cache[blendshape] = {'display_name': display, 'threshold': threshold}
        
    conn.close()
    return gesture_cache, expression_cache

def draw_manual_landmarks(image, landmarks, color=(0, 255, 0)):
    h, w, _ = image.shape
    for connection in HAND_CONNECTIONS:
        pt1 = (int(landmarks[connection[0]].x * w), int(landmarks[connection[0]].y * h))
        pt2 = (int(landmarks[connection[1]].x * w), int(landmarks[connection[1]].y * h))
        cv2.line(image, pt1, pt2, (200, 200, 200), 1)
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 3, color, -1)

def draw_ui_panel(frame, gestures, expressions):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    cv2.putText(frame, "DB-DRIVEN VISION SYSTEM | ACTIVE", (15, 32), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    y_offset = 80
    for g in gestures:
        cv2.rectangle(frame, (15, y_offset), (165, y_offset + 15), (50, 50, 50), -1)
        cv2.rectangle(frame, (15, y_offset), (15 + int(150 * g['score']), y_offset + 15), (0, 255, 0), -1)
        cv2.putText(frame, f"{g['label']}", (15, y_offset - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 45

    for name, score in expressions.items():
        cv2.rectangle(frame, (15, y_offset), (165, y_offset + 15), (50, 50, 50), -1)
        cv2.rectangle(frame, (15, y_offset), (15 + int(150 * score), y_offset + 15), (255, 150, 0), -1)
        cv2.putText(frame, f"{name.upper()}", (15, y_offset - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 45

def run_system():
    # Fetch cached lookups from database configurations
    gesture_config, expression_config = load_db_configurations()
    print(f"Loaded {len(gesture_config)} Gestures and {len(expression_config)} Expressions from DB Cache.")

    base_gesture = python.BaseOptions(model_asset_path=GESTURE_MODEL)
    gesture_rec = vision.GestureRecognizer.create_from_options(
        vision.GestureRecognizerOptions(base_options=base_gesture, num_hands=2))

    base_face = python.BaseOptions(model_asset_path=FACE_MODEL)
    face_landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(base_options=base_face, output_face_blendshapes=True))

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Infer frame
        g_result = gesture_rec.recognize(mp_image)
        f_result = face_landmarker.detect(mp_image)

        # Dynamic Hand Recognition Process
        active_gestures = []
        if g_result.gestures:
            for i, hand in enumerate(g_result.gestures):
                top = hand[0]
                raw_name = top.category_name
                
                # Check config cache in RAM
                if raw_name in gesture_config:
                    config = gesture_config[raw_name]
                    # Apply dynamic min_score threshold from db
                    if top.score >= config['min_score']:
                        active_gestures.append({
                            'label': config['display_name'], 
                            'score': top.score
                        })
                elif raw_name != "None":
                    active_gestures.append({'label': raw_name, 'score': top.score})
                
                draw_manual_landmarks(frame, g_result.hand_landmarks[i])

        # Dynamic Face Recognition Process
        active_expr = {}
        if f_result.face_blendshapes:
            shapes = {s.category_name: s.score for s in f_result.face_blendshapes[0]}
            
            # Check dynamic expression configurations from db cache
            for blend_name, config in expression_config.items():
                current_val = shapes.get(blend_name, 0.0)
                if current_val >= config['threshold']:
                    active_expr[config['display_name']] = current_val

        # Draw Interface HUD & Display
        draw_ui_panel(frame, active_gestures, active_expr)
        cv2.imshow('SAIgn Database-Driven Vision Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == 27: break # ESC Key Exit

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_system()