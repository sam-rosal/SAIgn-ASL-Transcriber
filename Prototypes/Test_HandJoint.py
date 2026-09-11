import cv2
import mediapipe as mp
import time
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Path Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'gesture_recognizer.task')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing 'gesture_recognizer.task' in: {SCRIPT_DIR}")

# Manual connection mapping
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17) # Pinky & Palm
]

# Global result storage
latest_result = None

def result_callback(result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# 2. Configuration
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=result_callback
)

def draw_confidence_meter(image, gesture_name, score):
    """Draws a visual bar and text for gesture confidence."""
    bar_x, bar_y = 50, 80
    bar_width = 200
    bar_height = 20
    
    # Calculate fill width based on score (0.0 to 1.0)
    fill_width = int(bar_width * score)
    
    # Colors (BGR)
    color = (0, 255, 0) if score > 0.8 else (0, 255, 255) # Green if high, Yellow if mid
    
    # Draw Background (Gray)
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
    # Draw Fill
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
    # Draw Text
    cv2.putText(image, f"{gesture_name}: {int(score * 100)}%", (bar_x, bar_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def draw_landmarks_manually(image, landmarks):
    h, w, _ = image.shape
    for connection in HAND_CONNECTIONS:
        pt1 = (int(landmarks[connection[0]].x * w), int(landmarks[connection[0]].y * h))
        pt2 = (int(landmarks[connection[1]].x * w), int(landmarks[connection[1]].y * h))
        cv2.line(image, pt1, pt2, (255, 255, 255), 2)
    for landmark in landmarks:
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

# 3. Main Loop
with vision.GestureRecognizer.create_from_options(options) as recognizer:
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        timestamp = int(time.time() * 1000)
        recognizer.recognize_async(mp_image, timestamp)

        if latest_result is not None:
            # Draw Landmarks
            if latest_result.hand_landmarks:
                for hand_landmarks in latest_result.hand_landmarks:
                    draw_landmarks_manually(frame, hand_landmarks)
            
            # Draw Gesture Info and Meter
            if latest_result.gestures:
                for i, gesture in enumerate(latest_result.gestures):
                    # The model identifies 'Victory' as the peace sign
                    top_gesture = gesture[0] 
                    if top_gesture.category_name != "None":
                        draw_confidence_meter(frame, top_gesture.category_name, top_gesture.score)

        cv2.imshow('Gesture Confidence Meter', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()