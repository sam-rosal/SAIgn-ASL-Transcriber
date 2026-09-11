import cv2
import mediapipe as mp
import time
import os
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- MANUAL CONFIGURATION ---
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),    # Index
    (9, 10), (10, 11), (11, 12),       # Middle
    (13, 14), (14, 15), (15, 16),      # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)          # Palm
]

latest_result = None

def result_callback(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

def get_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_gesture(landmarks, hand_label):
    # SWAP LOGIC: MediaPipe Left = Mirrored Right
    actual_side = "Right" if hand_label == "Left" else "Left"
    
    # PALS VS BACK (Corrected for mirrored view)
    is_palm = False
    if actual_side == "Left":
        is_palm = landmarks[2].x > landmarks[17].x
    else:
        is_palm = landmarks[2].x < landmarks[17].x

    # Finger states (1=open, 0=closed)
    # Index (8), Middle (12), Ring (16), Pinky (20)
    f = []
    for tip_id in [8, 12, 16, 20]:
        f.append(1 if landmarks[tip_id].y < landmarks[tip_id - 2].y else 0)

    # Thumb extension logic
    if actual_side == "Left":
        thumb_extended = landmarks[4].x > landmarks[3].x
    else:
        thumb_extended = landmarks[4].x < landmarks[3].x

    # --- GESTURE LOGIC (Thumbs Up/Down Removed) ---

    # 1. Hello: 5 fingers up + Palm Facing + Spread
    thumb_pinky_spread = get_distance(landmarks[4], landmarks[20])
    if sum(f) == 4 and thumb_extended and is_palm and thumb_pinky_spread > 0.12:
        return f"{actual_side}: Hello"

    # 2. Peace Sign: Index and Middle up, others closed
    if f[0] == 1 and f[1] == 1 and f[2] == 0 and f[3] == 0:
        return f"{actual_side}: Peace"

    # 3. OK Sign: Thumb near Index, other fingers up
    if is_palm and get_distance(landmarks[4], landmarks[8]) < 0.05 and f[1] == 1:
        return f"{actual_side}: OK"

    # 4. Fist: All fingers down + Thumb tucked
    if sum(f) == 0 and not thumb_extended:
        return f"{actual_side}: Fist"

    return f"{actual_side} ({'Palm' if is_palm else 'Back'})"

# --- MAIN LOOP SETUP ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'hand_landmarker.task')

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    result_callback=result_callback
)

cap = cv2.VideoCapture(0)

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        # Mirroring the frame
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Trigger detection
        landmarker.detect_async(mp_image, int(time.time() * 1000))

        if latest_result and latest_result.hand_landmarks:
            for idx, hand_landmarks in enumerate(latest_result.hand_landmarks):
                # Retrieve classification and run gesture logic
                raw_label = latest_result.handedness[idx][0].category_name
                gesture_text = get_gesture(hand_landmarks, raw_label)
                
                # Draw text near the wrist (landmark 0)
                wrist = hand_landmarks[0]
                cv2.putText(frame, gesture_text, (int(wrist.x * w), int(wrist.y * h) + 50),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)

                # Draw skeleton
                for connection in HAND_CONNECTIONS:
                    p1, p2 = hand_landmarks[connection[0]], hand_landmarks[connection[1]]
                    cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (255, 0, 0), 2)

        cv2.imshow('Gesture Recognition - Cleaned', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()