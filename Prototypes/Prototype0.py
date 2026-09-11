import cv2
import mediapipe as mp
import time
import os
import math
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class ASLRecognizer:
    def __init__(self, model_path):
        self.latest_result = None
        self.current_sentence = ""
        self.last_letter = ""
        self.frame_counter = 0
        self.CONFIDENCE_THRESHOLD = 15  # Lowered slightly for faster typing
        self.flash_timer = 0           # For visual feedback when a letter is caught
        
        # Initialize MediaPipe
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            result_callback=self.result_callback
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def result_callback(self, result, output_image, timestamp_ms):
        self.latest_result = result

    def get_dist(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def recognize(self, lm, handedness):
        # 1. Finger States (Tip vs PIP joint)
        f = []
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            f.append(1 if lm[tip].y < lm[pip].y else 0)

        thumb_tip, index_tip = lm[4], lm[8]
        middle_tip, ring_tip, pinky_tip = lm[12], lm[16], lm[20]
        
        is_right = handedness == "Right"
        thumb_out = thumb_tip.x > lm[2].x if is_right else thumb_tip.x < lm[2].x
        
        # --- GESTURE LOGIC ---
        res = ""

        # HELLO: Flat hand, all fingers up and touching
        if f == [1, 1, 1, 1] and self.get_dist(index_tip, pinky_tip) < 0.15:
            res = "HELLO"
        
        # SPACE: All fingers up and SPREAD wide
        elif f == [1, 1, 1, 1] and self.get_dist(index_tip, pinky_tip) > 0.2:
            res = " "

        # B: 4 fingers up, thumb tucked
        elif f == [1, 1, 1, 1] and not thumb_out:
            res = "B"

        # A: Fist with thumb on side
        elif sum(f) == 0 and thumb_out:
            res = "A"

        # L: Index up, thumb out
        elif f == [1, 0, 0, 0] and thumb_out:
            res = "L"

        # --- SENTENCE ENGINE ---
        if res != "" and res != "...":
            if res == self.last_letter:
                self.frame_counter += 1
            else:
                self.last_letter = res
                self.frame_counter = 0
            
            if self.frame_counter == self.CONFIDENCE_THRESHOLD:
                # Add word/letter to sentence
                if res == "HELLO":
                    self.current_sentence += " HELLO "
                else:
                    self.current_sentence += res
                self.flash_timer = 10 # Trigger visual flash
                self.frame_counter = 0 # Reset to prevent double typing
        else:
            self.frame_counter = 0
            
        return res if res != "" else "..."

# --- UI CONSTANTS ---
THEME_BLUE = (120, 60, 0)      # BGR Blue
DARK_NAVY = (40, 20, 10)
ACCENT_GOLD = (0, 215, 255)
BAR_H = 150

# --- MAIN SETUP ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(SCRIPT_DIR, 'hand_landmarker.task')

recognizer = ASLRecognizer(MODEL_FILE)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Create the modern UI Canvas
    canvas = np.zeros((h + BAR_H, w, 3), dtype=np.uint8)
    canvas[0:h, 0:w] = frame 
    cv2.rectangle(canvas, (0, h), (w, h + BAR_H), DARK_NAVY, -1)
    cv2.line(canvas, (0, h), (w, h), ACCENT_GOLD, 2)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    recognizer.landmarker.detect_async(mp_image, int(time.time() * 1000))

    detected = "..."
    if recognizer.latest_result and recognizer.latest_result.hand_landmarks:
        for idx, landmarks in enumerate(recognizer.latest_result.hand_landmarks):
            side = "Left" if recognizer.latest_result.handedness[idx][0].category_name == "Right" else "Right"
            detected = recognizer.recognize(landmarks, side)
            
            # Draw current sign above hand
            wrist = landmarks[0]
            color = (0, 255, 0) if recognizer.flash_timer > 0 else ACCENT_GOLD
            cv2.putText(canvas, detected, (int(wrist.x * w), int(wrist.y * h) - 40),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 3)

    # --- UI ELEMENTS ---
    # Letter Progress Bar
    if detected != "...":
        bar_width = int((recognizer.frame_counter / recognizer.CONFIDENCE_THRESHOLD) * 200)
        cv2.rectangle(canvas, (20, h + 20), (220, h + 35), (50, 50, 50), -1)
        cv2.rectangle(canvas, (20, h + 20), (20 + bar_width, h + 35), ACCENT_GOLD, -1)

    # Sentence Display
    display_text = recognizer.current_sentence[-25:] # Only show last 25 chars
    cv2.putText(canvas, "SENTENCE LOG:", (20, h + 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    
    # Visual Flash when a letter is added
    text_color = (255, 255, 255)
    if recognizer.flash_timer > 0:
        text_color = (0, 255, 0) # Flash green
        recognizer.flash_timer -= 1

    cv2.putText(canvas, f"> {display_text}", (20, h + 120), 
                cv2.FONT_HERSHEY_DUPLEX, 1.2, text_color, 2)

    cv2.imshow('ASL Blue Interface', canvas)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('c'): recognizer.current_sentence = ""

cap.release()
cv2.destroyAllWindows()