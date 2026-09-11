import cv2
import mediapipe as mp
import os
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==============================================================================
# 1. SETUP MODEL ASSETS & DIRECTORY PATHS
# ==============================================================================
# Dynamically locate the script's directory using absolute paths. This guarantees 
# that the program can find the model asset files (.task files) regardless of 
# how or where the script is executed.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GESTURE_MODEL = os.path.join(SCRIPT_DIR, 'gesture_recognizer.task')
FACE_MODEL = os.path.join(SCRIPT_DIR, 'face_landmarker.task')

# ==============================================================================
# 2. MANUAL SKELETON MAP (21 HAND LANDMARKS)
# ==============================================================================
# Because the legacy "mp.solutions.drawing_utils" is bypassed, we map out the 
# structural connections of the hand ourselves. Each tuple holds a start-and-end 
# landmark index pointing to MediaPipe's standard 21-point hand topology.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),                 # Thumb joints (base to tip)
    (0, 5), (5, 6), (6, 7), (7, 8),                 # Index finger joints
    (5, 9), (9, 10), (10, 11), (11, 12),            # Middle finger joints
    (9, 13), (13, 14), (14, 15), (15, 16),          # Ring finger joints
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17) # Pinky joints and baseline palm enclosure
]

# ==============================================================================
# FUNCTION: draw_manual_landmarks
# ==============================================================================
def draw_manual_landmarks(image, landmarks, color=(0, 255, 0)):
    """
    Translates MediaPipe's normalized coordinates (range 0.0 to 1.0) into actual 
    pixel coordinates based on the current window size, then draws the skeletal lines 
    and joint points directly onto the video frame using OpenCV.
    """
    h, w, _ = image.shape
    
    # Step A: Draw structural connections (bones)
    for connection in HAND_CONNECTIONS:
        # Convert raw coordinate scales (0.0 to 1.0) to actual pixel dimensions (integers)
        pt1 = (int(landmarks[connection[0]].x * w), int(landmarks[connection[0]].y * h))
        pt2 = (int(landmarks[connection[1]].x * w), int(landmarks[connection[1]].y * h))
        
        # Draw a thin, lightweight gray line between the joint coordinates
        cv2.line(image, pt1, pt2, (200, 200, 200), 1)
    
    # Step B: Draw individual joint coordinates (knuckles and tips)
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        
        # Draw a bright green solid circle at every calculated coordinate point
        cv2.circle(image, (cx, cy), 3, color, -1)

# ==============================================================================
# FUNCTION: draw_ui_panel
# ==============================================================================
def draw_ui_panel(frame, gestures, expressions):
    """
    Renders an on-screen HUD (Heads-Up Display) overlay featuring live system status, 
    active hand gesture classifications, facial expression scores, and confidence bars.
    """
    # Step A: Draw a semi-transparent dark banner at the top of the window
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), (20, 20, 20), -1) # Dark fill
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame) # Mix overlay with the live stream
    
    # Renders the system engine title text
    cv2.putText(frame, "MANUAL VISION ENGINE | RUNNING", (15, 32), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Step B: Render the active hand gestures (Labels and Confidence metrics)
    y_offset = 80 # Initial starting vertical height for the sidebar HUD elements
    for g in gestures:
        # Map generic recognizer classifications into highly recognizable labels for the UI
        label = g['label'].replace('Victory', 'PEACE').replace('Open_Palm', 'HELLO')
        
        # Draw the confidence tracking progress slot (Gray baseline container)
        cv2.rectangle(frame, (15, y_offset), (165, y_offset + 15), (50, 50, 50), -1)
        # Draw the fill bar indicating predictive confidence (Solid Green)
        cv2.rectangle(frame, (15, y_offset), (15 + int(150 * g['score']), y_offset + 15), (0, 255, 0), -1)
        # Print the mapped label name directly above the matching progress bar
        cv2.putText(frame, f"{label}", (15, y_offset - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 45 # Shift pointer down to make space for next bar

    # Step C: Render active facial blendshapes (e.g., Smile or Surprise thresholds)
    for name, score in expressions.items():
        # Draw the expression tracking slot (Gray baseline container)
        cv2.rectangle(frame, (15, y_offset), (165, y_offset + 15), (50, 50, 50), -1)
        # Draw progress bar indicating activation strength (Orange fill)
        cv2.rectangle(frame, (15, y_offset), (15 + int(150 * score), y_offset + 15), (255, 150, 0), -1)
        # Print the name of the expression category above the bar
        cv2.putText(frame, f"{name.upper()}", (15, y_offset - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 45

# ==============================================================================
# FUNCTION: run_system (Main Pipeline Execution Thread)
# ==============================================================================
def run_system():
    # --------------------------------------------------------------------------
    # PART A: INITIALIZE ADVANCED MEDIAPIPE TASK SOLUTIONS
    # --------------------------------------------------------------------------
    # Setup and configuration for the Hand Gesture Recognizer
    base_gesture = python.BaseOptions(model_asset_path=GESTURE_MODEL)
    gesture_rec = vision.GestureRecognizer.create_from_options(
        vision.GestureRecognizerOptions(base_options=base_gesture, num_hands=2))

    # Setup and configuration for Face Landmarker (Enables detailed blendshape weights)
    base_face = python.BaseOptions(model_asset_path=FACE_MODEL)
    face_landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(base_options=base_face, output_face_blendshapes=True))

    # Bind the system to default camera stream
    cap = cv2.VideoCapture(0)

    # --------------------------------------------------------------------------
    # PART B: FRAME PROCESSING LOOP
    # --------------------------------------------------------------------------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break # Terminate loop if webcam input fails or is interrupted
        
        # Mirror the frame horizontally so tracking matches standard human mirror reflexes
        frame = cv2.flip(frame, 1)
        
        # Convert native OpenCV color spaces (BGR) to the MediaPipe compatible profile (RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert image matrix into a standardized MediaPipe Image type wrapper
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # --------------------------------------------------------------------------
        # PART C: EXECUTE MODEL INFERENCE (PREDICTIONS)
        # --------------------------------------------------------------------------
        # Feed the frame to the underlying gesture recognition engine and face landmarker
        g_result = gesture_rec.recognize(mp_image)
        f_result = face_landmarker.detect(mp_image)

        # --------------------------------------------------------------------------
        # PART D: GATHER & ARRANGE HAND TRACKING DETAILS
        # --------------------------------------------------------------------------
        active_gestures = []
        if g_result.gestures:
            # Process results independently for each detected hand on screen (up to 2)
            for i, hand in enumerate(g_result.gestures):
                top = hand[0] # Focus on classification outcome with highest confidence
                
                # Check that active gesture meets classification parameters (is not None)
                if top.category_name != "None":
                    active_gestures.append({'label': top.category_name, 'score': top.score})
                
                # Plot customized skeletal hand trace and landmarks onto the camera frame
                draw_manual_landmarks(frame, g_result.hand_landmarks[i])

        # --------------------------------------------------------------------------
        # PART E: GATHER & ARRANGE FACIAL EXPRESSION DETAILS
        # --------------------------------------------------------------------------
        active_expr = {}
        if f_result.face_blendshapes:
            # Convert facial action scores (blendshapes) into a key-value dictionary
            shapes = {s.category_name: s.score for s in f_result.face_blendshapes[0]}
            
            # Smile Detection: Triggers if Left Mouth Corner retracts (threshold: >30% intensity)
            if shapes.get('mouthSmileLeft', 0) > 0.3: 
                active_expr['Smile'] = shapes['mouthSmileLeft']
                
            # Surprise Detection: Triggers if Inner Eyebrows shift upwards (threshold: >40% intensity)
            if shapes.get('browInnerUp', 0) > 0.4: 
                active_expr['Surprise'] = shapes['browInnerUp']

        # --------------------------------------------------------------------------
        # PART F: UPDATE USER UI INTERFACE & RENDER
        # --------------------------------------------------------------------------
        # Redraw status panels, text, and data progress overlays
        draw_ui_panel(frame, active_gestures, active_expr)

        # Output frame buffer to the render window
        cv2.imshow('No-Solutions Vision Prototype', frame)
        
        # Gracefully break thread if user hits "Esc" key (ASCII character 27)
        if cv2.waitKey(1) & 0xFF == 27: break

    # --------------------------------------------------------------------------
    # PART G: SYSTEM DISMANTLE & HOUSEKEEPING
    # --------------------------------------------------------------------------
    cap.release() # Relinquish hold on the system camera
    cv2.destroyAllWindows() # Clear computer memory buffers holding screen frames

if __name__ == "__main__":
    run_system()