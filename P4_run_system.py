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
# Dynamically locate the script's directory to ensure relative paths resolve cleanly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths for SQLite database and trained Keras LSTM neural network model
DB_PATH = os.path.join(SCRIPT_DIR, 'saign_vision.db')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'asl_model.keras')

# Sequence buffer size expected by the LSTM network (30 consecutive video frames)
SEQUENCE_LENGTH = 30 

# ==============================================================================
# PREDICTION DEBOUNCING CONFIGURATION
# ==============================================================================
# Number of consecutive frame predictions tracked in the sliding history buffer
DEBOUNCE_WINDOW_SIZE = 8

# Minimum matching votes required within the history window to accept a gesture state update.
# e.g., 5 out of 8 predictions must agree before updating the display HUD.
DEBOUNCE_CONSENSUS_COUNT = 5


# ==============================================================================
# HELPER FUNCTIONS: DATABASE & DATA PROCESSING
# ==============================================================================
def load_db_cache():
    """
    Connects to SQLite and reads configuration records into memory.
    
    Caching these settings avoids executing SQL queries inside the real-time 
    video processing loop (30+ FPS), eliminating performance bottlenecks.

    Returns:
        tuple: (gesture_cache dict, expression_thresholds dict)
    """
    gesture_cache = {}
    expression_thresholds = {}

    # Check if the database file exists before attempting connection
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # 1. Fetch gesture mapping definitions and minimum confidence thresholds
        cursor.execute("SELECT raw_label, display_name, min_score FROM gesture_mappings;")
        for raw_label, display_name, min_score in cursor.fetchall():
            gesture_cache[raw_label] = {
                'display_name': display_name,
                'min_score': min_score
            }

        # 2. Fetch facial blendshape/expression activation thresholds
        cursor.execute("SELECT blendshape_name, display_name, activation_threshold FROM expression_thresholds;")
        for blendshape, display_name, threshold in cursor.fetchall():
            expression_thresholds[blendshape] = {
                'display_name': display_name,
                'threshold': threshold
            }

        conn.close()

    return gesture_cache, expression_thresholds


def get_confidence_color(score):
    """
    Computes a dynamic BGR color gradient transitioning from Blue to Red 
    based on the model's prediction confidence score.

    Mathematical mapping:
      - Score = 0.0 -> BGR(255, 0, 0)   [Pure Blue  - Low Confidence]
      - Score = 1.0 -> BGR(0, 0, 255)   [Pure Red   - High Confidence]

    Args:
        score (float): Prediction probability between 0.0 and 1.0

    Returns:
        tuple: BGR color tuple for OpenCV rendering
    """
    # Clamp the confidence score strictly between 0.0 and 1.0 to prevent color overflows
    score = max(0.0, min(1.0, float(score)))
    
    # Linear interpolation between channel intensities
    b = int((1.0 - score) * 255)  # Decreases as confidence rises
    r = int(score * 255)          # Increases as confidence rises
    g = 0                         # Green channel remains zero
    
    return (b, g, r)


def extract_landmarks(results):
    """
    Flattens spatial (X, Y, Z) coordinates from MediaPipe Holistic output 
    into a single 1D feature vector for input into the LSTM model.

    Landmark breakdown:
      - Pose: 33 landmarks * 3 spatial dimensions = 99 values
      - Left Hand: 21 landmarks * 3 spatial dimensions = 63 values
      - Right Hand: 21 landmarks * 3 spatial dimensions = 63 values
      - Total Feature Dimension = 225 values per frame

    Args:
        results: MediaPipe Holistic framework output object

    Returns:
        np.ndarray: Flattened 1D numpy array containing landmark spatial features
    """
    # Extract pose landmarks (fall back to zeros array if body not detected)
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 3)
    
    # Extract left hand landmarks (fall back to zeros array if hand not detected)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)
    
    # Extract right hand landmarks (fall back to zeros array if hand not detected)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)
    
    # Concatenate all keypoint arrays into a single continuous feature vector
    return np.concatenate([pose, lh, rh])


def detect_expressions(results, thresholds):
    """
    Calculates Euclidean distances between strategic facial landmark index pairs 
    to infer active expressions based on SQLite threshold rules.

    Args:
        results: MediaPipe Holistic framework output object
        thresholds (dict): Expression thresholds loaded from SQLite

    Returns:
        list: Active expression display labels (e.g., ['Eyebrows Raised'])
    """
    active_emotions = []
    
    # Return empty if face keypoints are missing or database thresholds aren't loaded
    if not results.face_landmarks or not thresholds:
        return active_emotions

    landmarks = results.face_landmarks.landmark

    # 1. Eyebrow Squint / Lowering: Distance between inner eyebrow and upper nose bridge
    if 'browLowerer' in thresholds or 'browSquint' in thresholds:
        brow_dist = abs(landmarks[70].y - landmarks[159].y)
        if 'browLowerer' in thresholds and brow_dist < thresholds['browLowerer']['threshold']:
            active_emotions.append(thresholds['browLowerer']['display_name'])
        elif 'browSquint' in thresholds and brow_dist < thresholds['browSquint']['threshold']:
            active_emotions.append(thresholds['browSquint']['display_name'])

    # 2. Eyebrow Raising: Distance between inner brow landmark and mid-nose landmark
    if 'browInnerUp' in thresholds:
        inner_brow = abs(landmarks[55].y - landmarks[6].y)
        if inner_brow > thresholds['browInnerUp']['threshold']:
            active_emotions.append(thresholds['browInnerUp']['display_name'])

    # 3. Smiling: Distance between left corner and right corner of lips
    if 'mouthSmileLeft' in thresholds:
        smile_dist = abs(landmarks[61].x - landmarks[291].x)
        if smile_dist > thresholds['mouthSmileLeft']['threshold']:
            active_emotions.append(thresholds['mouthSmileLeft']['display_name'])

    return active_emotions


# ==============================================================================
# GUI SCREENS & STATE MANAGERS
# ==============================================================================
def show_main_menu():
    """
    Creates and displays the graphical main menu canvas using OpenCV drawing utilities.
    Handles keyboard input to route user to either the vision loop or shutdown.

    Returns:
        str: User choice signal ('START' or 'QUIT')
    """
    # Create a blank black canvas image (600px height x 800px width x 3 color channels)
    menu_bg = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Draw Outer Menu Border & Containers
    cv2.rectangle(menu_bg, (50, 50), (750, 550), (35, 35, 35), -1)
    cv2.rectangle(menu_bg, (50, 50), (750, 550), (0, 215, 255), 2)  # Gold border

    # Draw Title and Subtitle Text
    cv2.putText(menu_bg, "SAIgn Vision Engine", (160, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(menu_bg, "Automated ASL & Expression HUD", (230, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 1, cv2.LINE_AA)

    # Option 1 Button Graphics: Start Engine
    cv2.rectangle(menu_bg, (150, 280), (650, 350), (50, 50, 50), -1)
    cv2.putText(menu_bg, "[ SPACE / ENTER ] Start Recognition", (170, 325), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 127), 2, cv2.LINE_AA)

    # Option 2 Button Graphics: Exit Program
    cv2.rectangle(menu_bg, (150, 380), (650, 450), (50, 50, 50), -1)
    cv2.putText(menu_bg, "[ ESC ] Shut Down Engine", (240, 425), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 127, 255), 2, cv2.LINE_AA)

    cv2.imshow('SAIgn - Main Menu', menu_bg)

    # Wait for menu interaction keypresses
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC Key ASCII code -> Shut down application completely
            cv2.destroyAllWindows()
            return 'QUIT'
        elif key in (13, 32):  # ENTER (13) or SPACE (32) ASCII code -> Launch Recognition Loop
            cv2.destroyWindow('SAIgn - Main Menu')
            return 'START'


def run_recognition_engine(model, gesture_cache, expression_thresholds):
    """
    Initializes hardware webcam stream and MediaPipe pipeline.
    Maintains a temporal sliding window buffer for real-time LSTM gesture classification
    and integrates prediction debouncing to prevent live display flicker.

    Args:
        model: Loaded Keras neural network model instance
        gesture_cache (dict): Dynamic gesture key and label database dictionary
        expression_thresholds (dict): Dynamic expression boundary dictionary
    """
    # Reconstruct gesture label index lookup array based on database keys
    labels = list(gesture_cache.keys()) if gesture_cache else ['UNKNOWN']
    
    # Initialize MediaPipe Holistic solutions and webcam capture interface
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    sequence_buffer = []  # Rolling temporal sequence buffer holding 30 landmark feature vectors
    
    # Prediction Debouncing History Queue:
    # Maintains the last N raw candidate predictions to form a majority vote consensus
    prediction_history = deque(maxlen=DEBOUNCE_WINDOW_SIZE)
    
    current_gesture = "Waiting..."
    confidence = 0.0

    # Instantiate holistic tracking model with confidence thresholds
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR frame from OpenCV to RGB for MediaPipe inference
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False  # Improve processing performance
            results = holistic.process(image)

            # Re-enable writing and convert back to BGR for rendering in OpenCV window
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Overlay spatial landmark points on screen
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            # Extract 1D feature vector and push onto sliding window queue
            keypoints = extract_landmarks(results)
            sequence_buffer.append(keypoints)
            sequence_buffer = sequence_buffer[-SEQUENCE_LENGTH:]  # Maintain max queue size = 30

            # Execute LSTM classification when window buffer contains 30 frames
            if len(sequence_buffer) == SEQUENCE_LENGTH:
                # Add batch dimension: shape (1, 30, 225)
                res = model.predict(np.expand_dims(sequence_buffer, axis=0), verbose=0)[0]
                prediction_idx = np.argmax(res)
                confidence = float(res[prediction_idx])

                raw_label = labels[prediction_idx] if prediction_idx < len(labels) else "UNKNOWN"
                
                # Fetch threshold & display mapping from memory cache (with safety fallback)
                cached_data = gesture_cache.get(raw_label, {
                    'display_name': raw_label.replace('_', ' ').title(),
                    'min_score': 0.40
                })

                # Validate model probability against database threshold to determine frame candidate
                if confidence >= cached_data['min_score']:
                    frame_candidate = cached_data['display_name']
                else:
                    frame_candidate = "Uncertain"

                # Append immediate single-frame prediction to debouncing history queue
                prediction_history.append(frame_candidate)

                # ==============================================================================
                # PREDICTION DEBOUNCING MECHANISM
                # ==============================================================================
                # Count the frequency of each predicted gesture label across the history window
                vote_counts = Counter(prediction_history)
                most_common_gesture, vote_count = vote_counts.most_common(1)[0]

                # Update output HUD state ONLY if candidate receives majority consensus
                if vote_count >= DEBOUNCE_CONSENSUS_COUNT:
                    current_gesture = most_common_gesture

            # Perform facial expression analysis
            active_expressions = detect_expressions(results, expression_thresholds)
            emotion_text = f"Emotion: {', '.join(active_expressions)}" if active_expressions else "Emotion: Neutral"

            # Compute dynamic HUD border color based on model prediction certainty
            hud_color = get_confidence_color(confidence)

            # Draw HUD Overlays (Background Box & Dynamic Colored Border)
            cv2.rectangle(image, (10, 10), (480, 110), (30, 30, 30), -1)
            cv2.rectangle(image, (10, 10), (480, 110), hud_color, 2)

            # Render Recognized Gesture and Emotion Text Lines
            cv2.putText(image, f"Sign: {current_gesture} ({confidence*100:.1f}%)", 
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, hud_color, 2, cv2.LINE_AA)
            cv2.putText(image, emotion_text, 
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

            # Display composite visual output window
            cv2.imshow('SAIgn - Live Recognition Engine', image)

            # Listen for Escape key press (ASCII 27) -> Stop capture and exit loop
            if cv2.waitKey(10) & 0xFF == 27:
                break

    # Clean up video capture hardware and close vision window
    cap.release()
    cv2.destroyWindow('SAIgn - Live Recognition Engine')


# ==============================================================================
# MAIN PROGRAM EXECUTION ENTRY POINT
# ==============================================================================
def main():
    """
    Main entry point for the application.
    Loads models, handles state machine transitions between Menu and Vision loop.
    """
    # 1. Ensure required neural network binary file exists on disk
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Please run P4_train_model.py to generate the trained model."
        )

    # Load compiled Keras LSTM model into memory
    model = load_model(MODEL_PATH)

    # 2. Main Program GUI State Machine Loop
    while True:
        # Reload SQLite cache on each main menu loop to catch database updates on-the-fly
        gesture_cache, expression_thresholds = load_db_cache()

        # Render menu and block until user inputs action choice
        action = show_main_menu()

        if action == 'QUIT':
            print("Exiting SAIgn Engine cleanly. Goodbye!")
            break
        elif action == 'START':
            # Launch webcam vision engine loop (runs until ESC is pressed)
            run_recognition_engine(model, gesture_cache, expression_thresholds)


if __name__ == "__main__":
    main()