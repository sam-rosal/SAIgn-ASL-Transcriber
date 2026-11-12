import cv2
import mediapipe as mp
import numpy as np
import os
import sqlite3
from scipy.interpolate import interp1d

# --- NEW IMPORTS FOR MODEL TRAINING ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# --- System & Path Configurations ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, 'saign_vision.db')
DATASET_DIR = os.path.join(SCRIPT_DIR, 'media_dataset')
FEATURES_DIR = os.path.join(SCRIPT_DIR, 'extracted_features')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'asl_model.keras')

TARGET_FRAMES = 30  
VELOCITY_THRESHOLD = 0.015  
MIN_ACTIVE_FRAMES = 10     

# --- Initialize MediaPipe Holistic Pipeline ---
mp_holistic = mp.solutions.holistic
holistic_extractor = mp_holistic.Holistic(
    static_image_mode=False,        
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def extract_video_landmarks(video_path):
    """
    Reads a video frame-by-frame, runs MediaPipe Holistic tracking, and converts
    detected joint spatial coordinates into a structured NumPy matrix.

    Output matrix shape: (total_frames, 225)
    - Pose: 33 * 3 = 99
    - Left Hand: 21 * 3 = 63
    - Right Hand: 21 * 3 = 63
    """
    cap = cv2.VideoCapture(video_path)
    video_features = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic_extractor.process(rgb_frame)
        
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() \
            if results.pose_landmarks else np.zeros(33 * 3)
        
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
            if results.left_hand_landmarks else np.zeros(21 * 3)
        
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
            if results.right_hand_landmarks else np.zeros(21 * 3)
            
        frame_coordinates = np.concatenate([pose, lh, rh])
        video_features.append(frame_coordinates)
        
    cap.release()
    return np.array(video_features)


def apply_velocity_segmentation(features, threshold=VELOCITY_THRESHOLD, min_frames=MIN_ACTIVE_FRAMES):
    if len(features) < 2:
        return features
    frame_differences = features[1:] - features[:-1]
    velocities = np.linalg.norm(frame_differences, axis=1)
    active_mask = np.ones(len(features), dtype=bool)
    active_mask[1:] = velocities >= threshold
    filtered_features = features[active_mask]
    if len(filtered_features) < min_frames:
        return features
    return filtered_features


def resample_sequence(features, target_frames=TARGET_FRAMES):
    current_frames = len(features)
    if current_frames == target_frames:
        return features
    if current_frames < 2:
        pad_amount = max(0, target_frames - current_frames)
        return np.pad(features, ((0, pad_amount), (0, 0)), mode='edge')[:target_frames]
    x_old = np.linspace(0, 1, num=current_frames)
    x_new = np.linspace(0, 1, num=target_frames)
    interpolator = interp1d(x_old, features, axis=0, kind='linear')
    return interpolator(x_new)


def register_and_process_dataset():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    for label_dir in os.listdir(DATASET_DIR):
        label_path = os.path.join(DATASET_DIR, label_dir)
        if not os.path.isdir(label_path): 
            continue
            
        for video_file in os.listdir(label_path):
            if not video_file.lower().endswith(('.mp4', '.avi', '.mov')): 
                continue
                
            video_path = os.path.join(label_path, video_file)
            feature_filename = f"{os.path.splitext(video_file)[0]}_landmarks.npy"
            output_feature_path = os.path.join(FEATURES_DIR, feature_filename)
            
            # Use the existing filesystem check to skip redundancies
            if os.path.exists(output_feature_path):
                continue
            
            yt_id = video_file.split('_')[0] if '_' in video_file else video_file
            cursor.execute("""
                SELECT split FROM video_clips 
                WHERE file_title = ? OR url LIKE ? 
                LIMIT 1
            """, (video_file, f"%{yt_id}%"))
            result = cursor.fetchone()
            
            if not result:
                continue

            split_tag = result[0]
            raw_features = extract_video_landmarks(video_path)
            if len(raw_features) == 0: 
                continue
                
            segmented_features = apply_velocity_segmentation(raw_features)
            fixed_features = resample_sequence(segmented_features, target_frames=TARGET_FRAMES)
            
            np.save(output_feature_path, fixed_features)
            
            cursor.execute("""
                INSERT OR REPLACE INTO training_dataset 
                (video_path, label, num_frames, extracted_features_path, npy_path, split)
                VALUES (?, ?, ?, ?, ?, ?);
            """, (video_path, label_dir, len(fixed_features), output_feature_path, output_feature_path, split_tag))
            
    conn.commit()
    conn.close()


def load_split_data(split_name, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = "SELECT npy_path, label FROM training_dataset WHERE split = ?"
    data = cursor.execute(query, (split_name,)).fetchall()
    conn.close()
    return data

def prepare_tensors(data_list):
    """
    Reads the .npy files from the disk and structures them into X (features) and y (labels) lists.
    """
    X, y = [], []
    for npy_path, label in data_list:
        if os.path.exists(npy_path):
            features = np.load(npy_path)
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)


if __name__ == "__main__":
    # Step 1: Execute feature extraction pipeline
    print("Verifying feature extraction status...")
    register_and_process_dataset()

    # Step 2: Retrieve partitioned dataset samples for model training & evaluation
    print("\n--- Loading Dataset Splits from DB ---")
    train_data = load_split_data('train')
    val_data   = load_split_data('val')
    test_data  = load_split_data('test')

    print(f"Training clips found: {len(train_data)}")
    print(f"Validation clips found: {len(val_data)}")
    print(f"Testing clips found: {len(test_data)}")

    if len(train_data) == 0:
        print("No training data found. Exiting.")
        exit()

    # Step 3: Convert File Paths to NumPy Matrices
    print("\nLoading NumPy features into memory...")
    X_train, y_train_str = prepare_tensors(train_data)
    X_val, y_val_str = prepare_tensors(val_data)
    # Keeping X_test separate for later benchmarking
    X_test, y_test_str = prepare_tensors(test_data) 

    # Step 4: One-Hot Encode the String Labels
    label_encoder = LabelEncoder()
    # Fit encoder on ALL labels to ensure categorical consistency across splits
    all_labels = np.concatenate([y_train_str, y_val_str, y_test_str])
    label_encoder.fit(all_labels)

    y_train = to_categorical(label_encoder.transform(y_train_str))
    y_val = to_categorical(label_encoder.transform(y_val_str))

    num_classes = len(label_encoder.classes_)
    sequence_length = X_train.shape[1]  # 30 frames
    feature_dim = X_train.shape[2]      # Dynamically sets to 225 based on Holistic extraction

    # Step 5: Define LSTM Architecture
    print("\n--- Building LSTM Neural Network ---")
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, feature_dim)),
        LSTM(128, return_sequences=False, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # Step 6: Define Early Stopping Callback
    early_stopping = EarlyStopping(
        monitor='val_loss',          # Watch the validation loss
        patience=15,                 # Allow 15 epochs without improvement before stopping
        restore_best_weights=True,   # Revert to the epoch with the best validation score
        verbose=1
    )

    # Step 7: Train the Model
    print("\n--- Commencing Training ---")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val), # Directly applying the validation split here
        epochs=150,
        callbacks=[early_stopping],     # Passing the callback to prevent overfitting
        batch_size=32
    )

    # Step 8: Save Model
    model.save(MODEL_PATH)
    print(f"\nModel successfully saved to {MODEL_PATH}")