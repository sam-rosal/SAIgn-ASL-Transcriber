import cv2
import mediapipe as mp
import numpy as np
import os
import sqlite3
from scipy.interpolate import interp1d

# --- System & Path Configurations ---
# Resolve absolute directory path where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Filepaths for database connection, raw video inputs, and output feature binaries
DB_PATH = os.path.join(SCRIPT_DIR, 'saign_vision.db')
DATASET_DIR = os.path.join(SCRIPT_DIR, 'media_dataset')
FEATURES_DIR = os.path.join(SCRIPT_DIR, 'extracted_features')

# --- Sequence Length Configuration ---
# TARGET_FRAMES: Fixed frame length required for temporal model input standardization.
TARGET_FRAMES = 30  

# --- Velocity Segmentation Configuration Parameters ---
# VELOCITY_THRESHOLD: Minimum Euclidean distance (L2 norm displacement) between consecutive landmark vectors required to qualify a frame as active motion.
VELOCITY_THRESHOLD = 0.015  

# MIN_ACTIVE_FRAMES: Fallback guardrail to ensure fast or brief signs aren't 
# over-filtered below a usable sequence length.
MIN_ACTIVE_FRAMES = 10     

# --- Initialize MediaPipe Hands Pipeline ---
mp_hands = mp.solutions.hands
hands_extractor = mp_hands.Hands(
    static_image_mode=False,        # Continuous video stream mode for temporal tracking optimizations
    max_num_hands=2,               # Detect up to 2 hands simultaneously
    min_detection_confidence=0.5   # Minimum detection confidence threshold
)


def extract_video_landmarks(video_path):
    """
    Reads a video frame-by-frame, runs MediaPipe hand tracking, and converts
    detected joint spatial coordinates into a structured NumPy matrix.

    Output matrix shape: (total_frames, 126)
    - 2 hands * 21 landmarks * 3 coordinates (x, y, z) = 126 float values per frame.
    """
    cap = cv2.VideoCapture(video_path)
    video_features = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
        
        # OpenCV reads BGR; MediaPipe requires RGB frame inputs
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_extractor.process(rgb_frame)
        
        # Pre-allocate a zero array for up to 2 hands (2 * 21 * 3 = 126 elements)
        # Unused hand slots remain padded with 0.0 if < 2 hands are detected
        frame_coordinates = np.zeros(126) 
        
        # Populate coordinates if hands are detected
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                
                # Assign to slice: Hand 1 occupies [0:63], Hand 2 occupies [63:126]
                start_idx = idx * 63
                frame_coordinates[start_idx : start_idx + len(coords)] = coords
                
        video_features.append(frame_coordinates)
        
    cap.release()
    return np.array(video_features)


def apply_velocity_segmentation(features, threshold=VELOCITY_THRESHOLD, min_frames=MIN_ACTIVE_FRAMES):
    """
    Filters out static transition frames (e.g., hand at rest or moving into position)
    by computing frame-to-frame Euclidean landmark displacement (velocity).

    Mathematical Pipeline:
    1. Calculate delta vector between adjacent time steps: ΔF = F[t] - F[t-1]
    2. Compute L2 norm (magnitude) of ΔF across landmark dimensions.
    3. Retain frames where spatial displacement meets or exceeds the velocity threshold.
    """
    # Cannot calculate velocity across fewer than 2 frames
    if len(features) < 2:
        return features

    # Step 1: Compute coordinate differences between adjacent frames
    frame_differences = features[1:] - features[:-1]
    
    # Step 2: Calculate Euclidean displacement magnitude (L2 norm) per frame step
    velocities = np.linalg.norm(frame_differences, axis=1)

    # Step 3: Construct boolean mask; keep initial frame, then check threshold
    active_mask = np.ones(len(features), dtype=bool)
    active_mask[1:] = velocities >= threshold

    # Filter out static frames
    filtered_features = features[active_mask]

    # Fallback guardrail: If filtering dropped too many frames, revert to raw sequence
    if len(filtered_features) < min_frames:
        return features

    return filtered_features


def resample_sequence(features, target_frames=TARGET_FRAMES):
    """
    Resamples a keypoint sequence matrix of shape (T, 126) to exactly (target_frames, 126)
    using 1D linear temporal interpolation across the time axis.

    Mathematical Pipeline:
    1. Map original sequence indices [0, ..., T-1] to normalized timeline [0.0, 1.0].
    2. Generate target timeline [0.0, 1.0] with exactly target_frames steps.
    3. Linearly interpolate features across time axis (axis=0) to standardize shape.
    """
    current_frames = len(features)
    
    if current_frames == target_frames:
        return features

    if current_frames < 2:
        # Fallback padding if video yields fewer than 2 active frames
        pad_amount = max(0, target_frames - current_frames)
        return np.pad(features, ((0, pad_amount), (0, 0)), mode='edge')[:target_frames]

    # Map original and target time steps to normalized range [0.0, 1.0]
    x_old = np.linspace(0, 1, num=current_frames)
    x_new = np.linspace(0, 1, num=target_frames)
    
    # Perform 1D linear interpolation across temporal axis
    interpolator = interp1d(x_old, features, axis=0, kind='linear')
    return interpolator(x_new)


def register_and_process_dataset():
    """
    Scans dataset subdirectories, extracts landmarks, applies velocity segmentation, 
    resamples feature sequences to exactly 30 frames, saves feature matrices to disk (.npy), 
    and queries saign_vision.db to retrieve and associate official MS-ASL split tags ('train', 'val', 'test').
    """
    # Ensure features directory exists
    os.makedirs(FEATURES_DIR, exist_ok=True)
    
    # Connect to SQLite database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Iterate over sign label subdirectories (e.g., media_dataset/hello/)
    for label_dir in os.listdir(DATASET_DIR):
        label_path = os.path.join(DATASET_DIR, label_dir)
        if not os.path.isdir(label_path): 
            continue
            
        # Iterate over video files inside label directory
        for video_file in os.listdir(label_path):
            if not video_file.lower().endswith(('.mp4', '.avi', '.mov')): 
                continue
                
            video_path = os.path.join(label_path, video_file)
            
            # Construct output filepath for NumPy feature binary
            feature_filename = f"{os.path.splitext(video_file)[0]}_landmarks.npy"
            output_feature_path = os.path.join(FEATURES_DIR, feature_filename)
            
            # Retrieve partition split tag directly from video_clips table in saign_vision.db
            yt_id = video_file.split('_')[0] if '_' in video_file else video_file
            cursor.execute("""
                SELECT split FROM video_clips 
                WHERE file_title = ? OR url LIKE ? 
                LIMIT 1
            """, (video_file, f"%{yt_id}%"))
            result = cursor.fetchone()
            split_tag = result[0] if result else 'train'

            print(f"Processing media: [{label_dir}] -> {video_file} (Split: {split_tag})...")
            
            # Step 1: Extract 3D landmark coordinates frame-by-frame
            raw_features = extract_video_landmarks(video_path)
            if len(raw_features) == 0: 
                continue
                
            # Step 2: Apply Velocity Segmentation to remove idle frames
            segmented_features = apply_velocity_segmentation(raw_features)
            
            # Step 3: Apply Temporal Resampling to enforce fixed 30-frame sequence shape (30, 126)
            fixed_features = resample_sequence(segmented_features, target_frames=TARGET_FRAMES)
            
            # Step 4: Save processed feature array to disk
            np.save(output_feature_path, fixed_features)
            
            # Step 5: Index feature path, active frame count, and partition tag into SQLite DB
            cursor.execute("""
                INSERT OR REPLACE INTO training_dataset 
                (video_path, label, num_frames, extracted_features_path, npy_path, split)
                VALUES (?, ?, ?, ?, ?, ?);
            """, (
                video_path, 
                label_dir, 
                len(fixed_features), 
                output_feature_path, 
                output_feature_path, 
                split_tag
            ))
            
    # Commit database transactions and close connection
    conn.commit()
    conn.close()
    print("\nFeature extraction with Velocity Segmentation & 30-Frame Resampling complete! All entries logged in SQLite DB.")


def load_split_data(split_name, db_path=DB_PATH):
    """
    Queries the training_dataset table in SQLite DB to load extracted feature file paths (.npy)
    and associated sign labels for a specific data split ('train', 'val', or 'test').
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query training_dataset table for feature paths matching the split
    query = "SELECT npy_path, label FROM training_dataset WHERE split = ?"
    data = cursor.execute(query, (split_name,)).fetchall()
    conn.close()
    return data


if __name__ == "__main__":
    # Step 1: Execute feature extraction pipeline and log metadata to saign_vision.db
    register_and_process_dataset()

    # Step 2: Retrieve partitioned dataset samples for model training & evaluation
    print("\n--- Loading Dataset Splits from DB ---")
    train_data = load_split_data('train')
    val_data   = load_split_data('val')
    test_data  = load_split_data('test')

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Testing samples: {len(test_data)}")