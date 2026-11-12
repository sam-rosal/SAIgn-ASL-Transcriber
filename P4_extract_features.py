import cv2
import mediapipe as mp
import numpy as np
import os
import sqlite3
from scipy.interpolate import interp1d

# --- System & Path Configurations ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, 'saign_vision.db')
DATASET_DIR = os.path.join(SCRIPT_DIR, 'media_dataset')
FEATURES_DIR = os.path.join(SCRIPT_DIR, 'extracted_features')

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
    """
    Resamples a keypoint sequence matrix of shape (T, 225) to exactly (target_frames, 225)
    using 1D linear temporal interpolation across the time axis.
    """
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
            
            # --- NEW ADDITION: Check file system before processing ---
            if os.path.exists(output_feature_path):
                print(f"Skipping {video_file} - features already extracted.")
                continue
            # ---------------------------------------------------------
            
            yt_id = video_file.split('_')[0] if '_' in video_file else video_file
            cursor.execute("""
                SELECT split FROM video_clips 
                WHERE file_title = ? OR url LIKE ? 
                LIMIT 1
            """, (video_file, f"%{yt_id}%"))
            result = cursor.fetchone()
            
            if not result:
                print(f"[WARN] No matching metadata in video_clips for: {video_file}. Skipping processing...")
                continue

            split_tag = result[0]

            print(f"Processing media: [{label_dir}] -> {video_file} (Split: {split_tag})...")
            
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
    print("\nHolistic feature extraction & 30-Frame Resampling complete! 225-dimension entries logged.")

if __name__ == "__main__":
    register_and_process_dataset()