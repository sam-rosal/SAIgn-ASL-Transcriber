import cv2
import mediapipe as mp
import numpy as np
import os
import sqlite3

# Define relative project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, 'saign_vision.db')
DATASET_DIR = os.path.join(SCRIPT_DIR, 'media_dataset')
FEATURES_DIR = os.path.join(SCRIPT_DIR, 'extracted_features')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands_extractor = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def extract_video_landmarks(video_path):
    """Processes a video file and converts hands into raw coordinate points."""
    cap = cv2.VideoCapture(video_path)
    video_features = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_extractor.process(rgb_frame)
        
        # Space for up to 2 hands (2 hands * 21 landmarks * 3 coords [x,y,z] = 126 floats)
        frame_coordinates = np.zeros(126) 
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                
                start_idx = idx * 63
                frame_coordinates[start_idx : start_idx + len(coords)] = coords
                
        video_features.append(frame_coordinates)
        
    cap.release()
    return np.array(video_features)

def register_and_process_dataset():
    """Scans media_dataset folders, extracts points, and logs everything in SQLite."""
    os.makedirs(FEATURES_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Iterate over every sub-folder inside media_dataset/
    for label_dir in os.listdir(DATASET_DIR):
        label_path = os.path.join(DATASET_DIR, label_dir)
        if not os.path.isdir(label_path): continue
            
        for video_file in os.listdir(label_path):
            if not video_file.lower().endswith(('.mp4', '.avi', '.mov')): continue
                
            video_path = os.path.join(label_path, video_file)
            feature_filename = f"{os.path.splitext(video_file)[0]}_landmarks.npy"
            output_feature_path = os.path.join(FEATURES_DIR, feature_filename)
            
            print(f"Processing media: [{label_dir}] -> {video_file}...")
            
            # Step A: Extract landmarks
            features_array = extract_video_landmarks(video_path)
            if len(features_array) == 0: continue
                
            # Step B: Save numpy features to disk
            np.save(output_feature_path, features_array)
            
            # Step C: Upload metadata entry into SQLite!
            cursor.execute("""
                INSERT OR REPLACE INTO training_dataset 
                (video_path, label, num_frames, extracted_features_path)
                VALUES (?, ?, ?, ?);
            """, (video_path, label_dir, len(features_array), output_feature_path))
            
    conn.commit()
    conn.close()
    print("\nAll videos processed, features extracted, and entries saved into the SQLite DB!")

if __name__ == "__main__":
    register_and_process_dataset()