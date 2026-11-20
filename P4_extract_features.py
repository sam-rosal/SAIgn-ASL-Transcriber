import os
import sqlite3
import cv2
import mediapipe as mp
import numpy as np
from scipy.interpolate import interp1d

# --- System & Path Configurations ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, 'saign_vision.db')
DATASET_DIR = os.path.join(SCRIPT_DIR, 'media_dataset')
FEATURES_DIR = os.path.join(SCRIPT_DIR, 'extracted_features')

TARGET_FRAMES = 30
FEATURE_DIM = 225  # Pose (33*3=99) + Left Hand (21*3=63) + Right Hand (21*3=63)
VELOCITY_THRESHOLD = 0.015
MIN_ACTIVE_FRAMES = 10
VALID_VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv')


def extract_video_landmarks(video_path, holistic_extractor):
    """
    Reads a video frame-by-frame, runs MediaPipe Holistic tracking, and converts
    detected joint spatial coordinates into a structured NumPy matrix.

    Output matrix shape: (total_frames, 225)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Could not open video file: {video_path}. Skipping...")
        return np.empty((0, FEATURE_DIM), dtype=np.float32)

    video_features = []
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic_extractor.process(rgb_frame)

            pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark], dtype=np.float32).flatten() \
                if results.pose_landmarks else np.zeros(33 * 3, dtype=np.float32)

            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark], dtype=np.float32).flatten() \
                if results.left_hand_landmarks else np.zeros(21 * 3, dtype=np.float32)

            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark], dtype=np.float32).flatten() \
                if results.right_hand_landmarks else np.zeros(21 * 3, dtype=np.float32)

            frame_coordinates = np.concatenate([pose, lh, rh])
            video_features.append(frame_coordinates)
    finally:
        cap.release()

    if not video_features:
        return np.empty((0, FEATURE_DIM), dtype=np.float32)

    return np.nan_to_num(np.array(video_features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


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
        return features.astype(np.float32)

    if current_frames < 2:
        pad_amount = max(0, target_frames - current_frames)
        return np.pad(features, ((0, pad_amount), (0, 0)), mode='edge')[:target_frames].astype(np.float32)

    x_old = np.linspace(0, 1, num=current_frames)
    x_new = np.linspace(0, 1, num=target_frames)

    interpolator = interp1d(x_old, features, axis=0, kind='linear')
    return interpolator(x_new).astype(np.float32)


def ensure_tables_exist(cursor):
    """Ensures training_dataset exists even if P4_setup_db_3.py was not run prior."""
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS training_dataset (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        clip_id INTEGER,
        video_path TEXT UNIQUE,
        label TEXT NOT NULL,
        num_frames INTEGER,
        extracted_features_path TEXT,
        npy_path TEXT,
        split TEXT NOT NULL DEFAULT 'train'
    );""")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_dataset_split ON training_dataset(split);")


def resolve_split_tag(cursor, video_file):
    """Parses <tag> from <sign>_<number>_<tag>.mp4, with safe DB fallback for legacy names."""
    stem_parts = os.path.splitext(video_file)[0].split('_')
    if len(stem_parts) >= 3 and stem_parts[-1].lower() in ('train', 'val', 'test'):
        return stem_parts[-1].lower()

    yt_id = stem_parts[0]
    try:
        cursor.execute("""
            SELECT split FROM video_clips
            WHERE file_title = ? OR url LIKE ?
            LIMIT 1
        """, (video_file, f"%{yt_id}%"))
        result = cursor.fetchone()
        if result and result[0]:
            return result[0].strip().lower()
    except sqlite3.OperationalError:
        pass

    return None


def is_valid_cached_npy(npy_path):
    """Verifies an existing .npy file is readable and matches (TARGET_FRAMES, FEATURE_DIM)."""
    if not os.path.exists(npy_path):
        return False
    try:
        arr = np.load(npy_path)
        return arr.shape == (TARGET_FRAMES, FEATURE_DIM)
    except (OSError, ValueError):
        return False


def register_and_process_dataset():
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR, exist_ok=True)
        print(f"[WARN] Created empty dataset directory at {DATASET_DIR}. Add sign folders and videos before running extraction.")
        return

    os.makedirs(FEATURES_DIR, exist_ok=True)
    mp_holistic = mp.solutions.holistic

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        ensure_tables_exist(cursor)
        conn.commit()

        with mp_holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic_extractor:

            for label_dir in sorted(os.listdir(DATASET_DIR)):
                label_path = os.path.join(DATASET_DIR, label_dir)
                if not os.path.isdir(label_path) or label_dir.startswith('.'):
                    continue

                for video_file in sorted(os.listdir(label_path)):
                    if not video_file.lower().endswith(VALID_VIDEO_EXTS):
                        continue

                    video_path = os.path.join(label_path, video_file)
                    feature_filename = f"{os.path.splitext(video_file)[0]}_landmarks.npy"
                    output_feature_path = os.path.join(FEATURES_DIR, feature_filename)

                    split_tag = resolve_split_tag(cursor, video_file)
                    if not split_tag:
                        print(f"[WARN] No matching split metadata for: {video_file}. Skipping processing...")
                        continue

                    # If valid 225-dim features already exist, re-sync DB row immediately
                    if is_valid_cached_npy(output_feature_path):
                        cursor.execute("""
                            INSERT OR REPLACE INTO training_dataset
                            (video_path, label, num_frames, extracted_features_path, npy_path, split)
                            VALUES (?, ?, ?, ?, ?, ?);
                        """, (video_path, label_dir, TARGET_FRAMES, output_feature_path, output_feature_path, split_tag))
                        conn.commit()
                        print(f"Skipping {video_file} - valid 225-dim features already extracted (DB synced).")
                        continue

                    print(f"Processing media: [{label_dir}] -> {video_file} (Split: {split_tag})...")

                    holistic_extractor.reset()
                    raw_features = extract_video_landmarks(video_path, holistic_extractor)
                    if len(raw_features) == 0:
                        print(f"[WARN] No frames extracted from {video_file}. Skipping...")
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
    finally:
        conn.close()

    print("\nHolistic feature extraction & 30-Frame Resampling complete! 225-dimension entries logged.")


if __name__ == "__main__":
    register_and_process_dataset()