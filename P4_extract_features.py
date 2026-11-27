import os
import sqlite3
import tempfile
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


class HolisticSession:
    """Manages MediaPipe Holistic lifecycle with automatic recovery if a video corrupts the graph."""

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Nothing MediaPipe-related is touched here: the graph is created lazily, only when a video
        # actually needs extraction. Callers that only use already-extracted features (e.g. the training
        # script) then work even if this MediaPipe build has no legacy 'mp.solutions' API.
        self._det_conf = min_detection_confidence
        self._trk_conf = min_tracking_confidence
        self._extractor = None

    def _create(self):
        if not hasattr(mp, 'solutions'):
            raise ImportError(
                "This MediaPipe build does not expose the legacy 'mp.solutions' API, so new videos cannot "
                "be processed. Install a version that does, e.g.: pip install mediapipe==0.10.14"
            )
        return mp.solutions.holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=self._det_conf,
            min_tracking_confidence=self._trk_conf
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self._extractor is not None:
            try:
                self._extractor.close()
            except Exception:
                pass
            self._extractor = None

    def reset_or_recreate(self):
        if self._extractor is None:
            self._extractor = self._create()
            return
        try:
            self._extractor.reset()
        except Exception:
            self.close()
            self._extractor = self._create()

    def process(self, rgb_frame):
        if self._extractor is None:
            self._extractor = self._create()
        try:
            return self._extractor.process(rgb_frame)
        except Exception:
            # Close poisoned C++ graph before re-raising so subsequent videos stay clean
            self.close()
            raise


def extract_frame_landmarks(results):
    """
    Flattens ONE MediaPipe Holistic result into a single float32 vector of FEATURE_DIM (225) values:
    pose (33*3) + left hand (21*3) + right hand (21*3). A body part that was not detected becomes zeros.

    Single source of truth for the per-frame feature layout: training-time extraction
    (extract_video_landmarks, below) AND live inference (P4_run_system.py) both call this, so the two
    can never encode a frame differently.
    """
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark], dtype=np.float32).flatten() \
        if results.pose_landmarks else np.zeros(33 * 3, dtype=np.float32)

    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark], dtype=np.float32).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3, dtype=np.float32)

    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark], dtype=np.float32).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3, dtype=np.float32)

    return np.nan_to_num(np.concatenate([pose, lh, rh]), nan=0.0, posinf=0.0, neginf=0.0)


def extract_video_landmarks(video_path, holistic_session):
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
            if frame.size == 0:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic_session.process(rgb_frame)

            frame_coordinates = extract_frame_landmarks(results)
            video_features.append(frame_coordinates)
    except Exception as e:
        print(f"[WARN] MediaPipe processing error on '{video_path}': {e}. Skipping...")
        return np.empty((0, FEATURE_DIM), dtype=np.float32)
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

    if current_frames == 0:
        feat_cols = features.shape[1] if features.ndim == 2 else FEATURE_DIM
        return np.zeros((target_frames, feat_cols), dtype=np.float32)

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
    """Ensures training_dataset and its required columns/indexes exist even on legacy databases."""
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

    existing_cols = {
        row[1] for row in cursor.execute("PRAGMA table_info(training_dataset);").fetchall()
    }
    required_columns = {
        'video_path': "TEXT",
        'label': "TEXT",
        'num_frames': "INTEGER",
        'clip_id': "INTEGER",
        'extracted_features_path': "TEXT",
        'npy_path': "TEXT",
        'split': "TEXT NOT NULL DEFAULT 'train'"
    }
    for col_name, col_def in required_columns.items():
        if col_name not in existing_cols:
            cursor.execute(f"ALTER TABLE training_dataset ADD COLUMN {col_name} {col_def};")

    cursor.execute("""
        DELETE FROM training_dataset
        WHERE id NOT IN (
            SELECT MAX(id)
            FROM training_dataset
            GROUP BY video_path
        );
    """)
    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_training_dataset_video_path
        ON training_dataset(video_path);
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_dataset_split ON training_dataset(split);")


def prune_orphaned_db_rows(cursor):
    """Removes training_dataset entries whose source video file no longer exists on disk."""
    rows = cursor.execute("SELECT id, video_path FROM training_dataset").fetchall()
    missing_ids = [(row_id,) for row_id, vpath in rows if not vpath or not os.path.exists(vpath)]
    if missing_ids:
        cursor.executemany("DELETE FROM training_dataset WHERE id = ?", missing_ids)
        print(f"[INFO] Pruned {len(missing_ids)} orphaned database row(s) whose video files were removed.")


def upsert_training_record(cursor, video_path, label, num_frames, feature_path, split_tag):
    cursor.execute("""
        INSERT INTO training_dataset
            (video_path, label, num_frames, extracted_features_path, npy_path, split)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(video_path) DO UPDATE SET
            label = excluded.label,
            num_frames = excluded.num_frames,
            extracted_features_path = excluded.extracted_features_path,
            npy_path = excluded.npy_path,
            split = excluded.split;
    """, (video_path, label, num_frames, feature_path, feature_path, split_tag))


def resolve_split_tag(cursor, video_file):
    """Parses <tag> from <sign>_<number>_<tag>.mp4, with safe DB fallback for legacy names."""
    stem_parts = os.path.splitext(video_file)[0].split('_')
    if len(stem_parts) >= 3 and stem_parts[-1].lower() in ('train', 'val', 'test'):
        return stem_parts[-1].lower()

    yt_id = stem_parts[0]
    try:
        if len(yt_id) == 11:
            cursor.execute("""
                SELECT split FROM video_clips
                WHERE file_title = ? OR url LIKE ?
                LIMIT 1
            """, (video_file, f"%{yt_id}%"))
        else:
            cursor.execute("""
                SELECT split FROM video_clips
                WHERE file_title = ?
                LIMIT 1
            """, (video_file,))
        result = cursor.fetchone()
        if result and result[0]:
            split = str(result[0]).strip().lower()
            return {'validation': 'val', 'valid': 'val', 'training': 'train'}.get(split, split)
    except sqlite3.OperationalError:
        pass

    return None


def build_feature_output_path(label_dir, video_file):
    """Prevents filename collisions in extracted_features/ across different sign directories."""
    stem = os.path.splitext(video_file)[0]
    if stem.lower().startswith(f"{label_dir.lower()}_"):
        feature_filename = f"{stem}_landmarks.npy"
    else:
        feature_filename = f"{label_dir}_{stem}_landmarks.npy"
    return os.path.join(FEATURES_DIR, feature_filename)


def is_valid_cached_npy(npy_path):
    """Verifies an existing .npy file is readable, finite, and matches (TARGET_FRAMES, FEATURE_DIM)."""
    if not os.path.exists(npy_path):
        return False
    try:
        arr = np.load(npy_path)
        return arr.shape == (TARGET_FRAMES, FEATURE_DIM) and bool(np.isfinite(arr).all())
    except Exception:
        # Corrupt/empty/truncated files can raise EOFError, UnpicklingError, etc. -> just re-extract.
        return False


def save_npy_atomic(output_path, array):
    """Writes a NumPy array atomically so interrupted runs cannot leave truncated .npy files."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(output_path), suffix='.tmp.npy')
    os.close(fd)
    try:
        np.save(tmp_path, array)
        os.replace(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def register_and_process_dataset(verbose_skips=True):
    """
    Extracts/caches 30x225 landmark features for every video in media_dataset/ and syncs the
    training_dataset table. Already-extracted videos are only re-synced to the database.

    verbose_skips: print a line for every already-extracted / untagged video (set False when called
    from the training script to keep its output readable on large datasets).
    """
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR, exist_ok=True)
        print(f"[WARN] Created empty dataset directory at {DATASET_DIR}. Add sign folders and videos before running extraction.")
        return

    os.makedirs(FEATURES_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    try:
        cursor = conn.cursor()
        ensure_tables_exist(cursor)
        prune_orphaned_db_rows(cursor)
        conn.commit()

        extraction_available = True
        no_split_count = 0
        unextracted_count = 0
        failed_count = 0

        with HolisticSession(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic_session:
            for label_dir in sorted(os.listdir(DATASET_DIR)):
                label_path = os.path.join(DATASET_DIR, label_dir)
                if not os.path.isdir(label_path) or label_dir.startswith('.'):
                    continue

                for video_file in sorted(os.listdir(label_path)):
                    if not video_file.lower().endswith(VALID_VIDEO_EXTS):
                        continue

                    video_path = os.path.join(label_path, video_file)
                    output_feature_path = build_feature_output_path(label_dir, video_file)

                    split_tag = resolve_split_tag(cursor, video_file)
                    if not split_tag:
                        no_split_count += 1
                        if verbose_skips:
                            print(f"[WARN] No matching split metadata for: {video_file}. Skipping processing...")
                        continue

                    # If valid 225-dim features already exist, re-sync DB row immediately
                    if is_valid_cached_npy(output_feature_path):
                        upsert_training_record(cursor, video_path, label_dir, TARGET_FRAMES, output_feature_path, split_tag)
                        conn.commit()
                        if verbose_skips:
                            print(f"Skipping {video_file} - valid 225-dim features already extracted (DB synced).")
                        continue

                    if not extraction_available:
                        unextracted_count += 1
                        continue
                    try:
                        holistic_session.reset_or_recreate()
                    except ImportError as e:
                        print(f"[ERROR] {e}")
                        print("[WARN] Feature extraction is unavailable; continuing with already-extracted features only.")
                        extraction_available = False
                        unextracted_count += 1
                        continue

                    print(f"Processing media: [{label_dir}] -> {video_file} (Split: {split_tag})...")
                    try:
                        raw_features = extract_video_landmarks(video_path, holistic_session)
                        if len(raw_features) == 0:
                            print(f"[WARN] No frames extracted from {video_file}. Skipping...")
                            continue

                        segmented_features = apply_velocity_segmentation(raw_features)
                        fixed_features = resample_sequence(segmented_features, target_frames=TARGET_FRAMES)

                        save_npy_atomic(output_feature_path, fixed_features)
                        upsert_training_record(cursor, video_path, label_dir, len(fixed_features), output_feature_path, split_tag)
                        conn.commit()
                    except Exception as exc:
                        # A disk/permission error, SQLite lock, or any other unexpected failure on ONE
                        # clip must never abort an hours-long batch run; log it and keep going.
                        print(f"[ERROR] Failed to process {video_file}: {type(exc).__name__}: {exc}. Continuing...")
                        failed_count += 1

        if no_split_count and not verbose_skips:
            print(f"[WARN] {no_split_count} video(s) skipped: no train/val/test tag in the filename or video_clips table.")
        if unextracted_count:
            print(f"[WARN] {unextracted_count} video(s) were NOT extracted and will be missing from training.")
        if failed_count:
            print(f"[WARN] {failed_count} video(s) failed to process due to an unexpected error and were skipped.")
    finally:
        conn.close()

    print("\nHolistic feature extraction & 30-Frame Resampling complete! 225-dimension entries logged.")


if __name__ == "__main__":
    register_and_process_dataset()