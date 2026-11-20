import json
import os
import sqlite3
import sys
import tempfile
import cv2
import mediapipe as mp
import numpy as np
from scipy.interpolate import interp1d

# --- Model Training & Preprocessing Imports ---
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# --- System & Path Configurations ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, 'saign_vision.db')
DATASET_DIR = os.path.join(SCRIPT_DIR, 'media_dataset')
FEATURES_DIR = os.path.join(SCRIPT_DIR, 'extracted_features')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'asl_model.keras')
CLASSES_PATH = os.path.join(SCRIPT_DIR, 'classes.json')

TARGET_FRAMES = 30
FEATURE_DIM = 225  # Pose (33*3=99) + Left Hand (21*3=63) + Right Hand (21*3=63)
VELOCITY_THRESHOLD = 0.015
MIN_ACTIVE_FRAMES = 10
VALID_VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv')


class HolisticSession:
    """Manages MediaPipe Holistic lifecycle with automatic recovery if a video corrupts the graph."""

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self._mp_holistic = mp.solutions.holistic
        self._det_conf = min_detection_confidence
        self._trk_conf = min_tracking_confidence
        self._extractor = None

    def _create(self):
        return self._mp_holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=self._det_conf,
            min_tracking_confidence=self._trk_conf
        )

    def __enter__(self):
        self._extractor = self._create()
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
            self.close()
            raise


def extract_video_landmarks(video_path, holistic_session):
    """
    Reads a video frame-by-frame, runs MediaPipe Holistic tracking, and converts
    detected joint spatial coordinates into a structured NumPy matrix.

    Output matrix shape: (total_frames, 225)
    - Pose: 33 * 3 = 99
    - Left Hand: 21 * 3 = 63
    - Right Hand: 21 * 3 = 63
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

            pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark], dtype=np.float32).flatten() \
                if results.pose_landmarks else np.zeros(33 * 3, dtype=np.float32)

            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark], dtype=np.float32).flatten() \
                if results.left_hand_landmarks else np.zeros(21 * 3, dtype=np.float32)

            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark], dtype=np.float32).flatten() \
                if results.right_hand_landmarks else np.zeros(21 * 3, dtype=np.float32)

            frame_coordinates = np.concatenate([pose, lh, rh])
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
    """Parses <tag> directly from <sign>_<number>_<tag>.mp4, with safe DB fallback for legacy names."""
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
            return result[0].strip().lower()
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
    """Checks that an existing .npy file is readable, finite, and matches (TARGET_FRAMES, FEATURE_DIM)."""
    if not os.path.exists(npy_path):
        return False
    try:
        arr = np.load(npy_path)
        return arr.shape == (TARGET_FRAMES, FEATURE_DIM) and bool(np.isfinite(arr).all())
    except (OSError, ValueError):
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


def resolve_npy_path(npy_path):
    """Resolves .npy file path even if the project folder was moved since extraction."""
    if not npy_path:
        return None
    if os.path.exists(npy_path):
        return npy_path
    portable_path = os.path.join(FEATURES_DIR, os.path.basename(npy_path))
    return portable_path if os.path.exists(portable_path) else None


def register_and_process_dataset():
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR, exist_ok=True)
        print(f"[WARN] Dataset directory was missing and created at {DATASET_DIR}.")
        return

    os.makedirs(FEATURES_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    try:
        cursor = conn.cursor()
        ensure_tables_exist(cursor)
        prune_orphaned_db_rows(cursor)
        conn.commit()

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
                        continue

                    # Re-sync DB row if valid 225-dim features already exist on disk
                    if is_valid_cached_npy(output_feature_path):
                        upsert_training_record(cursor, video_path, label_dir, TARGET_FRAMES, output_feature_path, split_tag)
                        conn.commit()
                        continue

                    holistic_session.reset_or_recreate()
                    raw_features = extract_video_landmarks(video_path, holistic_session)
                    if len(raw_features) == 0:
                        continue

                    segmented_features = apply_velocity_segmentation(raw_features)
                    fixed_features = resample_sequence(segmented_features, target_frames=TARGET_FRAMES)

                    save_npy_atomic(output_feature_path, fixed_features)
                    upsert_training_record(cursor, video_path, label_dir, len(fixed_features), output_feature_path, split_tag)
                    conn.commit()
    finally:
        conn.close()


def load_split_data(split_name, db_path=DB_PATH):
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        cursor = conn.cursor()
        query = "SELECT npy_path, label FROM training_dataset WHERE split = ? AND npy_path IS NOT NULL AND label IS NOT NULL"
        return cursor.execute(query, (split_name,)).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def prepare_tensors(data_list, expected_shape=(TARGET_FRAMES, FEATURE_DIM)):
    """
    Reads .npy files from disk and structures them into uniform X (features) and y (labels) arrays.
    """
    X, y = [], []
    skipped = 0

    for raw_npy_path, label in data_list:
        npy_path = resolve_npy_path(raw_npy_path)
        if not npy_path:
            skipped += 1
            continue

        try:
            features = np.load(npy_path)
        except (OSError, ValueError) as e:
            print(f"[WARN] Failed to load '{npy_path}': {e}. Skipping...")
            skipped += 1
            continue

        if features.shape != expected_shape:
            print(f"[WARN] Shape mismatch in '{npy_path}': got {features.shape}, expected {expected_shape}. Skipping...")
            skipped += 1
            continue

        features = np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        X.append(features)
        y.append(label)

    if skipped > 0:
        print(f"[WARN] Skipped {skipped} missing, corrupted, or shape-mismatched feature file(s).")

    if not X:
        return np.empty((0, *expected_shape), dtype=np.float32), np.array([], dtype=str)

    return np.array(X, dtype=np.float32), np.array(y)


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
        sys.exit(1)

    # Step 3: Convert File Paths to NumPy Matrices
    print("\nLoading NumPy features into memory...")
    X_train, y_train_str = prepare_tensors(train_data)
    X_val, y_val_str     = prepare_tensors(val_data)

    if len(X_train) == 0:
        raise RuntimeError(
            "Zero valid training feature tensors could be loaded from disk. "
            "Verify that .npy files exist in 'extracted_features/' with shape (30, 225)."
        )

    # Step 4: One-Hot Encode Trained String Labels
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train_str)
    train_label_set = set(label_encoder.classes_.tolist())

    num_classes = len(label_encoder.classes_)
    if num_classes < 2:
        print(f"[WARN] Only {num_classes} unique sign class found in training split. Add at least 2 classes for meaningful classification.")

    y_train = to_categorical(label_encoder.transform(y_train_str), num_classes=num_classes)

    if len(X_val) > 0:
        known_val_mask = np.array([lbl in train_label_set for lbl in y_val_str], dtype=bool)
        unseen_val_labels = sorted(set(y_val_str[~known_val_mask].tolist()))
        if unseen_val_labels:
            print(
                f"[WARN] Excluding {int((~known_val_mask).sum())} validation sample(s) across "
                f"{len(unseen_val_labels)} class(es) with 0 training clips: {unseen_val_labels[:10]}"
            )
            X_val = X_val[known_val_mask]
            y_val_str = y_val_str[known_val_mask]

    has_validation = len(X_val) > 0
    if has_validation:
        y_val = to_categorical(label_encoder.transform(y_val_str), num_classes=num_classes)
        validation_tuple = (X_val, y_val)
        monitor_metric = 'val_loss'
    else:
        print("[WARN] No valid validation tensors matching trained classes found. Monitoring training 'loss'.")
        validation_tuple = None
        monitor_metric = 'loss'

    sequence_length = X_train.shape[1]  # 30 frames
    feature_dim = X_train.shape[2]      # 225 Holistic features

    # Step 5: Define LSTM + Temporal Attention Architecture
    print("\n--- Building LSTM + Attention Neural Network ---")
    inputs = Input(shape=(sequence_length, feature_dim))

    # Recurrent layers retain full 30-step temporal sequences (return_sequences=True)
    x = LSTM(64, return_sequences=True)(inputs)
    x = LSTM(128, return_sequences=True)(x)

    # Self-Attention layer weights key gesture frames across the 30-frame window
    attention_out = Attention()([x, x])

    # Condense temporal sequence into a single context vector before classification
    x = GlobalAveragePooling1D()(attention_out)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # Step 6: Define Early Stopping Callback
    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    # Step 7: Train the Model
    print("\n--- Commencing Training ---")
    history = model.fit(
        X_train,
        y_train,
        validation_data=validation_tuple,
        epochs=150,
        callbacks=[early_stopping],
        batch_size=32
    )

    # Step 8: Save Model & Frozen Class Index Mapping
    model.save(MODEL_PATH)
    with open(CLASSES_PATH, 'w', encoding='utf-8') as f:
        json.dump(label_encoder.classes_.tolist(), f, indent=2)

    print(f"\nModel successfully saved to {MODEL_PATH}")
    print(f"Class label mapping ({num_classes} classes) saved to {CLASSES_PATH}")