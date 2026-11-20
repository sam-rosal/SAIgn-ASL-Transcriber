import sqlite3
import os

# Defining the database path using the file name
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, 'saign_vision.db')

# Defining the structured folders for organizing raw media dataset and extracted landmarks
DATASET_DIR = os.path.join(SCRIPT_DIR, 'media_dataset')
FEATURES_DIR = os.path.join(SCRIPT_DIR, 'extracted_features')


def ensure_training_dataset_columns(cursor):
    """Migrates older training_dataset tables so required columns exist before indexing."""
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


def init_system():
    # 1. Create Directories
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(FEATURES_DIR, exist_ok=True)

    # 2. Setup SQLite Tables
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()

        # Enable foreign key enforcement in SQLite
        cursor.execute("PRAGMA foreign_keys = ON;")

        # Parent MS-ASL Metadata Table (must exist before child foreign key inserts)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS video_clips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clean_text TEXT,
            label INTEGER,
            signer_id INTEGER,
            signer INTEGER,
            start_time REAL,
            end_time REAL,
            start_frame INTEGER,
            end_frame INTEGER,
            url TEXT,
            file_title TEXT,
            split TEXT,
            box_ymin REAL,
            box_xmin REAL,
            box_ymax REAL,
            box_xmax REAL
        );""")

        # Gesture Mapping Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS gesture_mappings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_label TEXT UNIQUE NOT NULL,
            display_name TEXT NOT NULL,
            min_score REAL DEFAULT 0.4
        );""")

        # Facial Blendshape Expression Thresholds Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS expression_thresholds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            blendshape_name TEXT UNIQUE NOT NULL,
            display_name TEXT NOT NULL,
            activation_threshold REAL NOT NULL
        );""")

        # Training Dataset Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_dataset (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clip_id INTEGER,
            video_path TEXT UNIQUE,
            label TEXT NOT NULL,
            num_frames INTEGER,
            extracted_features_path TEXT,
            npy_path TEXT,
            split TEXT NOT NULL DEFAULT 'train',
            FOREIGN KEY (clip_id) REFERENCES video_clips(id) ON DELETE SET NULL
        );""")

        # Ensure schema compatibility if an older DB file was already present
        ensure_training_dataset_columns(cursor)

        # Database Index for Fast Partition Querying
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_dataset_split ON training_dataset(split);
        """)

        # 3. Auto-seed gesture database directly from media_dataset subdirectories
        if os.path.exists(DATASET_DIR):
            valid_folders = [
                d for d in os.listdir(DATASET_DIR)
                if os.path.isdir(os.path.join(DATASET_DIR, d)) and not d.startswith('.')
            ]

            gestures_seed = [
                (folder, folder.replace('_', ' ').title(), 0.45)
                for folder in valid_folders
            ]

            cursor.executemany("""
                INSERT OR IGNORE INTO gesture_mappings (raw_label, display_name, min_score)
                VALUES (?, ?, ?);
            """, gestures_seed)

        # 4. Seed active non-manual facial expression markers
        expressions_seed = [
            ('mouthSmileLeft', 'Happy', 0.12),
            ('mouthSmileRight', 'Happy', 0.12),
            ('browInnerUp', 'Surprised', 0.035),
            ('browLowerer', 'Angry', 0.025),
            ('browSquint', 'Confused / Question', 0.020),
            ('mouthFrownLeft', 'Sad / Concerned', 0.030),
            ('mouthFrownRight', 'Sad / Concerned', 0.030),
            ('noseSneerLeft', 'Disgust', 0.025),
            ('noseSneerRight', 'Disgust', 0.025),
            ('jawOpen', 'Shocked / Gasping', 0.045),
            ('eyeWideLeft', 'Fearful / Startled', 0.028),
            ('eyeWideRight', 'Fearful / Startled', 0.028),
            ('mouthPucker', 'Skeptical / Doubtful', 0.065),
            ('eyeSquintLeft', 'Suspicious / Wincing', 0.012),
            ('eyeSquintRight', 'Suspicious / Wincing', 0.012),
            ('faceNeutral', 'Neutral', 0.000)
        ]
        cursor.executemany("""
            INSERT OR REPLACE INTO expression_thresholds (blendshape_name, display_name, activation_threshold)
            VALUES (?, ?, ?);
        """, expressions_seed)

        conn.commit()
    finally:
        conn.close()

    print("Database & dataset structures initialized successfully.")
    print(f"Place your ASL videos in subdirectories here: {DATASET_DIR}")


if __name__ == "__main__":
    init_system()