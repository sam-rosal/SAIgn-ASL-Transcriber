import sqlite3
import os
import sys

# Defining the database path using the file name
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, 'saign_vision.db')

# Defining the structured folders for organizing raw media dataset and extracted landmarks
DATASET_DIR = os.path.join(SCRIPT_DIR, 'media_dataset')
FEATURES_DIR = os.path.join(SCRIPT_DIR, 'extracted_features')

# Bump this whenever the seed values/units below change. A bump overwrites expression_thresholds ONCE;
# on every other run existing rows (including your hand-tuned values) are preserved.
# Must match EXPRESSION_SEED_VERSION in P4_run_system.py.
EXPRESSION_SEED_VERSION = 2

# Thresholds are RATIOS of face size (outer-eye-corner distance), measured in pixels -- see
# detect_expressions() in P4_run_system.py for the exact metric behind each name.
# NOTE: these are starting values derived from typical adult face proportions, NOT measured on your
# webcam. Tune them directly in the expression_thresholds table; re-running setup won't overwrite them.
EXPRESSIONS_SEED = [
    ('mouthSmileLeft', 'Happy', 0.04),              # corner lift above upper-lip centre  (> thr)
    ('mouthSmileRight', 'Happy', 0.04),
    ('browInnerUp', 'Surprised', 0.27),             # inner brow to nose bridge           (> thr)
    ('browLowerer', 'Angry', 0.22),                 # brow-to-eyelid gap                  (< thr)
    ('browSquint', 'Confused / Question', 0.19),    # brow-to-eyelid gap, stricter        (< thr)
    ('mouthFrownLeft', 'Sad / Concerned', 0.06),    # corner drop below upper-lip centre  (> thr)
    ('mouthFrownRight', 'Sad / Concerned', 0.06),
    ('noseSneerLeft', 'Disgust', 0.31),             # nose bridge to upper lip            (< thr)
    ('noseSneerRight', 'Disgust', 0.31),
    ('jawOpen', 'Shocked / Gasping', 0.20),         # inner-lip gap                       (> thr)
    ('eyeWideLeft', 'Fearful / Startled', 0.15),    # eyelid opening                      (> thr)
    ('eyeWideRight', 'Fearful / Startled', 0.15),
    ('mouthPucker', 'Skeptical / Doubtful', 0.42),  # mouth width                         (< thr)
    ('eyeSquintLeft', 'Suspicious / Wincing', 0.05),   # eyelid opening                   (< thr)
    ('eyeSquintRight', 'Suspicious / Wincing', 0.05),
    ('faceNeutral', 'Neutral', 0.000),
]


def seed_expression_thresholds(cursor, force=False):
    """Seeds expression thresholds without clobbering user edits (see EXPRESSION_SEED_VERSION)."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS app_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );""")
    row = cursor.execute("SELECT value FROM app_meta WHERE key = 'expression_seed_version';").fetchone()
    stored = int(row[0]) if row and str(row[0]).strip().isdigit() else 0

    if force or stored < EXPRESSION_SEED_VERSION:
        cursor.executemany("""
            INSERT OR REPLACE INTO expression_thresholds (blendshape_name, display_name, activation_threshold)
            VALUES (?, ?, ?);
        """, EXPRESSIONS_SEED)
        cursor.execute(
            "INSERT OR REPLACE INTO app_meta (key, value) VALUES ('expression_seed_version', ?);",
            (str(EXPRESSION_SEED_VERSION),)
        )
        print(f"Expression thresholds installed (seed v{EXPRESSION_SEED_VERSION}).")
    else:
        cursor.executemany("""
            INSERT OR IGNORE INTO expression_thresholds (blendshape_name, display_name, activation_threshold)
            VALUES (?, ?, ?);
        """, EXPRESSIONS_SEED)
        print("Expression thresholds already up to date; existing values preserved "
              "(use --reset-expressions to restore defaults).")


def ensure_training_dataset_columns(cursor):
    """Migrates older training_dataset tables so required columns and unique indexes exist."""
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

    # Deduplicate legacy rows before enforcing UNIQUE index on video_path
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


def init_system():
    # 1. Create Directories
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(FEATURES_DIR, exist_ok=True)

    # 2. Setup SQLite Tables
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
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

        # 4. Seed active non-manual facial expression markers (version-aware, preserves user edits)
        seed_expression_thresholds(cursor, force='--reset-expressions' in sys.argv)

        conn.commit()
    finally:
        conn.close()

    print("Database & dataset structures initialized successfully.")
    print(f"Place your ASL videos in subdirectories here: {DATASET_DIR}")


if __name__ == "__main__":
    init_system()