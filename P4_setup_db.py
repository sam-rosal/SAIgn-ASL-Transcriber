import sqlite3
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, 'saign_vision.db')

# Define structured folders for organizing your raw media dataset and extracted landmarks
DATASET_DIR = os.path.join(SCRIPT_DIR, 'media_dataset')
FEATURES_DIR = os.path.join(SCRIPT_DIR, 'extracted_features')

def init_system():
    # 1. Create Directories
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(FEATURES_DIR, exist_ok=True)
    
    # 2. Setup SQLite Tables
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS gesture_mappings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        raw_label TEXT UNIQUE NOT NULL,
        display_name TEXT NOT NULL,
        min_score REAL DEFAULT 0.4
    );""")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS expression_thresholds (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        blendshape_name TEXT UNIQUE NOT NULL,
        display_name TEXT NOT NULL,
        activation_threshold REAL NOT NULL
    );""")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS training_dataset (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_path TEXT UNIQUE NOT NULL,
        label TEXT NOT NULL,
        num_frames INTEGER,
        extracted_features_path TEXT
    );""")
    
    # Seed default UI lookups and sensitivities
    gestures_seed = [
        ('Victory', 'PEACE', 0.45),
        ('Open_Palm', 'HELLO', 0.40),
        ('Thumbs_Up', 'YES', 0.50),
        ('Thumbs_Down', 'NO', 0.50)
    ]
    cursor.executemany("""
        INSERT OR REPLACE INTO gesture_mappings (raw_label, display_name, min_score)
        VALUES (?, ?, ?);
    """, gestures_seed)
    
    expressions_seed = [
        ('mouthSmileLeft', 'Smile', 0.35),
        ('browInnerUp', 'Surprise', 0.40)
    ]
    cursor.executemany("""
        INSERT OR REPLACE INTO expression_thresholds (blendshape_name, display_name, activation_threshold)
        VALUES (?, ?, ?);
    """, expressions_seed)
    
    conn.commit()
    conn.close()
    print("Database & dataset structures initialized.")
    print(f"Place your ASL videos in subdirectories here: {DATASET_DIR}")

if __name__ == "__main__":
    init_system()