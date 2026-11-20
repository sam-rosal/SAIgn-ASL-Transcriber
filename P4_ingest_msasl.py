import json
import os
import sqlite3

# Define paths anchored to the script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, 'saign_vision.db')

# Define input split files
json_files = {
    'train': os.path.join(SCRIPT_DIR, 'MSASL_train.json'),
    'val': os.path.join(SCRIPT_DIR, 'MSASL_val.json'),
    'test': os.path.join(SCRIPT_DIR, 'MSASL_test.json')
}


def normalize_box(raw_box):
    """Safely extracts 4 bounding-box coordinates even if 'box' is null or malformed."""
    if isinstance(raw_box, (list, tuple)):
        padded = list(raw_box[:4]) + [None] * max(0, 4 - len(raw_box))
        return padded[0], padded[1], padded[2], padded[3]
    return None, None, None, None


def ingest_data():
    """Reads JSON files and populates the SQLite database table."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()

        # Create target table if it doesn't exist
        cursor.execute('''
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
            )
        ''')

        # Prevent duplicate clip rows if the ingestion script is executed multiple times
        try:
            cursor.execute('''
                CREATE UNIQUE INDEX IF NOT EXISTS idx_video_clips_unique
                ON video_clips (clean_text, url, start_time, end_time, split)
            ''')
        except sqlite3.OperationalError:
            pass

        # Ingest records split by split
        for split_name, file_path in json_files.items():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except FileNotFoundError:
                print(f"Warning: {file_path} not found. Skipping.")
                continue
            except (json.JSONDecodeError, OSError) as e:
                print(f"Warning: Failed to read {file_path} ({e}). Skipping.")
                continue

            if not isinstance(data, list):
                print(f"Warning: Expected a JSON list in {file_path}. Skipping.")
                continue

            records_to_insert = []
            for entry in data:
                if not isinstance(entry, dict):
                    continue

                b_ymin, b_xmin, b_ymax, b_xmax = normalize_box(entry.get('box'))
                records_to_insert.append((
                    entry.get('clean_text'),
                    entry.get('label'),
                    entry.get('signer_id'),
                    entry.get('signer'),
                    entry.get('start_time'),
                    entry.get('end_time'),
                    entry.get('start'),
                    entry.get('end'),
                    entry.get('url'),
                    entry.get('file'),
                    split_name,  # Tag split based on source file
                    b_ymin, b_xmin, b_ymax, b_xmax
                ))

            cursor.executemany('''
                INSERT OR IGNORE INTO video_clips (
                    clean_text, label, signer_id, signer, start_time, end_time,
                    start_frame, end_frame, url, file_title, split,
                    box_ymin, box_xmin, box_ymax, box_xmax
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', records_to_insert)

            print(f"Processed {len(records_to_insert)} items for split: {split_name} (new rows inserted: {cursor.rowcount})")

        conn.commit()
    finally:
        conn.close()


def print_split_percentages():
    """Queries the database and displays the distribution of train/val/test splits."""
    if not os.path.exists(DB_PATH):
        print("Database file not found. Nothing to summarize.")
        return

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        query = '''
            SELECT
                split,
                COUNT(*) AS clip_count,
                ROUND(COUNT(*) * 100.0 / NULLIF((SELECT COUNT(*) FROM video_clips), 0), 2) AS percentage
            FROM video_clips
            GROUP BY split;
        '''
        results = cursor.execute(query).fetchall()
        print("\n--- Database Split Distribution ---")
        for split, count, pct in results:
            pct_display = f"{pct}%" if pct is not None else "0.0%"
            print(f"Split: {split:<6} | Clips: {count:<6} | Share: {pct_display}")
    except sqlite3.OperationalError as e:
        print(f"Error reading database: {e}")
    finally:
        conn.close()


if __name__ == '__main__':
    ingest_data()
    print_split_percentages()