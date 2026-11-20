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


def to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_box(raw_box):
    """Safely extracts 4 float bounding-box coordinates even if 'box' is null or malformed."""
    if isinstance(raw_box, (list, tuple)):
        padded = [to_float(v) for v in raw_box[:4]] + [None] * max(0, 4 - len(raw_box))
        return padded[0], padded[1], padded[2], padded[3]
    return None, None, None, None


def ingest_data():
    """Reads JSON files and populates the SQLite database table."""
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
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

        # Only run one-time cleanup migration if the unique index does not exist yet
        has_unique_idx = cursor.execute("""
            SELECT 1 FROM sqlite_master
            WHERE type = 'index' AND name = 'idx_video_clips_unique'
        """).fetchone() is not None

        if not has_unique_idx:
            cursor.execute('''
                UPDATE video_clips
                SET url = TRIM(url)
                WHERE url IS NOT NULL AND url != TRIM(url)
            ''')
            cursor.execute('''
                DELETE FROM video_clips
                WHERE id NOT IN (
                    SELECT MIN(id)
                    FROM video_clips
                    GROUP BY
                        COALESCE(clean_text, ''),
                        COALESCE(TRIM(url), ''),
                        COALESCE(start_time, -1.0),
                        COALESCE(end_time, -1.0),
                        COALESCE(split, '')
                )
            ''')

        # COALESCE ensures NULL fields are still deduplicated properly in SQLite
        cursor.execute('''
            CREATE UNIQUE INDEX IF NOT EXISTS idx_video_clips_unique
            ON video_clips (
                COALESCE(clean_text, ''),
                COALESCE(url, ''),
                COALESCE(start_time, -1.0),
                COALESCE(end_time, -1.0),
                COALESCE(split, '')
            )
        ''')

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

                raw_url = entry.get('url')
                clean_url = raw_url.strip() if isinstance(raw_url, str) else raw_url

                raw_text = entry.get('clean_text') or entry.get('text')
                clean_text = raw_text.strip() if isinstance(raw_text, str) else raw_text

                b_ymin, b_xmin, b_ymax, b_xmax = normalize_box(entry.get('box'))
                records_to_insert.append((
                    clean_text,
                    to_int(entry.get('label')),
                    to_int(entry.get('signer_id')),
                    to_int(entry.get('signer')),
                    to_float(entry.get('start_time')),
                    to_float(entry.get('end_time')),
                    to_int(entry.get('start')),
                    to_int(entry.get('end')),
                    clean_url,
                    entry.get('file'),
                    split_name,
                    b_ymin, b_xmin, b_ymax, b_xmax
                ))

            changes_before = conn.total_changes
            cursor.executemany('''
                INSERT OR IGNORE INTO video_clips (
                    clean_text, label, signer_id, signer, start_time, end_time,
                    start_frame, end_frame, url, file_title, split,
                    box_ymin, box_xmin, box_ymax, box_xmax
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', records_to_insert)
            inserted_count = conn.total_changes - changes_before

            print(f"Processed {len(records_to_insert)} items for split: {split_name} (new rows inserted: {inserted_count})")

        conn.commit()
    finally:
        conn.close()


def print_split_percentages():
    """Queries the database and displays the distribution of train/val/test splits."""
    if not os.path.exists(DB_PATH):
        print("Database file not found. Nothing to summarize.")
        return

    conn = sqlite3.connect(DB_PATH, timeout=30.0)
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