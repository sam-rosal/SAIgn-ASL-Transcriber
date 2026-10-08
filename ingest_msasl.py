import json
import sqlite3

# Define database name
DB_NAME = 'saign_vision.db'

# Define input split files
json_files = {
    'train': 'MSASL_train.json',
    'val': 'MSASL_val.json',
    'test': 'MSASL_test.json'
}

def ingest_data():
    """Reads JSON files and populates the SQLite database table."""
    conn = sqlite3.connect(DB_NAME)
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

    # Ingest records split by split
    for split_name, file_path in json_files.items():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            records_to_insert = []
            for entry in data:
                box = entry.get('box', [None, None, None, None])
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
                    box[0], box[1], box[2], box[3]
                ))

            cursor.executemany('''
                INSERT INTO video_clips (
                    clean_text, label, signer_id, signer, start_time, end_time, 
                    start_frame, end_frame, url, file_title, split, 
                    box_ymin, box_xmin, box_ymax, box_xmax
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', records_to_insert)

            print(f"Successfully inserted {len(records_to_insert)} items into split: {split_name}")

        except FileNotFoundError:
            print(f"Warning: {file_path} not found. Skipping.")

    conn.commit()
    conn.close()

def print_split_percentages():
    """Queries the database and displays the distribution of train/val/test splits."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    query = '''
        SELECT 
            split, 
            COUNT(*) AS clip_count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM video_clips), 2) AS percentage
        FROM video_clips
        GROUP BY split;
    '''

    try:
        results = cursor.execute(query).fetchall()
        print("\n--- Database Split Distribution ---")
        for split, count, pct in results:
            print(f"Split: {split:<6} | Clips: {count:<6} | Share: {pct}%")
    except sqlite3.OperationalError as e:
        print(f"Error reading database: {e}")
    finally:
        conn.close()

if __name__ == '__main__':
    ingest_data()
    print_split_percentages()