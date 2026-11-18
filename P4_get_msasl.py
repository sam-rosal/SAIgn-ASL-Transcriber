import os
import json
import re
import sqlite3
import subprocess
from collections import defaultdict

# ==============================================================================
# SYSTEM & DIRECTORY CONFIGURATION
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, 'saign_vision.db')
MEDIA_DATASET_DIR = os.path.join(SCRIPT_DIR, 'media_dataset')
MANIFEST_PATH = os.path.join(MEDIA_DATASET_DIR, '.msasl_download_manifest.json')
PRIORITY_FILE_PATH = os.path.join(SCRIPT_DIR, 'priority_signs.txt')

JSON_FILES = [
    os.path.join(SCRIPT_DIR, 'MSASL_train.json'),
    os.path.join(SCRIPT_DIR, 'MSASL_val.json'),
    os.path.join(SCRIPT_DIR, 'MSASL_test.json')
]

MIN_VIDEOS_PER_CLASS = 3
BATCH_SIZE_CLASSES = 10  # Process 10 sign classes per execution run
VALID_VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv')

# Set to True to download ONLY signs listed in priority_signs.txt and ignore the rest of MS-ASL
ONLY_DOWNLOAD_PRIORITY = False

# Used strictly to auto-create priority_signs.txt on first run if the file does not exist
DEFAULT_PRIORITY_VOCAB = {
    "1. Greetings & Courtesies": [
        "hello", "goodbye", "please", "thank_you", "thanks",
        "youre_welcome", "welcome", "sorry", "excuse_me", "nice_to_meet_you"
    ],
    "2. Pronouns & People": [
        "i", "me", "you", "he", "she", "it", "we", "us", "they", "them",
        "friend", "family", "name", "person", "man", "woman"
    ],
    "3. Question Words": [
        "who", "what", "where", "when", "why", "how", "which"
    ],
    "4. Everyday Verbs & Actions": [
        "want", "need", "help", "like", "know", "understand", "think",
        "go", "come", "work", "learn", "stop", "wait", "finish", "done",
        "make", "see", "look", "have", "give", "take"
    ],
    "5. Responses & Descriptors": [
        "yes", "no", "good", "bad", "happy", "sad", "fine", "more", "same"
    ],
    "6. Time & Connectors": [
        "today", "now", "tomorrow", "yesterday", "time"
    ],
}


def write_default_priority_file(file_path):
    """Formats DEFAULT_PRIORITY_VOCAB into a clean, commented priority_signs.txt file."""
    lines = [
        "# SAIgn Conversational Priority Vocabulary",
        "# Signs listed higher are downloaded first. Lines starting with '#' are ignored.\n"
    ]
    for category, signs in DEFAULT_PRIORITY_VOCAB.items():
        lines.append(f"# {category}")
        lines.extend(signs)
        lines.append("")

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines).strip() + "\n")


def load_priority_map(file_path=PRIORITY_FILE_PATH):
    """
    Loads priority signs from an external text file into an O(1) rank lookup dictionary.
    Auto-generates priority_signs.txt on first run if missing.
    """
    if not os.path.exists(file_path):
        write_default_priority_file(file_path)
        print(f"[INFO] Created default priority list at: {file_path}")

    priority_map = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            # Strip inline comments and whitespace
            clean_line = raw_line.split('#', 1)[0].strip()
            if not clean_line:
                continue

            normalized = clean_line.lower().replace(" ", "_")
            if normalized not in priority_map:
                priority_map[normalized] = len(priority_map)

    return priority_map


def get_youtube_id(url):
    """
    Extracts the unique 11-character YouTube video ID from a URL string.
    e.g., 'https://www.youtube.com/watch?v=dQw4w9WgXcQ' -> 'dQw4w9WgXcQ'
    """
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None


def get_clip_key(entry, idx):
    """Generates a unique identifier for a metadata entry to avoid duplicate downloads."""
    url = entry.get('url', '')
    start_time = entry.get('start_time', 0)
    end_time = entry.get('end_time', 0)

    yt_id = get_youtube_id(url)
    if yt_id:
        start_sec = int(float(start_time))
        end_sec = int(float(end_time))
        return f"{yt_id}_{start_sec}_{end_sec}"
    return f"msasl_{idx}_{entry.get('split', 'train')}"


def load_download_manifest():
    """Loads the record of already-downloaded clip keys mapped to their local filenames."""
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_download_manifest(manifest):
    """Persists the download manifest to disk."""
    os.makedirs(MEDIA_DATASET_DIR, exist_ok=True)
    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)


def load_db_split_tags():
    """
    Loads (clean_text, url, start_time, end_time) -> split tag mappings from
    saign_vision.db (populated by P4_ingest_msasl_4.py).
    """
    tag_lookup = {}
    if not os.path.exists(DB_PATH):
        return tag_lookup

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT clean_text, url, start_time, end_time, split
            FROM video_clips
            WHERE split IS NOT NULL
        """)
        for clean_text, url, start_time, end_time, split in cursor.fetchall():
            norm_text = (clean_text or "").strip().lower().replace(" ", "_")
            key = (norm_text, url, round(float(start_time or 0), 3), round(float(end_time or 0), 3))
            tag_lookup[key] = split.strip().lower()
        conn.close()
    except sqlite3.Error:
        pass

    return tag_lookup


def get_existing_folders():
    """Scans the local media_dataset/ directory for existing gesture subdirectories."""
    folder_map = {}
    if os.path.exists(MEDIA_DATASET_DIR):
        for folder in os.listdir(MEDIA_DATASET_DIR):
            folder_path = os.path.join(MEDIA_DATASET_DIR, folder)
            if os.path.isdir(folder_path) and not folder.startswith('.'):
                normalized_key = folder.lower().replace(" ", "_")
                folder_map[normalized_key] = folder
    return folder_map


def count_local_videos(folder_path):
    """Counts total playable video files currently stored in a given local directory."""
    if not os.path.exists(folder_path):
        return 0
    return len([f for f in os.listdir(folder_path) if f.lower().endswith(VALID_VIDEO_EXTS)])


def get_next_video_number(folder_path, sign_name):
    """
    Determines the next sequential video number (1, 2, 3, ...) for a sign folder,
    accounting for any existing videos already on disk.
    """
    if not os.path.exists(folder_path):
        return 1

    pattern = re.compile(rf"^{re.escape(sign_name)}_(\d+)_[^.]+", re.IGNORECASE)
    max_num = 0
    total_videos = 0

    for fname in os.listdir(folder_path):
        if fname.lower().endswith(VALID_VIDEO_EXTS):
            total_videos += 1
            match = pattern.match(os.path.splitext(fname)[0])
            if match:
                max_num = max(max_num, int(match.group(1)))

    return max(max_num, total_videos) + 1


def is_clip_downloaded(folder_name, folder_path, clip_key, manifest):
    """Checks if a clip is already recorded in the manifest and present on disk."""
    saved_filename = manifest.get(folder_name, {}).get(clip_key)
    if saved_filename and os.path.exists(os.path.join(folder_path, saved_filename)):
        return True
    return False


def load_and_filter_metadata(priority_map, min_samples=3):
    """Parses MS-ASL JSON files, resolves split tags, and filters classes by min sample count."""
    existing_folders = get_existing_folders()
    db_tags = load_db_split_tags()
    manifest = load_download_manifest()
    class_to_entries = defaultdict(list)
    total_entries = 0

    for json_path in JSON_FILES:
        if not os.path.exists(json_path):
            print(f"[WARN] Partition metadata file not found: {json_path}. Skipping...")
            continue

        if 'train' in os.path.basename(json_path).lower():
            file_split = 'train'
        elif 'val' in os.path.basename(json_path).lower():
            file_split = 'val'
        elif 'test' in os.path.basename(json_path).lower():
            file_split = 'test'
        else:
            file_split = 'train'

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data:
                raw_text = entry.get('clean_text', entry.get('text', '')).strip().lower().replace(" ", "_")
                if not raw_text:
                    continue

                if ONLY_DOWNLOAD_PRIORITY and raw_text not in priority_map:
                    continue

                # Prefer the split tag ingested in P4_ingest_msasl_4, fallback to source JSON split
                db_key = (
                    raw_text,
                    entry.get('url', ''),
                    round(float(entry.get('start_time', 0) or 0), 3),
                    round(float(entry.get('end_time', 0) or 0), 3)
                )
                entry['split'] = db_tags.get(db_key, file_split)

                target_folder_name = existing_folders.get(raw_text, raw_text)
                entry['target_folder'] = target_folder_name

                class_to_entries[target_folder_name].append(entry)
                total_entries += 1

    print(f"Total video metadata entries parsed across JSON splits: {total_entries}")

    valid_dataset = {}
    discarded_classes = 0

    for folder_name, entries in class_to_entries.items():
        folder_path = os.path.join(MEDIA_DATASET_DIR, folder_name)
        existing_count = count_local_videos(folder_path)

        pending_count = 0
        for idx, entry in enumerate(entries):
            clip_key = get_clip_key(entry, idx)
            if not is_clip_downloaded(folder_name, folder_path, clip_key, manifest):
                pending_count += 1

        total_potential = existing_count + pending_count

        if total_potential >= min_samples:
            valid_dataset[folder_name] = entries
        else:
            discarded_classes += 1

    priority_matches = sum(1 for k in valid_dataset if k.lower().replace(" ", "_") in priority_map)
    print(f"\n--- Pre-Download Filtering ---")
    print(f"Classes meeting >= {min_samples} total possible video requirement: {len(valid_dataset)} ({priority_matches} priority signs matched)")
    print(f"Classes discarded: {discarded_classes}")

    return valid_dataset


def download_clip(url, start_time, end_time, output_path):
    """Streams and trims YouTube clip with yt-dlp + ffmpeg."""
    if os.path.exists(output_path):
        return True

    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--format", "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "--external-downloader", "ffmpeg",
        "--external-downloader-args", f"ffmpeg_i:-ss {start_time} -to {end_time}",
        "-o", output_path,
        url
    ]

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception:
        return False


def fetch_msasl_dataset():
    """Main execution loop: reads priority_signs.txt, downloads priority classes first, then general classes."""
    os.makedirs(MEDIA_DATASET_DIR, exist_ok=True)
    priority_map = load_priority_map(PRIORITY_FILE_PATH)
    print(f"Loaded {len(priority_map)} priority signs from {os.path.basename(PRIORITY_FILE_PATH)}.")

    valid_dataset = load_and_filter_metadata(priority_map, min_samples=MIN_VIDEOS_PER_CLASS)

    if not valid_dataset:
        print("No valid classes found.")
        return

    manifest = load_download_manifest()
    fallback_rank = len(priority_map)

    # Identify classes that still have pending downloads
    classes_with_pending = []
    for folder_name, entries in valid_dataset.items():
        label_dir = os.path.join(MEDIA_DATASET_DIR, folder_name)
        existing_count = count_local_videos(label_dir)
        norm_name = folder_name.lower().replace(" ", "_")
        is_priority = norm_name in priority_map

        pending = []
        for idx, entry in enumerate(entries):
            clip_key = get_clip_key(entry, idx)
            if not is_clip_downloaded(folder_name, label_dir, clip_key, manifest):
                pending.append((entry, clip_key))

        if pending:
            classes_with_pending.append({
                'folder_name': folder_name,
                'existing_count': existing_count,
                'is_priority': is_priority,
                'priority_rank': priority_map.get(norm_name, fallback_rank),
                'pending': pending
            })

    if not classes_with_pending:
        print("\nAll eligible sign classes are already fully downloaded!")
        return

    # Sort order:
    # 1. Priority signs strictly before general signs
    # 2. Classes with < MIN_VIDEOS_PER_CLASS existing clips first
    # 3. Exact line order in priority_signs.txt (alphabetical fallback for general signs)
    classes_with_pending.sort(
        key=lambda x: (
            not x['is_priority'],
            x['existing_count'] >= MIN_VIDEOS_PER_CLASS,
            x['priority_rank'],
            x['folder_name']
        )
    )

    batch_classes = classes_with_pending[:BATCH_SIZE_CLASSES]
    pending_priority_total = sum(1 for x in classes_with_pending if x['is_priority'])

    print(f"\n--- Starting Download Batch ---")
    print(f"Total sign classes with pending downloads: {len(classes_with_pending)} ({pending_priority_total} priority signs remaining)")
    print(f"Processing next {len(batch_classes)} sign classes in this batch...\n")

    for c_idx, item in enumerate(batch_classes, 1):
        folder_name = item['folder_name']
        label_dir = os.path.join(MEDIA_DATASET_DIR, folder_name)
        os.makedirs(label_dir, exist_ok=True)
        manifest.setdefault(folder_name, {})

        next_num = get_next_video_number(label_dir, folder_name)
        tier_tag = "[PRIORITY]" if item['is_priority'] else "[GENERAL]"

        print(f"[{c_idx}/{len(batch_classes)}] {tier_tag} Sign Class: '{folder_name}' (Existing on disk: {item['existing_count']}, Pending in batch: {len(item['pending'])})")

        success_count = 0
        for entry, clip_key in item['pending']:
            url = entry['url']
            start_time = entry.get('start_time', 0)
            end_time = entry.get('end_time', 0)
            tag = entry.get('split', 'train')

            # Format: <sign>_<number>_<tag>.mp4 (e.g., hello_1_train.mp4)
            file_name = f"{folder_name}_{next_num}_{tag}.mp4"
            output_path = os.path.join(label_dir, file_name)

            success = download_clip(url, start_time, end_time, output_path)
            if success:
                manifest[folder_name][clip_key] = file_name
                success_count += 1
                next_num += 1

        save_download_manifest(manifest)
        total_now = count_local_videos(label_dir)
        print(f"   -> Successfully downloaded {success_count}/{len(item['pending'])} new clips. Current total in folder: {total_now}\n")

    remaining_classes = len(classes_with_pending) - len(batch_classes)
    remaining_priority = sum(1 for x in classes_with_pending[BATCH_SIZE_CLASSES:] if x['is_priority'])
    print("--- Batch Execution Summary ---")
    if remaining_classes > 0:
        print(f"Batch completed! Remaining classes: {remaining_classes} ({remaining_priority} priority signs left).")
        print(f"Run 'python {os.path.basename(__file__)}' again to process the next batch of 10 sign classes.")
    else:
        print("All pending sign classes across the dataset have been processed!")


if __name__ == "__main__":
    fetch_msasl_dataset()