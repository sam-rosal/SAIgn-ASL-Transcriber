import os
import json
import re
import subprocess
from collections import defaultdict

# ==============================================================================
# SYSTEM & DIRECTORY CONFIGURATION
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_DATASET_DIR = os.path.join(SCRIPT_DIR, 'media_dataset')

JSON_FILES = [
    os.path.join(SCRIPT_DIR, 'MSASL_train.json'),
    os.path.join(SCRIPT_DIR, 'MSASL_val.json'),
    os.path.join(SCRIPT_DIR, 'MSASL_test.json')
]

MIN_VIDEOS_PER_CLASS = 3
BATCH_SIZE_CLASSES = 10  # Process 10 sign classes per execution run


def get_youtube_id(url):
    """
    Extracts the unique 11-character YouTube video ID from a URL string.
    e.g., 'https://www.youtube.com/watch?v=dQw4w9WgXcQ' -> 'dQw4w9WgXcQ'
    """
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None


def get_clip_filename(entry, idx):
    """Generates deterministic filename for a video clip."""
    url = entry.get('url', '')
    start_time = entry.get('start_time', 0)
    end_time = entry.get('end_time', 0)
    
    yt_id = get_youtube_id(url)
    if yt_id:
        start_sec = int(float(start_time))
        end_sec = int(float(end_time))
        return f"{yt_id}_{start_sec}_{end_sec}.mp4"
    else:
        return f"msasl_{idx}_{entry.get('split', 'train')}.mp4"


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
    valid_exts = ('.mp4', '.avi', '.mov', '.mkv')
    return len([f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)])


def load_and_filter_metadata(min_samples=3):
    """Parses MS-ASL JSON files and filters gesture classes by min sample count."""
    existing_folders = get_existing_folders()
    class_to_entries = defaultdict(list)
    total_entries = 0

    for json_path in JSON_FILES:
        if not os.path.exists(json_path):
            print(f"[WARN] Partition metadata file not found: {json_path}. Skipping...")
            continue
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data:
                if 'train' in json_path:
                    entry['split'] = 'train'
                elif 'val' in json_path:
                    entry['split'] = 'val'
                elif 'test' in json_path:
                    entry['split'] = 'test'
                else:
                    entry['split'] = 'train'

                raw_text = entry.get('clean_text', entry.get('text')).strip().lower().replace(" ", "_")
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
        
        # Count how many entries are not yet downloaded on disk
        pending_count = 0
        for idx, entry in enumerate(entries):
            fname = get_clip_filename(entry, idx)
            if not os.path.exists(os.path.join(folder_path, fname)):
                pending_count += 1

        total_potential = existing_count + pending_count

        if total_potential >= min_samples:
            valid_dataset[folder_name] = entries
        else:
            discarded_classes += 1

    print(f"\n--- Pre-Download Filtering ---")
    print(f"Classes meeting >= {min_samples} total possible video requirement: {len(valid_dataset)}")
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
    """Main execution loop: downloads clips in batches of 10 sign classes at a time."""
    os.makedirs(MEDIA_DATASET_DIR, exist_ok=True)
    valid_dataset = load_and_filter_metadata(min_samples=MIN_VIDEOS_PER_CLASS)

    if not valid_dataset:
        print("No valid classes found.")
        return

    # Identify classes that still have pending downloads
    classes_with_pending = []
    for folder_name, entries in valid_dataset.items():
        label_dir = os.path.join(MEDIA_DATASET_DIR, folder_name)
        existing_count = count_local_videos(label_dir)
        
        pending = []
        for idx, entry in enumerate(entries):
            fname = get_clip_filename(entry, idx)
            fpath = os.path.join(label_dir, fname)
            if not os.path.exists(fpath):
                pending.append((entry, fname, fpath))
        
        if pending:
            classes_with_pending.append({
                'folder_name': folder_name,
                'existing_count': existing_count,
                'pending': pending
            })

    if not classes_with_pending:
        print("\nAll eligible sign classes are already fully downloaded!")
        return

    # Prioritize classes with < MIN_VIDEOS_PER_CLASS existing videos first
    classes_with_pending.sort(
        key=lambda x: (x['existing_count'] >= MIN_VIDEOS_PER_CLASS, x['folder_name'])
    )

    batch_classes = classes_with_pending[:BATCH_SIZE_CLASSES]

    print(f"\n--- Starting Download Batch ---")
    print(f"Total sign classes with pending downloads: {len(classes_with_pending)}")
    print(f"Processing next {len(batch_classes)} sign classes in this batch...\n")

    for c_idx, item in enumerate(batch_classes, 1):
        folder_name = item['folder_name']
        label_dir = os.path.join(MEDIA_DATASET_DIR, folder_name)
        os.makedirs(label_dir, exist_ok=True)

        print(f"[{c_idx}/{len(batch_classes)}] Sign Class: '{folder_name}' (Existing on disk: {item['existing_count']}, Pending in batch: {len(item['pending'])})")

        success_count = 0
        for entry, file_name, output_path in item['pending']:
            url = entry['url']
            start_time = entry.get('start_time', 0)
            end_time = entry.get('end_time', 0)

            success = download_clip(url, start_time, end_time, output_path)
            if success:
                success_count += 1

        total_now = count_local_videos(label_dir)
        print(f"   -> Successfully downloaded {success_count}/{len(item['pending'])} new clips. Current total in folder: {total_now}\n")

    remaining_classes = len(classes_with_pending) - len(batch_classes)
    print("--- Batch Execution Summary ---")
    if remaining_classes > 0:
        print(f"Batch completed! There are {remaining_classes} more sign classes with pending downloads.")
        print("Run 'python get_msasl.py' again to process the next batch of 10 sign classes.")
    else:
        print("All pending sign classes across the dataset have been processed!")


if __name__ == "__main__":
    fetch_msasl_dataset()