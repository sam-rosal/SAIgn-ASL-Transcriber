import os
import json
import subprocess
from collections import defaultdict

# --- System & Directory Configuration ---
# Get the absolute directory path where this script resides
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the target directory where raw MP4 videos will be saved, organized by class folder
MEDIA_DATASET_DIR = os.path.join(SCRIPT_DIR, 'media_dataset')

# Define filepaths for the official MS-ASL metadata partition files
JSON_FILES = [
    os.path.join(SCRIPT_DIR, 'MSASL_train.json'),
    os.path.join(SCRIPT_DIR, 'MSASL_val.json'),
    os.path.join(SCRIPT_DIR, 'MSASL_test.json')
]

# Set the minimum video sample threshold required per sign/word for training suitability
MIN_VIDEOS_PER_CLASS = 3


def load_and_filter_metadata(min_samples=3):
    """
    Parses MS-ASL metadata JSON files, groups video clip records by sign label,
    attaches dataset partition tags ('train', 'val', 'test'), and filters out 
    any class that does not meet the minimum sample threshold before downloading.
    """
    # Dictionary mapping each unique clean sign label to its list of video metadata entries
    class_to_entries = defaultdict(list)
    total_entries = 0

    # 1. Iterate through each JSON split file and aggregate records
    for json_path in JSON_FILES:
        if not os.path.exists(json_path):
            print(f"[WARN] Partition metadata file not found: {json_path}. Skipping...")
            continue
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data:
                # Tag each record with its originating dataset partition for downstream database logging
                if 'train' in json_path:
                    entry['split'] = 'train'
                elif 'val' in json_path:
                    entry['split'] = 'val'
                elif 'test' in json_path:
                    entry['split'] = 'test'
                else:
                    entry['split'] = 'train'

                # Clean label string: lowercase, trim whitespace, and replace spaces with underscores
                label = entry.get('clean_text', entry.get('text')).strip().lower().replace(" ", "_")
                
                # Append entry to its corresponding label bucket
                class_to_entries[label].append(entry)
                total_entries += 1

    print(f"Total video metadata entries parsed across JSON splits: {total_entries}")

    # 2. Pre-download Filter: Retain only classes containing at least `min_samples` videos
    valid_dataset = {}
    discarded_classes = 0

    for label, entries in class_to_entries.items():
        if len(entries) >= min_samples:
            valid_dataset[label] = entries
        else:
            discarded_classes += 1

    print(f"\n--- Pre-Download Filtering ---")
    print(f"Classes meeting >= {min_samples} video requirement: {len(valid_dataset)}")
    print(f"Classes discarded (< {min_samples} video entries in metadata): {discarded_classes}")

    return valid_dataset


def download_clip(url, start_time, end_time, output_path):
    """
    Invokes yt-dlp to fetch the raw YouTube video stream and pipes it through ffmpeg 
    to extract only the specific start/end timestamp segment containing the sign gesture.
    """
    # Skip download if the video clip already exists on disk
    if os.path.exists(output_path):
        return True

    # Construct command-line execution string for yt-dlp + ffmpeg trimming
    cmd = [
        "yt-dlp",
        "--quiet",                                           # Suppress standard terminal output
        "--no-warnings",                                      # Suppress minor warning prints
        "--format", "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]", # Force MP4 container output
        "--external-downloader", "ffmpeg",                   # Delegate stream trimming to ffmpeg
        "--external-downloader-args", f"ffmpeg_i:-ss {start_time} -to {end_time}", # Clip exact start and end seconds
        "-o", output_path,                                   # Set target output filepath
        url                                                  # YouTube target URL
    ]

    try:
        # Execute subprocess with a 60-second timeout per video clip to prevent hanging on slow downloads
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        
        # Verify successful exit code and confirm output file creation
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception:
        # Catch network errors, timeouts, or invalid URL failures gracefully
        return False


def fetch_msasl_dataset():
    """
    Main execution pipeline: Sets up dataset subdirectories, executes downloads for 
    qualified classes, and runs a post-download audit to report dead/unavailable links.
    """
    # Ensure root destination folder exists
    os.makedirs(MEDIA_DATASET_DIR, exist_ok=True)
    
    # Load metadata filtered by the minimum required sample count
    valid_dataset = load_and_filter_metadata(min_samples=MIN_VIDEOS_PER_CLASS)

    if not valid_dataset:
        print("No valid classes found. Ensure MSASL_*.json files exist in the script directory.")
        return

    print("\nStarting video download process...")
    
    # Dictionary tracking successful file downloads per sign class
    download_stats = defaultdict(int)

    # Process each qualified sign class
    for label, entries in valid_dataset.items():
        # Create a dedicated subdirectory for the gesture label (e.g., media_dataset/hello/)
        label_dir = os.path.join(MEDIA_DATASET_DIR, label)
        os.makedirs(label_dir, exist_ok=True)

        for idx, entry in enumerate(entries):
            url = entry['url']
            start_time = entry.get('start_time', 0)
            end_time = entry.get('end_time', 0)
            
            # Format output filename: {label}_{index}_{partition}.mp4 (e.g., hello_0_train.mp4)
            file_name = f"{label}_{idx}_{entry['split']}.mp4"
            output_path = os.path.join(label_dir, file_name)

            # Attempt stream extraction and clipping
            success = download_clip(url, start_time, end_time, output_path)
            if success:
                download_stats[label] += 1

    # --- Post-Download Audit ---
    # Accounts for YouTube video deletions, private status, or regional restrictions
    print("\n--- Post-Download Verification ---")
    insufficient_classes = []

    for label, count in download_stats.items():
        if count < MIN_VIDEOS_PER_CLASS:
            insufficient_classes.append((label, count))

    if insufficient_classes:
        print(f"\n[WARN] {len(insufficient_classes)} classes fell below {MIN_VIDEOS_PER_CLASS} videos due to dead/deleted links:")
        for label, count in insufficient_classes:
            print(f"  - '{label}': Only {count} video(s) downloaded successfully")
        print(f"\nRecommendation: Filter database indexing to ignore classes with < {MIN_VIDEOS_PER_CLASS} downloaded samples.")
    else:
        print(f"All target classes successfully downloaded with at least {MIN_VIDEOS_PER_CLASS} playable videos!")


if __name__ == "__main__":
    fetch_msasl_dataset()