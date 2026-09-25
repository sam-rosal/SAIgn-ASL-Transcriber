import os
import json
import subprocess
from collections import defaultdict

# ==============================================================================
# SYSTEM & DIRECTORY CONFIGURATION
# ==============================================================================
# Dynamically locate the absolute directory path where this script resides
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Target directory where raw MP4 sign videos will be stored, organized by gesture folder
MEDIA_DATASET_DIR = os.path.join(SCRIPT_DIR, 'media_dataset')

# Filepaths for official MS-ASL dataset partition metadata JSON files
JSON_FILES = [
    os.path.join(SCRIPT_DIR, 'MSASL_train.json'),
    os.path.join(SCRIPT_DIR, 'MSASL_val.json'),
    os.path.join(SCRIPT_DIR, 'MSASL_test.json')
]

# Set minimum required video sample threshold per gesture for training viability
MIN_VIDEOS_PER_CLASS = 3


def get_existing_folders():
    """
    Scans the local media_dataset/ directory for existing gesture subdirectories.
    
    Returns:
        dict: Normalized key mapping to actual directory name.
              e.g., {'hello': 'hello', 'thank_you': 'Thank_You'}
    
    This prevents creating duplicate folders with casing mismatches (e.g., 'Hello' vs 'hello').
    """
    folder_map = {}
    if os.path.exists(MEDIA_DATASET_DIR):
        for folder in os.listdir(MEDIA_DATASET_DIR):
            folder_path = os.path.join(MEDIA_DATASET_DIR, folder)
            # Filter out hidden files (like .DS_Store) and non-directory items
            if os.path.isdir(folder_path) and not folder.startswith('.'):
                # Normalize folder name string for lookups (lowercase, spaces to underscores)
                normalized_key = folder.lower().replace(" ", "_")
                folder_map[normalized_key] = folder
    return folder_map


def count_local_videos(folder_path):
    """
    Counts total playable video files currently stored in a given local directory.

    Args:
        folder_path (str): Path to local gesture subdirectory.

    Returns:
        int: Number of existing video files.
    """
    if not os.path.exists(folder_path):
        return 0
    valid_exts = ('.mp4', '.avi', '.mov', '.mkv')
    return len([f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)])


def load_and_filter_metadata(min_samples=3):
    """
    Parses MS-ASL metadata JSON files, groups video records by clean sign label, 
    maps them to existing local subfolders, and filters out classes where 
    (Existing Local Videos + MS-ASL Metadata Entries) < min_samples.

    Args:
        min_samples (int): Required minimum sample count threshold per sign.

    Returns:
        dict: Filtered dataset structure mapping target folder names to video metadata lists.
    """
    # Step 1: Query pre-existing local folders to enable appending to existing datasets
    existing_folders = get_existing_folders()
    class_to_entries = defaultdict(list)
    total_entries = 0

    # Step 2: Iterate through all JSON split partitions and aggregate entries
    for json_path in JSON_FILES:
        if not os.path.exists(json_path):
            print(f"[WARN] Partition metadata file not found: {json_path}. Skipping...")
            continue
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data:
                # Assign dataset partition split tag for downstream database logging
                if 'train' in json_path:
                    entry['split'] = 'train'
                elif 'val' in json_path:
                    entry['split'] = 'val'
                elif 'test' in json_path:
                    entry['split'] = 'test'
                else:
                    entry['split'] = 'train'

                # Format text label cleanly (lowercase, strip whitespace, replace spaces)
                raw_text = entry.get('clean_text', entry.get('text')).strip().lower().replace(" ", "_")
                
                # Check if a local folder already exists for this sign; reuse exact name if found
                target_folder_name = existing_folders.get(raw_text, raw_text)
                entry['target_folder'] = target_folder_name

                # Append record to target folder array
                class_to_entries[target_folder_name].append(entry)
                total_entries += 1

    print(f"Total video metadata entries parsed across JSON splits: {total_entries}")

    # Step 3: Combined Pre-Download Filtering (Local Files + MS-ASL Metadata)
    valid_dataset = {}
    discarded_classes = 0

    for folder_name, entries in class_to_entries.items():
        folder_path = os.path.join(MEDIA_DATASET_DIR, folder_name)
        existing_count = count_local_videos(folder_path)
        combined_count = existing_count + len(entries)

        # Keep class if total combined samples meet threshold
        if combined_count >= min_samples:
            valid_dataset[folder_name] = entries
        else:
            discarded_classes += 1

    print(f"\n--- Pre-Download Filtering ---")
    print(f"Classes meeting >= {min_samples} video requirement (Local + MS-ASL): {len(valid_dataset)}")
    print(f"Classes discarded (< {min_samples} total potential samples): {discarded_classes}")

    return valid_dataset


def download_clip(url, start_time, end_time, output_path):
    """
    Invokes yt-dlp to stream raw YouTube video and pipes it through ffmpeg 
    to extract only the precise timestamp slice containing the sign gesture.

    Args:
        url (str): Target YouTube video URL.
        start_time (float): Gesture start timestamp in seconds.
        end_time (float): Gesture end timestamp in seconds.
        output_path (str): Destination file path for cropped MP4 clip.

    Returns:
        bool: True if video exists or downloaded successfully, False otherwise.
    """
    # Skip download attempt if output file already exists on disk
    if os.path.exists(output_path):
        return True  

    # Construct yt-dlp execution command with external ffmpeg trimming arguments
    cmd = [
        "yt-dlp",
        "--quiet",                                           # Suppress standard terminal outputs
        "--no-warnings",                                      # Suppress minor warning prints
        "--format", "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]", # Force MP4 container
        "--external-downloader", "ffmpeg",                   # Delegate stream clipping to ffmpeg
        "--external-downloader-args", f"ffmpeg_i:-ss {start_time} -to {end_time}", # Trim exact start/end
        "-o", output_path,                                   # Output filepath
        url                                                  # Source URL
    ]

    try:
        # Execute subprocess with a 60-second timeout to prevent stalling on dead links
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception:
        # Gracefully handle network timeouts, private videos, or missing streams
        return False


def fetch_msasl_dataset():
    """
    Main download pipeline: Resolves local folder paths, fetches MS-ASL clips, 
    appends downloads using collision-free file naming (`msasl_{idx}_{split}.mp4`), 
    and executes a post-download file audit.
    """
    os.makedirs(MEDIA_DATASET_DIR, exist_ok=True)
    
    # Load dataset structure filtered by local + MS-ASL combined video availability
    valid_dataset = load_and_filter_metadata(min_samples=MIN_VIDEOS_PER_CLASS)

    if not valid_dataset:
        print("No valid classes found. Please check your MSASL_*.json metadata files.")
        return

    print("\nStarting video download process...")

    # Iterate over qualified sign folder targets
    for folder_name, entries in valid_dataset.items():
        # Resolve target subdirectory (creates new folder or reuses existing folder)
        label_dir = os.path.join(MEDIA_DATASET_DIR, folder_name)
        os.makedirs(label_dir, exist_ok=True)

        for idx, entry in enumerate(entries):
            url = entry['url']
            start_time = entry.get('start_time', 0)
            end_time = entry.get('end_time', 0)
            
            # Use 'msasl_' prefix to prevent overwriting custom webcam recordings (e.g. webcam_1.mp4)
            file_name = f"msasl_{idx}_{entry['split']}.mp4"
            output_path = os.path.join(label_dir, file_name)

            # Download and slice video clip
            download_clip(url, start_time, end_time, output_path)

    # ==============================================================================
    # POST-DOWNLOAD VERIFICATION & AUDIT
    # ==============================================================================
    # Account for deleted/private YouTube videos that failed during download
    print("\n--- Post-Download Verification ---")
    insufficient_classes = []

    for folder_name in valid_dataset.keys():
        label_dir = os.path.join(MEDIA_DATASET_DIR, folder_name)
        total_videos = count_local_videos(label_dir)

        # Flag classes that fell below required sample threshold due to dead links
        if total_videos < MIN_VIDEOS_PER_CLASS:
            insufficient_classes.append((folder_name, total_videos))

    if insufficient_classes:
        print(f"\n[WARN] The following {len(insufficient_classes)} classes dropped below {MIN_VIDEOS_PER_CLASS} total playable videos due to dead links:")
        for folder_name, count in insufficient_classes:
            print(f"  - '{folder_name}': Only {count} total video(s) present in directory")
        print("\nRecommendation: Exclude flagged folders during database indexing or add custom webcam recordings.")
    else:
        print(f"All target sign folders contain at least {MIN_VIDEOS_PER_CLASS} playable videos!")


if __name__ == "__main__":
    fetch_msasl_dataset()