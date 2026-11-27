import argparse
import glob
import json
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
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
BATCH_SIZE_CLASSES = 20  # Process _ signs classes per execution run (up to developer discretion; smaller batches are safer for flaky networks or throttled YouTube access)
VALID_VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv')
DOWNLOAD_TIMEOUT_SEC = 120

# Pause (random, in this range of seconds) between actual downloads. YouTube rate-limits rapid repeated
# requests (~300 videos/hour for guest sessions, ~2000/hour when signed in); the yt-dlp wiki recommends
# 5-10 s between downloads. Set to (0, 0) to disable.
DOWNLOAD_DELAY_RANGE_SEC = (5.0, 10.0)

# Stop the run (and un-count those attempts) after this many transient failures IN A ROW. That pattern
# means YouTube/network trouble, not individually bad clips, so it must not burn per-clip retry budgets.
MAX_CONSECUTIVE_TRANSIENT = 5

# Public, long-lived video used by the pre-flight check ("Me at the zoo", YouTube's first upload).
PREFLIGHT_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"

# YouTube Authentication Configuration
# YouTube now often demands a signed-in session ("Please sign in" / "confirm you're not a bot").
# Cookie source, highest priority first (the one in use is printed at startup):
#   1. --browser NAME / --cookies-file PATH on the command line (override everything below)
#   2. a cookies.txt file (Netscape format) in this folder  <- MOST RELIABLE, see how-to below
#   3. COOKIES_FROM_BROWSER, e.g. "firefox", "chrome", "edge", "brave", "safari",
#      or with a profile: "chrome:Profile 1" / "firefox:myprofile"
#
# Why cookies.txt is most reliable: YouTube rotates cookies in any browser tab that stays open, which
# silently invalidates live browser cookies. To export cookies that are never rotated:
#   a. open a NEW private/incognito window and sign in to YouTube (a throwaway Google account is wise:
#      automated use of an account can get it flagged),
#   b. in that same tab go to https://www.youtube.com/robots.txt (keep it the only private tab),
#   c. export youtube.com cookies with a "cookies.txt" extension (e.g. "Get cookies.txt LOCALLY"),
#   d. save the file as cookies.txt next to this script, then CLOSE the private window (don't sign out).
# Chrome/Edge on Windows encrypt their cookie store so yt-dlp often cannot read it: use Firefox or cookies.txt.
COOKIES_FROM_BROWSER = "chrome"
COOKIES_FILE_PATH = os.path.join(SCRIPT_DIR, 'cookies.txt')
NO_COOKIES = False  # set by --no-cookies

# Manifest status values (per clip): "<filename>" = downloaded, FAILED_MARKER = permanently
# unavailable, "__RETRY__:<n>" = n transient failures so far (timeouts, flaky network, ...).
FAILED_MARKER = "__FAILED__"
RETRY_PREFIX = "__RETRY__:"
MAX_TRANSIENT_ATTEMPTS = 3

# Set to True to download ONLY signs listed in priority_signs.txt and ignore the rest of MS-ASL
ONLY_DOWNLOAD_PRIORITY = False

# Set to True to re-attempt previously failed/unavailable YouTube clips
RETRY_FAILED = False

# yt-dlp error text that means the clip is gone for good (never worth retrying)
PERMANENT_ERROR_MARKERS = (
    "video unavailable", "this video is unavailable", "this video is not available",
    "private video", "has been removed", "no longer available", "has been terminated",
    "copyright", "members-only", "join this channel", "confirm your age",
    "is not a valid url", "unsupported url", "incomplete youtube id",
)
# yt-dlp error text meaning YouTube is throttling/blocking us or demands sign-in: stop instead of poisoning the manifest
BLOCKED_ERROR_MARKERS = (
    "not a bot", "http error 429", "too many requests",
    "please sign in", "sign in to confirm", "use --cookies-from-browser",
)
# yt-dlp could not READ or USE the configured cookies. Every later clip would fail the same way, so stop.
COOKIE_PROBLEM_MARKERS = (
    "cookie database", "cookies database", "failed to decrypt", "keyring",
    "no longer valid", "likely been rotated", "not a supported browser", "unsupported browser",
)
AGE_GATE_MARKER = "confirm your age"
JS_RUNTIME_WARNING_MARKER = "no supported javascript runtime"
ALREADY_ON_DISK = "already on disk"
NO_NETWORK_DETAILS = (ALREADY_ON_DISK, "invalid url or time range")

DL_OK, DL_PERMANENT, DL_TRANSIENT, DL_BLOCKED = "ok", "permanent", "transient", "blocked"

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
    "7. Alphabet (Fingerspelling)": [
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"
    ],
}


# ==============================================================================
# SMALL UTILITIES
# ==============================================================================
def sanitize_label(text):
    """Normalizes sign text into a safe snake_case folder/file identifier."""
    cleaned = (text or "").strip().lower().replace(" ", "_")
    return re.sub(r'[\\/:*?"<>|%]+', '_', cleaned)


def parse_timestamp(value, default=0.0):
    """Safely converts a metadata timestamp to float. None -> default; unparseable -> None."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
            clean_line = raw_line.split('#', 1)[0].strip()
            if not clean_line:
                continue

            normalized = sanitize_label(clean_line)
            if normalized and normalized not in priority_map:
                priority_map[normalized] = len(priority_map)

    return priority_map


def get_youtube_id(url):
    """
    Extracts the unique 11-character YouTube video ID from a URL string.
    e.g., 'https://www.youtube.com/watch?v=dQw4w9WgXcQ' -> 'dQw4w9WgXcQ'
    """
    if not isinstance(url, str) or not url:
        return None
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:[&#?/]|$)", url)
    return match.group(1) if match else None


def get_clip_keys(entry):
    """
    Returns candidate manifest keys for a metadata entry, most-specific first (index 0 is the
    "primary" key new records are written under). Includes a legacy integer-truncated key
    (the format used before split-aware, sub-second-precise keys existed) so manifests built
    by older versions of this script are still recognized and not re-downloaded.
    """
    url = (entry.get('url') or '').strip()
    start_time = parse_timestamp(entry.get('start_time'), 0.0) or 0.0
    end_time = parse_timestamp(entry.get('end_time'), 0.0) or 0.0
    split_tag = (entry.get('split') or 'train').strip().lower()

    yt_id = get_youtube_id(url)
    if yt_id:
        primary = f"{yt_id}_{round(start_time, 3)}_{round(end_time, 3)}_{split_tag}"
        if start_time.is_integer() and end_time.is_integer():
            legacy = f"{yt_id}_{int(start_time)}_{int(end_time)}"
            return (primary, legacy)
        return (primary,)

    primary = f"{url}_{round(start_time, 3)}_{round(end_time, 3)}_{split_tag}"
    return (primary,)


# ==============================================================================
# MANIFEST (crash-safe)
# ==============================================================================
def load_download_manifest():
    """Loads clip_key -> status records. A corrupt manifest is preserved aside, never silently discarded."""
    if not os.path.exists(MANIFEST_PATH):
        return {}
    try:
        with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        raise ValueError("manifest root is not an object")
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        backup = MANIFEST_PATH + '.corrupt'
        try:
            os.replace(MANIFEST_PATH, backup)
            print(f"[WARN] Download manifest was unreadable ({exc}); moved to {backup}. Starting a fresh manifest.")
        except OSError:
            print(f"[WARN] Download manifest was unreadable ({exc}). Starting a fresh manifest.")
        return {}


def save_download_manifest(manifest):
    """Atomically persists the manifest (temp file + os.replace) so an interruption can't corrupt it."""
    os.makedirs(MEDIA_DATASET_DIR, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=MEDIA_DATASET_DIR, suffix='.manifest.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp_path, MANIFEST_PATH)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def transient_attempts(status):
    """Number of transient failures recorded in a '__RETRY__:<n>' status (0 for anything else)."""
    if isinstance(status, str) and status.startswith(RETRY_PREFIX):
        try:
            return int(status[len(RETRY_PREFIX):])
        except ValueError:
            return MAX_TRANSIENT_ATTEMPTS
    return 0


def is_clip_handled(folder_name, folder_path, clip_keys, manifest, retry_failed=None):
    """
    True if NONE of a clip's candidate keys need another download attempt: the clip is on disk,
    permanently unavailable, or has exhausted its transient-retry budget. Checking every
    candidate key keeps old (legacy-format) manifest records recognized across upgrades.
    """
    if retry_failed is None:
        retry_failed = RETRY_FAILED  # read at call time so the --retry-failed flag takes effect
    folder_manifest = manifest.get(folder_name)
    if not isinstance(folder_manifest, dict):
        return False

    for key in clip_keys:
        status = folder_manifest.get(key)
        if not status or not isinstance(status, str):
            continue

        if status == FAILED_MARKER:
            if not retry_failed:
                return True
            continue

        if status.startswith(RETRY_PREFIX):
            if (not retry_failed) and transient_attempts(status) >= MAX_TRANSIENT_ATTEMPTS:
                return True
            continue

        full_path = os.path.join(folder_path, status)
        if os.path.isfile(full_path) and os.path.getsize(full_path) > 0:
            return True

    return False


# ==============================================================================
# LOCAL DATASET HELPERS
# ==============================================================================
def load_db_split_tags():
    """
    Loads (clean_text, url, start_time, end_time) -> split tag mappings from
    saign_vision.db (populated by the MS-ASL ingest script).
    """
    tag_lookup = {}
    if not os.path.exists(DB_PATH):
        return tag_lookup

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT clean_text, url, start_time, end_time, split
            FROM video_clips
            WHERE split IS NOT NULL
        """)
        for clean_text, url, start_time, end_time, split in cursor.fetchall():
            norm_text = sanitize_label(clean_text)
            st = parse_timestamp(start_time, 0.0)
            et = parse_timestamp(end_time, 0.0)
            if st is None or et is None:
                continue
            key = (norm_text, (url or '').strip(), round(st, 3), round(et, 3))
            tag_lookup[key] = (split or 'train').strip().lower()
    except sqlite3.Error:
        pass  # table not created yet: fall back to the split implied by each JSON filename
    finally:
        if conn:
            conn.close()

    return tag_lookup


def get_existing_folders():
    """Scans the local media_dataset/ directory for existing gesture subdirectories."""
    folder_map = {}
    if os.path.exists(MEDIA_DATASET_DIR):
        for folder in os.listdir(MEDIA_DATASET_DIR):
            folder_path = os.path.join(MEDIA_DATASET_DIR, folder)
            if os.path.isdir(folder_path) and not folder.startswith('.'):
                folder_map[sanitize_label(folder)] = folder
    return folder_map


def count_local_videos(folder_path):
    """Counts total non-empty playable video files currently stored in a local directory."""
    if not os.path.exists(folder_path):
        return 0
    count = 0
    for f in os.listdir(folder_path):
        if f.lower().endswith(VALID_VIDEO_EXTS):
            fpath = os.path.join(folder_path, f)
            if os.path.isfile(fpath) and os.path.getsize(fpath) > 0:
                count += 1
    return count


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


# ==============================================================================
# METADATA LOADING & FILTERING
# ==============================================================================
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

        base_name = os.path.basename(json_path).lower()
        if 'train' in base_name:
            file_split = 'train'
        elif 'val' in base_name:
            file_split = 'val'
        elif 'test' in base_name:
            file_split = 'test'
        else:
            file_split = 'train'

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[WARN] Failed to parse {json_path}: {exc}. Skipping...")
            continue

        if not isinstance(data, list):
            print(f"[WARN] Unexpected JSON format in {json_path}. Expected a list. Skipping...")
            continue

        for entry in data:
            if not isinstance(entry, dict):
                continue

            raw_text = sanitize_label(entry.get('clean_text') or entry.get('text'))
            url = (entry.get('url') or '').strip()
            if not raw_text or not url:
                continue

            if ONLY_DOWNLOAD_PRIORITY and raw_text not in priority_map:
                continue

            start_time = parse_timestamp(entry.get('start_time'), 0.0)
            end_time = parse_timestamp(entry.get('end_time'), 0.0)
            if start_time is None or end_time is None or end_time <= start_time:
                continue

            db_key = (raw_text, url, round(start_time, 3), round(end_time, 3))
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

        # De-duplicate by primary clip key: the same clip can legitimately appear twice in the
        # source JSON (or across train/val/test files); it must only count once toward the
        # class's sample total and only be queued for download once.
        seen_pending_keys = set()
        for entry in entries:
            clip_keys = get_clip_keys(entry)
            if clip_keys[0] in seen_pending_keys:
                continue
            if not is_clip_handled(folder_name, folder_path, clip_keys, manifest):
                seen_pending_keys.add(clip_keys[0])

        total_potential = existing_count + len(seen_pending_keys)

        if total_potential >= min_samples:
            valid_dataset[folder_name] = entries
        else:
            discarded_classes += 1

    priority_matches = sum(1 for k in valid_dataset if sanitize_label(k) in priority_map)
    print(f"\n--- Pre-Download Filtering ---")
    print(f"Classes meeting >= {min_samples} total possible video requirement: {len(valid_dataset)} ({priority_matches} priority signs matched)")
    print(f"Classes discarded: {discarded_classes}")

    return valid_dataset


# ==============================================================================
# DOWNLOADING
# ==============================================================================
def resolve_ytdlp_command():
    """Returns the argv prefix for yt-dlp (PATH executable, or `python -m yt_dlp`), or None if unavailable."""
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]
    try:
        probe = subprocess.run([sys.executable, "-m", "yt_dlp", "--version"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        if probe.returncode == 0:
            return [sys.executable, "-m", "yt_dlp"]
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def summarize_error(stderr_text):
    """Most informative single line of yt-dlp stderr: the last 'ERROR' line, else the last non-empty line."""
    lines = [ln.strip() for ln in (stderr_text or "").splitlines() if ln.strip()]
    errors = [ln for ln in lines if ln.upper().startswith("ERROR")]
    if errors:
        return errors[-1]
    return lines[-1] if lines else ""


def classify_failure(stderr_text, have_cookies=True):
    text = (stderr_text or "").lower()
    if any(marker in text for marker in COOKIE_PROBLEM_MARKERS):
        return DL_BLOCKED
    # Age-gate with a signed-in session configured = this particular clip is out of reach (e.g. the account
    # is not age-verified). Without cookies, cookies are the fix, so stop and tell the user instead.
    if AGE_GATE_MARKER in text and "not a bot" not in text:
        return DL_PERMANENT if have_cookies else DL_BLOCKED
    if any(marker in text for marker in BLOCKED_ERROR_MARKERS):
        return DL_BLOCKED
    if any(marker in text for marker in PERMANENT_ERROR_MARKERS):
        return DL_PERMANENT
    return DL_TRANSIENT


def get_cookie_args():
    """Builds yt-dlp authentication flags: cookies.txt first, then browser cookies (see config notes)."""
    if NO_COOKIES:
        return []
    if COOKIES_FILE_PATH and os.path.isfile(COOKIES_FILE_PATH):
        return ["--cookies", COOKIES_FILE_PATH]
    if COOKIES_FROM_BROWSER:
        return ["--cookies-from-browser", COOKIES_FROM_BROWSER]
    return []


def describe_cookie_source():
    args = get_cookie_args()
    if not args:
        return "none (guest session)"
    if args[0] == "--cookies":
        return f"cookies file {args[1]}"
    return f"{args[1]} browser cookies"


_JS_SUPPORT_CACHE = {}


def ytdlp_supports_js_runtimes(ytdlp_cmd):
    """True if this yt-dlp knows --js-runtimes (added together with YouTube JS-runtime support, 2025.11.12)."""
    key = tuple(ytdlp_cmd)
    if key not in _JS_SUPPORT_CACHE:
        try:
            probe = subprocess.run(ytdlp_cmd + ["--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   timeout=30, encoding='utf-8', errors='replace')
            _JS_SUPPORT_CACHE[key] = "--js-runtimes" in (probe.stdout or "")
        except (OSError, subprocess.SubprocessError):
            _JS_SUPPORT_CACHE[key] = False
    return _JS_SUPPORT_CACHE[key]


def get_js_runtime_args(ytdlp_cmd):
    """
    Recent yt-dlp needs an external JavaScript runtime for full YouTube support. Deno is used
    automatically when installed; Node/Bun must be enabled explicitly. Returns [] when not applicable.
    """
    if not ytdlp_supports_js_runtimes(ytdlp_cmd):
        return []
    if shutil.which("deno"):
        return []
    for runtime in ("node", "bun"):
        if shutil.which(runtime):
            return ["--js-runtimes", runtime]
    return []


def diagnose_failure(stderr_text, have_cookies):
    """Human-readable, actionable explanations for the failure patterns seen in yt-dlp's stderr."""
    text = (stderr_text or "").lower()
    hints = []
    if any(m in text for m in COOKIE_PROBLEM_MARKERS):
        hints.append("yt-dlp could not read or use your cookies (browser cookie store locked/encrypted, or the "
                     "cookies were rotated by YouTube). Close the browser, or export a fresh cookies.txt from a "
                     "private window (see the notes above COOKIES_FROM_BROWSER) and re-run.")
    if any(m in text for m in ("please sign in", "sign in to confirm", "not a bot", "use --cookies-from-browser")):
        if not have_cookies:
            hints.append("YouTube wants a signed-in session and no cookies are configured. Add a cookies.txt next to "
                         "this script, or run with --browser firefox (or chrome, edge, brave).")
        else:
            hints.append("YouTube rejected the cookies in use (not signed in, rotated, or the account/IP is flagged). "
                         "Export a FRESH cookies.txt from a private window while signed in (a throwaway account is "
                         "safest), wait some time if you were rate-limited, then re-run.")
    if "http error 429" in text or "too many requests" in text or "rate-limit" in text or "rate limit" in text:
        hints.append("YouTube is rate-limiting this IP/session. Wait an hour or so; keep DOWNLOAD_DELAY_RANGE_SEC at 5-10 s or higher.")
    if JS_RUNTIME_WARNING_MARKER in text:
        hints.append("No JavaScript runtime found for yt-dlp (needed for full YouTube support). Install Deno "
                     "(https://deno.com) or Node 20+, and update yt-dlp: pip install -U \"yt-dlp[default]\".")
    return hints


def download_clip(ytdlp_cmd, url, start_time, end_time, output_path):
    """
    Streams and trims a YouTube clip (capped at 720p for fast landmark extraction) with
    yt-dlp + ffmpeg. Returns (status, detail, stderr_text) where status is one of
    DL_OK / DL_PERMANENT / DL_TRANSIENT / DL_BLOCKED.

    yt-dlp's own docs warn against hard-coding a file extension in -o ("is not recommended and
    could break some post-processing") because when --merge-output-format has to remux/merge
    streams, "the actual output filename might differ" from the literal name you passed. An
    earlier version of this function hard-coded a literal ".partial" suffix after ".mp4" in -o;
    yt-dlp would then silently write the real file under a DIFFERENT name (e.g.
    "clip.mp4.partial.mp4"), so every download looked like a failure (yt-dlp itself reported exit
    code 0, but our literal expected path never appeared) -- and the cleanup step then deleted the
    real file it mistook for leftover debris, before the clip was ever retried.

    A second attempt fixed that by using the %(ext)s template field, but appended it as the final
    component of a name still living directly in label_dir -- so the resulting file legitimately
    ended in ".mp4" and get_next_video_number()/count_local_videos() (which only check the
    extension) miscounted an in-progress download as an already-finished clip, reintroducing a
    duplicate-on-crash bug through a different path.

    The fix here: download into a disposable temp SUBDIRECTORY inside label_dir, using yt-dlp's
    own recommended "name.%(ext)s" pattern with nothing after the extension field (its documented
    safe form). Whatever file yt-dlp actually produces, under whatever extension, is found by
    glob afterward and moved onto output_path -- the temp directory itself is never mistaken for a
    video file (directories don't match VALID_VIDEO_EXTS), so a kill mid-download leaves nothing
    that could be miscounted, and it's removed in `finally` either way.
    """
    if not url or end_time <= start_time:
        return DL_PERMANENT, "invalid url or time range", ""

    if os.path.exists(output_path):
        if os.path.getsize(output_path) > 0:
            return DL_OK, ALREADY_ON_DISK, ""
        os.remove(output_path)

    label_dir = os.path.dirname(output_path)
    os.makedirs(label_dir, exist_ok=True)

    # A leftover staging directory from a previous crashed/killed attempt (any clip in this
    # folder) is necessarily stale -- a live run never leaves one behind past a single
    # synchronous call to this function -- so clear any before starting a fresh one.
    for stale in glob.glob(os.path.join(glob.escape(label_dir), '.dl_staging_*')):
        shutil.rmtree(stale, ignore_errors=True)

    staging_dir = tempfile.mkdtemp(dir=label_dir, prefix='.dl_staging_')
    try:
        cookie_args = get_cookie_args()
        # Warnings are deliberately NOT suppressed: they carry the cookie / JS-runtime diagnostics.
        cmd = ytdlp_cmd + [
            "--quiet",
            "--no-playlist",
            "--format", "best[ext=mp4][height<=720]/bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "--external-downloader", "ffmpeg",
            "--external-downloader-args", f"ffmpeg_i:-ss {start_time} -to {end_time}",
            # %(ext)s as the LAST path component, nothing after it -- yt-dlp's documented safe
            # form, so its post-merge extension-correction logic has nothing ambiguous to collide
            # with (see docstring above).
            "-o", os.path.join(staging_dir, "clip.%(ext)s"),
            *get_js_runtime_args(ytdlp_cmd),
            *cookie_args,
            "--",
            url
        ]

        try:
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=DOWNLOAD_TIMEOUT_SEC, encoding='utf-8', errors='replace'
            )
        except subprocess.TimeoutExpired:
            return DL_TRANSIENT, f"timed out after {DOWNLOAD_TIMEOUT_SEC}s", ""
        except OSError as exc:
            return DL_TRANSIENT, f"could not launch yt-dlp: {exc}", ""

        junk_suffixes = ('.part', '.ytdl', '.aria2', '.temp')
        produced = [
            p for p in glob.glob(os.path.join(glob.escape(staging_dir), 'clip.*'))
            if not p.endswith(junk_suffixes)
        ]

        if result.returncode == 0 and len(produced) == 1 and os.path.getsize(produced[0]) > 0:
            os.replace(produced[0], output_path)  # real filename appears only once complete
            return DL_OK, "", ""

        stderr_text = result.stderr or ""
        if result.returncode == 0 and not stderr_text.strip():
            detail = "yt-dlp reported success (exit code 0) but produced no single matching output file"
        else:
            detail = summarize_error(stderr_text) or f"exit code {result.returncode}"
        return classify_failure(stderr_text, bool(cookie_args)), detail, stderr_text
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def run_preflight(ytdlp_cmd):
    """
    One cheap yt-dlp request (metadata only, no download) against a public video, using the configured
    cookies, BEFORE touching the dataset. Returns False only when the failure would block every download.
    """
    print("Running pre-flight check against YouTube...")
    cookie_args = get_cookie_args()
    cmd = ytdlp_cmd + [
        "--simulate", "--no-playlist", "--print", "id",
        *get_js_runtime_args(ytdlp_cmd), *cookie_args, "--", PREFLIGHT_URL
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=90, encoding='utf-8', errors='replace')
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"[WARN] Pre-flight check could not complete ({exc}). Continuing anyway.")
        return True

    stderr_text = result.stderr or ""
    hints = diagnose_failure(stderr_text, bool(cookie_args))

    if result.returncode == 0:
        print("[OK] yt-dlp can reach YouTube with the current settings.")
        for hint in hints:
            print(f"[WARN] {hint}")
        print("     (This check cannot prove the cookies are signed in; if clips still ask you to sign in, refresh them.)")
        return True

    print(f"[PRE-FLIGHT FAILED] {summarize_error(stderr_text) or 'exit code ' + str(result.returncode)}")
    for hint in hints:
        print(f"  -> {hint}")
    if classify_failure(stderr_text, bool(cookie_args)) == DL_BLOCKED:
        return False
    print("[WARN] This looks like a temporary or network problem, not a sign-in block. Continuing anyway.")
    return True


# ==============================================================================
# MAIN LOOP
# ==============================================================================
def fetch_msasl_dataset(skip_preflight=False):
    """Main execution loop: reads priority_signs.txt, downloads priority classes first, then general classes."""
    ytdlp_cmd = resolve_ytdlp_command()
    if ytdlp_cmd is None or not shutil.which("ffmpeg"):
        raise RuntimeError(
            "Both 'yt-dlp' and 'ffmpeg' must be installed and available on your PATH "
            "before downloading MS-ASL clips (pip install yt-dlp; install ffmpeg separately)."
        )

    have_cookies = bool(get_cookie_args())
    print(f"YouTube cookies: {describe_cookie_source()}")
    if not have_cookies:
        print("[WARN] No YouTube cookies configured. If YouTube asks you to sign in, add a cookies.txt next to this script or use --browser NAME.")

    if not skip_preflight and not run_preflight(ytdlp_cmd):
        print("\nStopping before any download: fix the problem above, then re-run (nothing was changed in your dataset or manifest).")
        return

    os.makedirs(MEDIA_DATASET_DIR, exist_ok=True)
    priority_map = load_priority_map(PRIORITY_FILE_PATH)
    print(f"Loaded {len(priority_map)} priority signs from {os.path.basename(PRIORITY_FILE_PATH)}.")

    valid_dataset = load_and_filter_metadata(priority_map, min_samples=MIN_VIDEOS_PER_CLASS)

    if not valid_dataset:
        print("No valid classes found.")
        return

    manifest = load_download_manifest()
    fallback_rank = len(priority_map)

    classes_with_pending = []
    for folder_name, entries in valid_dataset.items():
        label_dir = os.path.join(MEDIA_DATASET_DIR, folder_name)
        existing_count = count_local_videos(label_dir)
        norm_name = sanitize_label(folder_name)
        is_priority = norm_name in priority_map

        pending = []
        seen_keys = set()
        for entry in entries:
            clip_keys = get_clip_keys(entry)
            primary_key = clip_keys[0]
            if primary_key in seen_keys:
                continue
            if not is_clip_handled(folder_name, label_dir, clip_keys, manifest):
                seen_keys.add(primary_key)
                pending.append((entry, primary_key))

        if pending:
            classes_with_pending.append({
                'folder_name': folder_name,
                'existing_count': existing_count,
                'is_priority': is_priority,
                'priority_rank': priority_map.get(norm_name, fallback_rank),
                'pending': pending
            })

    if not classes_with_pending:
        print("\nAll eligible sign classes are already processed!")
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

    blocked = False
    stop_message = ""
    transient_streak = []      # (folder, clip_key, previous manifest value) for failures in a row
    did_network_call = False

    for c_idx, item in enumerate(batch_classes, 1):
        folder_name = item['folder_name']
        label_dir = os.path.join(MEDIA_DATASET_DIR, folder_name)
        os.makedirs(label_dir, exist_ok=True)
        if not isinstance(manifest.get(folder_name), dict):
            manifest[folder_name] = {}

        next_num = get_next_video_number(label_dir, folder_name)
        tier_tag = "[PRIORITY]" if item['is_priority'] else "[GENERAL]"

        print(f"[{c_idx}/{len(batch_classes)}] {tier_tag} Sign Class: '{folder_name}' (Existing on disk: {item['existing_count']}, Pending in batch: {len(item['pending'])})")

        success_count = 0
        unavailable_count = 0
        retry_later_count = 0

        for entry, clip_key in item['pending']:
            url = (entry.get('url') or '').strip()
            start_time = parse_timestamp(entry.get('start_time'), 0.0) or 0.0
            end_time = parse_timestamp(entry.get('end_time'), 0.0) or 0.0
            tag = (entry.get('split') or 'train').strip().lower()

            file_name = f"{folder_name}_{next_num}_{tag}.mp4"
            output_path = os.path.join(label_dir, file_name)

            if did_network_call and DOWNLOAD_DELAY_RANGE_SEC[1] > 0:
                time.sleep(random.uniform(*DOWNLOAD_DELAY_RANGE_SEC))  # be polite: avoids rate limits

            status, detail, stderr_text = download_clip(ytdlp_cmd, url, start_time, end_time, output_path)
            if detail not in NO_NETWORK_DETAILS:
                did_network_call = True

            if status == DL_OK:
                manifest[folder_name][clip_key] = file_name
                success_count += 1
                next_num += 1
                transient_streak.clear()
            elif status == DL_PERMANENT:
                manifest[folder_name][clip_key] = FAILED_MARKER
                unavailable_count += 1
                transient_streak.clear()
            elif status == DL_TRANSIENT:
                previous = manifest[folder_name].get(clip_key)
                attempts = transient_attempts(previous) + 1
                manifest[folder_name][clip_key] = f"{RETRY_PREFIX}{attempts}"
                retry_later_count += 1
                print(f"   [retry {attempts}/{MAX_TRANSIENT_ATTEMPTS}] {clip_key}: {detail}")
                transient_streak.append((folder_name, clip_key, previous))
                if len(transient_streak) >= MAX_CONSECUTIVE_TRANSIENT:
                    # Many failures in a row = a systemic problem. Un-count them so no clip loses retry budget.
                    for f_name, c_key, prev in transient_streak:
                        if prev is None:
                            manifest[f_name].pop(c_key, None)
                        else:
                            manifest[f_name][c_key] = prev
                    print(f"\n[STOP] {MAX_CONSECUTIVE_TRANSIENT} downloads in a row failed temporarily (last: {detail}).")
                    print("       Those attempts were NOT counted against the clips' retry budgets.")
                    for hint in diagnose_failure(stderr_text, have_cookies):
                        print(f"  -> {hint}")
                    stop_message = "Run stopped early after repeated temporary failures. Check your connection / YouTube access and re-run."
                    blocked = True
                    break
            else:  # DL_BLOCKED: record nothing, stop so good clips aren't marked failed
                print(f"\n[STOP] YouTube is blocking requests or the cookies are not usable ({detail}).")
                print("       Nothing was marked as failed.")
                for hint in diagnose_failure(stderr_text, have_cookies):
                    print(f"  -> {hint}")
                stop_message = "Run stopped early because of YouTube sign-in enforcement or throttling. Fix the cookies (see above) and re-run."
                blocked = True
                break

            # Save after EVERY clip: an interruption can never leave downloaded files unrecorded
            # (which would otherwise cause them to be re-downloaded as duplicates next run).
            save_download_manifest(manifest)

        total_now = count_local_videos(label_dir)
        print(
            f"   -> Downloaded {success_count}/{len(item['pending'])} new clips "
            f"({unavailable_count} permanently unavailable, {retry_later_count} will retry). "
            f"Current total in folder: {total_now}\n"
        )

        if blocked:
            break

    save_download_manifest(manifest)

    remaining_classes = len(classes_with_pending) - len(batch_classes)
    remaining_priority = sum(1 for x in classes_with_pending[BATCH_SIZE_CLASSES:] if x['is_priority'])
    print("--- Batch Execution Summary ---")
    if blocked:
        print(stop_message)
    elif remaining_classes > 0:
        print(f"Batch completed! Remaining classes: {remaining_classes} ({remaining_priority} priority signs left).")
        print(f"Run 'python {os.path.basename(__file__)}' again to process the next batch of {BATCH_SIZE_CLASSES} sign classes.")
    else:
        print("All pending sign classes across the dataset have been processed!")



def reset_transient_retries():
    """Clears '__RETRY__:<n>' records (e.g. ones burned by earlier sign-in errors) so those clips are retried."""
    manifest = load_download_manifest()
    cleared = 0
    for clips in manifest.values():
        if isinstance(clips, dict):
            for key in [k for k, v in clips.items() if isinstance(v, str) and v.startswith(RETRY_PREFIX)]:
                del clips[key]
                cleared += 1
    if cleared:
        save_download_manifest(manifest)
    print(f"Reset {cleared} temporary-failure record(s) in the download manifest.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Download MS-ASL sign clips from YouTube (in batches).")
    parser.add_argument("--browser", metavar="NAME[:PROFILE]",
                        help="read cookies from this browser (firefox, chrome, edge, brave, safari); overrides cookies.txt")
    parser.add_argument("--cookies-file", metavar="PATH", help="use this Netscape cookies.txt file")
    parser.add_argument("--no-cookies", action="store_true", help="do not send any cookies (guest session)")
    parser.add_argument("--skip-preflight", action="store_true", help="skip the YouTube pre-flight check")
    parser.add_argument("--reset-retries", action="store_true",
                        help="forget earlier temporary-failure counts (fixes clips wrongly skipped after sign-in errors)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="also re-attempt clips previously recorded as permanently unavailable")
    return parser.parse_args(argv)


def apply_cli_overrides(args):
    global COOKIES_FROM_BROWSER, COOKIES_FILE_PATH, NO_COOKIES, RETRY_FAILED
    if args.retry_failed:
        RETRY_FAILED = True
    if args.no_cookies:
        NO_COOKIES = True
    elif args.browser:
        COOKIES_FROM_BROWSER = args.browser
        COOKIES_FILE_PATH = None  # an explicit browser choice beats a stale cookies.txt
    elif args.cookies_file:
        if not os.path.isfile(args.cookies_file):
            raise SystemExit(f"[ERROR] --cookies-file not found: {args.cookies_file}")
        COOKIES_FILE_PATH = os.path.abspath(args.cookies_file)


if __name__ == "__main__":
    cli_args = parse_args()
    apply_cli_overrides(cli_args)
    if cli_args.reset_retries:
        reset_transient_retries()
    fetch_msasl_dataset(skip_preflight=cli_args.skip_preflight)