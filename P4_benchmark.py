import json
import os
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==============================================================================
# SYSTEM & PATH CONFIGURATIONS
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, 'saign_vision.db')
FEATURES_DIR = os.path.join(SCRIPT_DIR, 'extracted_features')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'asl_model.keras')
CLASSES_PATH = os.path.join(SCRIPT_DIR, 'classes.json')


def load_all_labels():
    """
    Loads the exact class list saved during training (classes.json),
    falling back to alphabetical unique labels in SQLite if classes.json is absent.
    """
    if os.path.exists(CLASSES_PATH):
        try:
            with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
                classes = json.load(f)
                if isinstance(classes, list) and classes:
                    return [str(c) for c in classes]
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] Failed to read {CLASSES_PATH} ({e}). Falling back to SQLite...")

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Neither {CLASSES_PATH} nor {DB_PATH} found. Run your training script first.")

    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    try:
        cursor = conn.cursor()
        data = cursor.execute("SELECT DISTINCT label FROM training_dataset WHERE label IS NOT NULL ORDER BY label ASC").fetchall()
        return [row[0] for row in data]
    except sqlite3.OperationalError as e:
        raise RuntimeError(f"Database table 'training_dataset' is missing or unreadable: {e}") from e
    finally:
        conn.close()


def load_benchmark_data(split_name):
    """Queries the specific dataset split for benchmarking."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}.")

    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    try:
        cursor = conn.cursor()
        return cursor.execute(
            "SELECT npy_path, label FROM training_dataset WHERE LOWER(split) = ? AND npy_path IS NOT NULL AND label IS NOT NULL",
            (split_name.lower(),)
        ).fetchall()
    finally:
        conn.close()


def resolve_npy_path(npy_path):
    """Resolves .npy file path even if the project folder was moved since extraction."""
    if not npy_path:
        return None
    if os.path.exists(npy_path):
        return npy_path
    # Normalise Windows-style separators so paths saved on another OS still resolve
    portable_path = os.path.join(FEATURES_DIR, os.path.basename(npy_path.replace('\\', '/')))
    return portable_path if os.path.exists(portable_path) else None


def prepare_tensors(data_list, expected_shape, class_to_idx):
    """Loads .npy feature matrices and integer class indices into uniform NumPy arrays."""
    X, y = [], []
    skipped = 0

    for raw_npy_path, label in data_list:
        if label not in class_to_idx:
            print(f"[WARN] Label '{label}' is not in trained classes.json. Skipping...")
            skipped += 1
            continue

        npy_path = resolve_npy_path(raw_npy_path)
        if not npy_path:
            skipped += 1
            continue

        try:
            features = np.load(npy_path)
        except Exception as e:  # EOFError / UnpicklingError etc. from truncated files
            print(f"[WARN] Failed to load corrupted feature file '{npy_path}': {e}. Skipping...")
            skipped += 1
            continue

        if features.shape != expected_shape:
            print(f"[WARN] Shape mismatch in '{npy_path}': got {features.shape}, expected {expected_shape}. Skipping...")
            skipped += 1
            continue

        features = np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        X.append(features)
        y.append(class_to_idx[label])

    if skipped > 0:
        print(f"[WARN] Skipped {skipped} missing, unknown-label, or shape-mismatched feature file(s).")

    if not X:
        return np.empty((0, *expected_shape), dtype=np.float32), np.array([], dtype=np.int64)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def run_benchmark():
    print("==================================================")
    print("             SAIgn BENCHMARK UTILITY              ")
    print("==================================================")
    print("[1] VALIDATION MODE (Iterative Development)")
    print("[2] FINAL TEST MODE (Pre-Deployment Holdout)")
    print("==================================================")

    try:
        choice = input("Select mode (1 or 2): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBenchmark cancelled.")
        return

    if choice == '1':
        target_split = 'val'
        output_filename = 'confusion_matrix_val.png'
        print("\n--> Launching Validation Benchmark...")
    elif choice == '2':
        print("\n[WARNING] You have selected the FINAL TEST holdout split.")
        print("This split should only be evaluated once at the very end of the project cycle.")
        try:
            confirm = input("Are you sure you want to proceed? (y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nTest benchmark cancelled. Returning safely.")
            return

        if confirm != 'y':
            print("\nTest benchmark cancelled. Returning safely.")
            return

        target_split = 'test'
        output_filename = 'confusion_matrix_test.png'
        print("\n--> Launching Final Test Benchmark...")
    else:
        print("\nInvalid selection. Exiting.")
        return

    cm_output_path = os.path.join(SCRIPT_DIR, output_filename)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Run your training script first.")

    print("Loading AI model into memory...")
    model = load_model(MODEL_PATH, compile=False)  # inference only; avoids optimizer/loss deserialization issues
    input_shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
    output_shape = model.output_shape[0] if isinstance(model.output_shape, list) else model.output_shape
    expected_input_shape = (input_shape[1] or 30, input_shape[2] or 225)
    model_num_classes = output_shape[-1]

    print("Synchronizing class mappings...")
    class_names = load_all_labels()
    if not class_names:
        print("[ERROR] No class labels found. Run your training script first.")
        return

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    class_indices = np.arange(len(class_names))

    if len(class_names) != model_num_classes:
        raise ValueError(
            f"Class count mismatch: model outputs {model_num_classes} classes, "
            f"but loaded class mapping has {len(class_names)} classes. Retrain the model."
        )

    print(f"Loading '{target_split}' dataset...")
    benchmark_data = load_benchmark_data(target_split)
    if not benchmark_data:
        print(f"[ERROR] No '{target_split}' data found in saign_vision.db. Cannot run benchmark.")
        return

    X_test, y_true = prepare_tensors(benchmark_data, expected_input_shape, class_to_idx)
    if len(X_test) == 0:
        print(f"[ERROR] Zero valid '{target_split}' feature arrays available on disk matching shape {expected_input_shape}.")
        return

    print(f"Running predictions on {len(X_test)} samples...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    overall_acc = accuracy_score(y_true, y_pred)

    print("\n" + "=" * 60)
    print(f"CLASSIFICATION REPORT: {target_split.upper()} SPLIT (Accuracy: {overall_acc * 100:.2f}%)")
    print("=" * 60)
    print(classification_report(
        y_true,
        y_pred,
        labels=class_indices,
        target_names=class_names,
        zero_division=0
    ))

    print("\nGenerating visual Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred, labels=class_indices)

    num_classes = len(class_names)
    fig_dim = max(10, min(32, int(num_classes * 0.45)))
    show_annotations = num_classes <= 30

    plt.figure(figsize=(fig_dim + 2, fig_dim))
    sns.heatmap(
        cm,
        annot=show_annotations,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(f'ASL Model Benchmark ({target_split.capitalize()} Set): Confusion Matrix', pad=20, fontsize=16)
    plt.ylabel('Actual Sign (Ground Truth)', fontsize=12)
    plt.xlabel('Predicted Sign (Model Output)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Very large class counts at 300 DPI can need hundreds of MB of RAM; scale DPI down for big figures
    save_dpi = 300 if fig_dim <= 16 else 150 if fig_dim <= 24 else 100
    plt.savefig(cm_output_path, dpi=save_dpi)
    print(f"Confusion matrix heat map saved successfully to: {cm_output_path}")

    try:
        plt.show()
    except Exception as e:
        print(f"[WARN] Could not open interactive display window ({e}). Saved chart is at {cm_output_path}.")
    finally:
        plt.close('all')


if __name__ == "__main__":
    run_benchmark()