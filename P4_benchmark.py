import json
import os
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ==============================================================================
# SYSTEM & PATH CONFIGURATIONS
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, 'saign_vision.db')
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
                    return classes
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] Failed to read {CLASSES_PATH} ({e}). Falling back to SQLite...")

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Neither {CLASSES_PATH} nor {DB_PATH} found. Run P4_train_model_3.py first.")

    conn = sqlite3.connect(DB_PATH)
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

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        return cursor.execute(
            "SELECT npy_path, label FROM training_dataset WHERE split = ? AND npy_path IS NOT NULL AND label IS NOT NULL",
            (split_name,)
        ).fetchall()
    finally:
        conn.close()


def prepare_tensors(data_list, expected_shape, known_classes):
    """Loads .npy feature matrices and string labels into uniform NumPy arrays."""
    X, y = [], []
    skipped = 0
    known_set = set(known_classes)

    for npy_path, label in data_list:
        if label not in known_set:
            print(f"[WARN] Label '{label}' is not in trained classes.json. Skipping...")
            skipped += 1
            continue

        if not npy_path or not os.path.exists(npy_path):
            skipped += 1
            continue

        try:
            features = np.load(npy_path)
        except (OSError, ValueError) as e:
            print(f"[WARN] Failed to load corrupted feature file '{npy_path}': {e}. Skipping...")
            skipped += 1
            continue

        if features.shape != expected_shape:
            print(f"[WARN] Shape mismatch in '{npy_path}': got {features.shape}, expected {expected_shape}. Skipping...")
            skipped += 1
            continue

        X.append(features)
        y.append(label)

    if skipped > 0:
        print(f"[WARN] Skipped {skipped} missing, unknown-label, or shape-mismatched feature file(s).")

    if not X:
        return np.empty((0, *expected_shape), dtype=np.float32), np.array([], dtype=str)

    return np.array(X, dtype=np.float32), np.array(y)


def run_benchmark():
    print("==================================================")
    print("             SAIgn BENCHMARK UTILITY              ")
    print("==================================================")
    print("[1] VALIDATION MODE (Iterative Development)")
    print("[2] FINAL TEST MODE (Pre-Deployment Holdout)")
    print("==================================================")

    choice = input("Select mode (1 or 2): ").strip()

    if choice == '1':
        target_split = 'val'
        output_filename = 'confusion_matrix_val.png'
        print("\n--> Launching Validation Benchmark...")
    elif choice == '2':
        print("\n[WARNING] You have selected the FINAL TEST holdout split.")
        print("This split should only be evaluated once at the very end of the project cycle.")
        confirm = input("Are you sure you want to proceed? (y/n): ").strip().lower()

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
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Run P4_train_model_3.py first.")

    print("Loading AI model into memory...")
    model = load_model(MODEL_PATH)
    expected_input_shape = tuple(model.input_shape[1:])
    model_num_classes = model.output_shape[-1]

    print("Synchronizing class mappings...")
    all_labels = load_all_labels()
    if not all_labels:
        print("[ERROR] No class labels found. Run P4_train_model_3.py first.")
        return

    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    class_names = label_encoder.classes_
    class_indices = np.arange(len(class_names))

    if len(class_names) != model_num_classes:
        raise ValueError(
            f"Class count mismatch: model outputs {model_num_classes} classes, "
            f"but loaded class mapping has {len(class_names)} classes. Retrain with P4_train_model_3.py."
        )

    print(f"Loading '{target_split}' dataset...")
    benchmark_data = load_benchmark_data(target_split)
    if not benchmark_data:
        print(f"[ERROR] No '{target_split}' data found in saign_vision.db. Cannot run benchmark.")
        return

    X_test, y_test_str = prepare_tensors(benchmark_data, expected_input_shape, class_names)
    if len(X_test) == 0:
        print(f"[ERROR] Zero valid '{target_split}' feature arrays available on disk matching shape {expected_input_shape}.")
        return

    y_true = label_encoder.transform(y_test_str)

    print(f"Running predictions on {len(X_test)} samples...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n" + "=" * 60)
    print(f"CLASSIFICATION REPORT: {target_split.upper()} SPLIT")
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

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
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

    plt.savefig(cm_output_path, dpi=300)
    print(f"Confusion matrix heat map saved successfully to: {cm_output_path}")
    plt.show()
    plt.close()


if __name__ == "__main__":
    run_benchmark()