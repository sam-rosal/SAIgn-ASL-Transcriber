import json
import os
import sqlite3
import sys
import numpy as np

# --- Model Training & Preprocessing Imports ---
try:
    from keras.models import Model
    from keras.layers import Input, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D
    from keras.callbacks import EarlyStopping
    from keras.utils import to_categorical
except ImportError:  # TensorFlow-bundled Keras fallback
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# --- System & Path Configurations ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:  # make the sibling P4_extract_features.py importable from any working dir
    sys.path.insert(0, SCRIPT_DIR)

# Feature extraction/caching lives in ONE place: P4_extract_features.py (no duplicated copy here).
from P4_extract_features import (  # noqa: E402
    register_and_process_dataset,
    DB_PATH,
    DATASET_DIR,
    FEATURES_DIR,
    TARGET_FRAMES,
    FEATURE_DIM,
)

MODEL_PATH = os.path.join(SCRIPT_DIR, 'asl_model.keras')
CLASSES_PATH = os.path.join(SCRIPT_DIR, 'classes.json')


def resolve_npy_path(npy_path):
    """Resolves .npy file path even if the project folder was moved since extraction."""
    if not npy_path:
        return None
    if os.path.exists(npy_path):
        return npy_path
    # Normalise Windows-style separators so paths saved on another OS still resolve
    portable_path = os.path.join(FEATURES_DIR, os.path.basename(npy_path.replace('\\', '/')))
    return portable_path if os.path.exists(portable_path) else None


def load_split_data(split_name, db_path=DB_PATH):
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        cursor = conn.cursor()
        query = "SELECT npy_path, label FROM training_dataset WHERE LOWER(split) = ? AND npy_path IS NOT NULL AND label IS NOT NULL"
        return cursor.execute(query, (split_name.lower(),)).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def prepare_tensors(data_list, expected_shape=(TARGET_FRAMES, FEATURE_DIM)):
    """
    Reads .npy files from disk and structures them into uniform X (features) and y (labels) arrays.
    """
    X, y = [], []
    skipped = 0

    for raw_npy_path, label in data_list:
        npy_path = resolve_npy_path(raw_npy_path)
        if not npy_path:
            skipped += 1
            continue

        try:
            features = np.load(npy_path)
        except Exception as e:  # EOFError / UnpicklingError etc. from truncated files
            print(f"[WARN] Failed to load '{npy_path}': {e}. Skipping...")
            skipped += 1
            continue

        if features.shape != expected_shape:
            print(f"[WARN] Shape mismatch in '{npy_path}': got {features.shape}, expected {expected_shape}. Skipping...")
            skipped += 1
            continue

        features = np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        X.append(features)
        y.append(label)

    if skipped > 0:
        print(f"[WARN] Skipped {skipped} missing, corrupted, or shape-mismatched feature file(s).")

    if not X:
        return np.empty((0, *expected_shape), dtype=np.float32), np.array([], dtype=str)

    return np.array(X, dtype=np.float32), np.array(y)


def build_model(sequence_length, feature_dim, num_classes):
    """LSTM + temporal self-attention classifier."""
    inputs = Input(shape=(sequence_length, feature_dim))

    # Recurrent layers retain full 30-step temporal sequences (return_sequences=True)
    x = LSTM(64, return_sequences=True)(inputs)
    x = LSTM(128, return_sequences=True)(x)

    # Self-Attention layer weights key gesture frames across the 30-frame window
    attention_out = Attention()([x, x])

    # Condense temporal sequence into a single context vector before classification
    x = GlobalAveragePooling1D()(attention_out)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def save_model_and_classes(model, class_list):
    """
    Writes model + classes.json via temp files and swaps both in only after both were written, so a
    failure mid-save can't leave a new model paired with an old (or missing) class list.
    """
    tmp_model = os.path.join(SCRIPT_DIR, 'asl_model.tmp.keras')  # must keep the .keras suffix
    tmp_classes = CLASSES_PATH + '.tmp'
    try:
        model.save(tmp_model)
        with open(tmp_classes, 'w', encoding='utf-8') as f:
            json.dump(class_list, f, indent=2)
        os.replace(tmp_model, MODEL_PATH)
        os.replace(tmp_classes, CLASSES_PATH)
    finally:
        for leftover in (tmp_model, tmp_classes):
            if os.path.exists(leftover):
                try:
                    os.remove(leftover)
                except OSError:
                    pass


def main():
    try:  # reproducible weight init / shuffling where supported
        from keras.utils import set_random_seed
        set_random_seed(42)
    except Exception:
        pass

    # Step 1: Execute feature extraction pipeline
    print("Verifying feature extraction status...")
    register_and_process_dataset(verbose_skips=False)

    # Step 2: Retrieve partitioned dataset samples for model training & evaluation
    print("\n--- Loading Dataset Splits from DB ---")
    train_data = load_split_data('train')
    val_data   = load_split_data('val')
    test_data  = load_split_data('test')

    print(f"Training clips found: {len(train_data)}")
    print(f"Validation clips found: {len(val_data)}")
    print(f"Testing clips found: {len(test_data)}")

    if len(train_data) == 0:
        print("No training data found. Add videos named <sign>_<number>_<train|val|test>.mp4 under "
              f"{DATASET_DIR}/<sign>/ (run P4_setup_db.py first if the database is missing). Exiting.")
        sys.exit(1)

    # Step 3: Convert File Paths to NumPy Matrices
    print("\nLoading NumPy features into memory...")
    X_train, y_train_str = prepare_tensors(train_data)
    X_val, y_val_str     = prepare_tensors(val_data)

    if len(X_train) == 0:
        raise RuntimeError(
            "Zero valid training feature tensors could be loaded from disk. "
            "Verify that .npy files exist in 'extracted_features/' with shape (30, 225)."
        )

    # Step 4: One-Hot Encode Trained String Labels
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train_str)
    train_label_set = set(label_encoder.classes_.tolist())

    num_classes = len(label_encoder.classes_)
    if num_classes < 2:
        print(f"[WARN] Only {num_classes} unique sign class found in training split. Add at least 2 classes for meaningful classification.")

    y_train = to_categorical(label_encoder.transform(y_train_str), num_classes=num_classes)

    if len(X_val) > 0:
        known_val_mask = np.array([lbl in train_label_set for lbl in y_val_str], dtype=bool)
        unseen_val_labels = sorted(set(y_val_str[~known_val_mask].tolist()))
        if unseen_val_labels:
            print(
                f"[WARN] Excluding {int((~known_val_mask).sum())} validation sample(s) across "
                f"{len(unseen_val_labels)} class(es) with 0 training clips: {unseen_val_labels[:10]}"
            )
            X_val = X_val[known_val_mask]
            y_val_str = y_val_str[known_val_mask]

    has_validation = len(X_val) > 0
    if has_validation:
        y_val = to_categorical(label_encoder.transform(y_val_str), num_classes=num_classes)
        validation_tuple = (X_val, y_val)
        monitor_metric = 'val_loss'
    else:
        print("[WARN] No valid validation tensors matching trained classes found. Monitoring training 'loss'.")
        validation_tuple = None
        monitor_metric = 'loss'

    sequence_length = X_train.shape[1]  # 30 frames
    feature_dim = X_train.shape[2]      # 225 Holistic features

    # Step 5: Define LSTM + Temporal Attention Architecture
    print("\n--- Building LSTM + Attention Neural Network ---")
    model = build_model(sequence_length, feature_dim, num_classes)

    # Step 6: Define Early Stopping Callback
    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    # Step 7: Train the Model
    print("\n--- Commencing Training ---")
    model.fit(
        X_train,
        y_train,
        validation_data=validation_tuple,
        epochs=150,
        callbacks=[early_stopping],
        batch_size=32
    )

    if has_validation:
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)[:2]
        print(f"\nBest-weights validation accuracy: {val_acc * 100:.2f}% (loss {val_loss:.4f})")

    # Step 8: Save Model & Frozen Class Index Mapping (atomically)
    save_model_and_classes(model, label_encoder.classes_.tolist())

    print(f"\nModel successfully saved to {MODEL_PATH}")
    print(f"Class label mapping ({num_classes} classes) saved to {CLASSES_PATH}")


if __name__ == "__main__":
    main()