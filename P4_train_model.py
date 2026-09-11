import os
import sqlite3
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical

# ==========================================
# 1. PATH RESOLUTION & CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, 'saign_vision.db')
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, 'asl_model.keras')

# Fixed sequence length (pads shorter clips, truncates longer clips)
MAX_SEQUENCE_LENGTH = 30  # Standard frame count for sign gesture clips
BATCH_SIZE = 16
EPOCHS = 50

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def pad_or_truncate_sequence(seq, target_len=MAX_SEQUENCE_LENGTH):
    """Ensures all video feature matrices have identical frame dimensions."""
    num_frames, num_features = seq.shape
    if num_frames == target_len:
        return seq
    elif num_frames > target_len:
        # Truncate extra frames
        return seq[:target_len, :]
    else:
        # Zero-pad missing frames at the end
        padding = np.zeros((target_len - num_frames, num_features))
        return np.vstack([seq, padding])

def load_data_from_sqlite():
    """Queries SQLite for training paths and loads landmark matrices."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}. Run P4_setup_db.py first.")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT extracted_features_path, label FROM training_dataset")
    records = cursor.fetchall()
    conn.close()

    if not records:
        raise ValueError("No records found in training_dataset table. Run P4_extract_features.py first.")

    X, y = [], []
    for feat_path, label in records:
        full_path = os.path.join(SCRIPT_DIR, feat_path) if not os.path.isabs(feat_path) else feat_path
        
        if os.path.exists(full_path):
            data = np.load(full_path)
            if data.size == 0:
                continue
            
            # Ensure sequence length uniformity
            padded_data = pad_or_truncate_sequence(data)
            X.append(padded_data)
            y.append(label)
        else:
            print(f"Warning: Missing feature file at {full_path}")

    return np.array(X), np.array(y)

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
def build_lstm_model(input_shape, num_classes):
    """Builds a stacked LSTM network optimized for sequence landmark data."""
    model = Sequential([
        # Ignore zero-padded frames during gradient updates
        Masking(mask_value=0.0, input_shape=input_shape),
        
        LSTM(64, return_sequences=True, activation='tanh'),
        Dropout(0.2),
        
        LSTM(64, return_sequences=False, activation='tanh'),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ==========================================
# 4. MAIN TRAINING PIPELINE
# ==========================================
def train():
    print("Fetching training dataset from SQLite...")
    X, y_raw = load_data_from_sqlite()

    print(f"Loaded {len(X)} samples.")
    print(f"Input Shape per sample: {X.shape[1]} frames x {X.shape[2]} features")

    # Encode string labels ('A', 'HELLO') into integers and one-hot vectors
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    y_one_hot = to_categorical(y_encoded)

    num_classes = len(label_encoder.classes_)
    print(f"Detected Classes ({num_classes}): {list(label_encoder.classes_)}")

    # Check if dataset samples are too small for stratified splitting
    min_class_samples = np.min(np.bincount(y_encoded))

    if min_class_samples < 2:
        print("\n[INFO] Single video per class detected. Skipping dataset split for dry-run testing.")
        X_train, y_train = X, y_one_hot
        X_val, y_val = X, y_one_hot
    else:
        # Split dataset (80% Train / 20% Validation)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_one_hot, test_size=0.2, random_state=42, stratify=y_encoded
        )

    # Build network
    input_shape = (X.shape[1], X.shape[2])
    model = build_lstm_model(input_shape, num_classes)
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    # Train
    print("\nStarting model training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    # Save final model state
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel training complete! Saved to: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train()