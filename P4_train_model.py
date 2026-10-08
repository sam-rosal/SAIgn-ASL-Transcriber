import sqlite3
import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Masking, LSTM, Dropout, Dense, Attention, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint

# --- System Configuration & File Paths ---
DB_PATH = "saign_vision.db"         # SQLite database containing frame feature metadata
MODEL_SAVE_PATH = "asl_model.keras"  # Output path for the trained model artifact
EPOCHS = 50                          # Maximum training iterations
BATCH_SIZE = 16                      # Batch size for gradient descent updates
TARGET_FRAMES = 30                   # Fixed temporal sequence length required for input tensor shape


def resample_sequence(features, target_frames=TARGET_FRAMES):
    """
    Safeguard function: Resamples any keypoint array to (target_frames, 126)
    using 1D linear temporal interpolation if it was extracted with non-standard frame lengths.
    
    Mathematical Pipeline:
    1. Map original sequence timeline to normalized range [0.0, 1.0].
    2. Interpolate across the temporal axis to generate exactly target_frames timesteps.
    """
    current_frames = len(features)
    if current_frames == target_frames:
        return features

    # Fallback for sequences with fewer than 2 frames
    if current_frames < 2:
        pad_amount = max(0, target_frames - current_frames)
        return np.pad(features, ((0, pad_amount), (0, 0)), mode='edge')[:target_frames]

    # Map original and target time steps to normalized range [0.0, 1.0]
    x_old = np.linspace(0, 1, num=current_frames)
    x_new = np.linspace(0, 1, num=target_frames)
    
    # Perform 1D linear interpolation across temporal axis (axis=0)
    interpolator = interp1d(x_old, features, axis=0, kind='linear')
    return interpolator(x_new)


def load_split_from_sqlite(split_name):
    """Loads features and labels from SQLite filtered by dataset split, enforcing 30-frame sequence shape."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT npy_path, label FROM training_dataset WHERE split = ?", (split_name,))
    rows = cursor.fetchall()
    conn.close()

    X, y = [], []
    for npy_path, label in rows:
        try:
            raw_data = np.load(npy_path)
            # Enforce strict (30, 126) input tensor shape via temporal resampling guardrail
            standardized_data = resample_sequence(raw_data, target_frames=TARGET_FRAMES)
            X.append(standardized_data)
            y.append(label)
        except Exception as e:
            print(f"Warning: Missing file or corrupt binary {npy_path}: {e}")

    return np.array(X), np.array(y)


def build_lstm_attention_model(input_shape, num_classes):
    """
    Constructs a Functional API Neural Network with stacked LSTM layers, 
    a Temporal Self-Attention Mechanism, and Global Pooling.
    """
    # Input Layer: shape expected as (sequence_length=30, feature_dimensions)
    inputs = Input(shape=input_shape, name="landmark_input")
    
    # Masking Layer: Ignores zero-padded frames added during preprocessing
    masked = Masking(mask_value=0.0)(inputs)
    
    # First LSTM Layer: return_sequences=True retains temporal sequence outputs for attention
    lstm1 = LSTM(64, return_sequences=True, activation='tanh')(masked)
    lstm1 = Dropout(0.2)(lstm1)  # Regularization to prevent overfitting
    
    # Second LSTM Layer: Preserves full sequence dimensions (30, 64) required by the Attention layer
    lstm2 = LSTM(64, return_sequences=True, activation='tanh')(lstm1)
    lstm2 = Dropout(0.2)(lstm2)
    
    # Self-Attention Mechanism: Calculates query-key context scores across all 30 frames 
    # to weight peak gesture frames higher than start/end transition frames
    attn_out = Attention(name="attention_layer")([lstm2, lstm2])
    
    # Global Average Pooling: Flattens temporal sequence outputs (30, 64) -> (64,) vector
    pooled = GlobalAveragePooling1D()(attn_out)
    
    # Dense Feature Mapping Layer
    dense1 = Dense(32, activation='relu')(pooled)
    
    # Final Classification Output Layer: Probabilities across all ASL classes
    outputs = Dense(num_classes, activation='softmax')(dense1)
    
    # Build Keras Functional Model
    model = Model(inputs=inputs, outputs=outputs, name="SAIgn_LSTM_Attention")
    
    # Compile model using categorical cross-entropy loss for multi-class classification
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train():
    print("Loading MS-ASL partitions from SQLite...")
    X_train_raw, y_train_raw = load_split_from_sqlite("train")
    X_val_raw, y_val_raw     = load_split_from_sqlite("val")
    X_test_raw, y_test_raw   = load_split_from_sqlite("test")

    # Fit LabelEncoder on training set targets
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train_raw)
    
    # Transform validation and test targets using fitted encoder
    y_val_enc  = label_encoder.transform(y_val_raw)
    y_test_enc = label_encoder.transform(y_test_raw)

    num_classes = len(label_encoder.classes_)
    np.save("classes.npy", label_encoder.classes_)

    # Convert to One-Hot Encoding
    y_train = to_categorical(y_train_enc, num_classes=num_classes)
    y_val   = to_categorical(y_val_enc,   num_classes=num_classes)
    y_test  = to_categorical(y_test_enc,  num_classes=num_classes)

    print(f"Data Split -> Train: {len(X_train_raw)} | Val: {len(X_val_raw)} | Test: {len(X_test_raw)}")

    # Build model using training shape
    input_shape = (X_train_raw.shape[1], X_train_raw.shape[2])
    model = build_lstm_attention_model(input_shape, num_classes)

########################### Early Stopping ####################################
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    model.fit(
        X_train_raw, y_train,
        validation_data=(X_val_raw, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    print("\nEvaluating model on official MS-ASL Test Set...")
    test_loss, test_acc = model.evaluate(X_test_raw, y_test)
    print(f"Official Test Accuracy: {test_acc * 100:.2f}%")

    model.save(MODEL_SAVE_PATH)


if __name__ == "__main__":
    train()