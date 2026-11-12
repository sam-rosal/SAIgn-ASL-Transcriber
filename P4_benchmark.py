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

def load_all_labels():
    """Fetches all unique labels across the entire dataset to ensure exact LabelEncoder mapping."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    data = cursor.execute("SELECT DISTINCT label FROM training_dataset").fetchall()
    conn.close()
    return [row[0] for row in data]

def load_benchmark_data(split_name):
    """Queries the specific dataset split for benchmarking."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    data = cursor.execute("SELECT npy_path, label FROM training_dataset WHERE split = ?", (split_name,)).fetchall()
    conn.close()
    return data

def prepare_tensors(data_list):
    """Loads .npy feature matrices and string labels into NumPy arrays."""
    X, y = [], []
    for npy_path, label in data_list:
        if os.path.exists(npy_path):
            features = np.load(npy_path)
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

def run_benchmark():
    # 1. Mode Selection Menu
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

    # 2. Verification
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Run P4_train_model.py first.")

    # 3. Reconstruct Label Encoder
    print("Synchronizing class mappings from database...")
    all_labels = load_all_labels()
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    class_names = label_encoder.classes_

    # 4. Load Requested Data Split
    print(f"Loading '{target_split}' dataset...")
    benchmark_data = load_benchmark_data(target_split)
    if not benchmark_data:
        print(f"[ERROR] No {target_split} data found in saign_vision.db. Cannot run benchmark.")
        return

    X_test, y_test_str = prepare_tensors(benchmark_data)
    y_true = label_encoder.transform(y_test_str) 

    # 5. Run Model Predictions
    print("Loading AI model into memory...")
    model = load_model(MODEL_PATH)
    
    print(f"Running predictions on {len(X_test)} samples...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1) 

    # 6. Output Classification Report
    print("\n" + "="*60)
    print(f"CLASSIFICATION REPORT: {target_split.upper()} SPLIT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # 7. Generate and Save Confusion Matrix
    print("\nGenerating visual Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(f'ASL Model Benchmark ({target_split.capitalize()} Set): Confusion Matrix', pad=20, fontsize=16)
    plt.ylabel('Actual Sign (Ground Truth)', fontsize=12)
    plt.xlabel('Predicted Sign (Model Output)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(cm_output_path, dpi=300)
    print(f"Confusion matrix heat map saved successfully to: {cm_output_path}")
    plt.show()

if __name__ == "__main__":
    run_benchmark()