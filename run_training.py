import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import video_utils
import train_model

# --- GTX 1650 / RTX 3050 Settings ---
DATASET_DIR = "dataset"
CLASSES = ["real", "fake"]
IMG_SIZE = 224      # Full HD-ish quality
MAX_SEQ_LENGTH = 20 # 20 Frames per video
BATCH_SIZE = 4      # Optimized for 4GB VRAM
EPOCHS = 20

# --- Generator (Standard) ---
class VideoDataGenerator(Sequence):
    def __init__(self, paths, labels, batch_size):
        self.paths, self.labels, self.batch_size = paths, labels, batch_size
    def __len__(self): return int(np.ceil(len(self.paths) / self.batch_size))
    def __getitem__(self, idx):
        batch_paths = self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = []
        for p in batch_paths:
            v = video_utils.load_video(p, MAX_SEQ_LENGTH, (IMG_SIZE, IMG_SIZE))
            batch_x.append(v/255.0 if v is not None else np.zeros((MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3)))
        return np.array(batch_x), np.array(batch_labels)

# --- Main Execution ---
if __name__ == "__main__":
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus: print(f"✅ GPU Detected: {gpus}")
    else: print("⚠️ Warning: No GPU detected. Training will be slow.")

    # Load Paths
    all_paths, all_labels = [], []
    for i, cls in enumerate(CLASSES):
        d = os.path.join(DATASET_DIR, cls)
        files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(('.mp4', '.avi'))]
        all_paths.extend(files)
        all_labels.extend([i]*len(files))

    # Split
    X_train, X_val, y_train, y_val = train_test_split(all_paths, all_labels, test_size=0.2, random_state=42)
    train_gen = VideoDataGenerator(X_train, y_train, BATCH_SIZE)
    val_gen = VideoDataGenerator(X_val, y_val, BATCH_SIZE)

    # Build
    print("Building EfficientNet-LSTM Model...")
    model = train_model.build_convlstm_model(MAX_SEQ_LENGTH, IMG_SIZE, len(CLASSES))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # Low learning rate for fine-tuning
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint('final_model.h5', save_best_only=True, monitor='val_loss', verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]

    # Train
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)
    print("🎉 Training Complete! Model saved as 'final_model.h5'")