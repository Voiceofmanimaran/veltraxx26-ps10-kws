import os
import glob
import pathlib
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models

SAMPLE_RATE = 16000
N_MFCC = 13
N_FFT = 480
HOP_LENGTH = 500
TARGET_FRAMES = 32
EPOCHS = 15
BATCH_SIZE = 64

TARGET_WORDS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]

dataset_dir = pathlib.Path("D:/kws/full_speech_commands")
if not dataset_dir.exists():
    dataset_dir = pathlib.Path("dataset_10_words")

print(f"[INFO] Training on 10 Target Classes: {TARGET_WORDS}")
print(f"[INFO] Dataset Directory: {dataset_dir}")

def extract_mfcc(audio_path):
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    if len(y) < SAMPLE_RATE:
        y = np.pad(y, (0, SAMPLE_RATE - len(y)))
    else:
        y = y[:SAMPLE_RATE]
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH).T
    if mfcc.shape[0] < TARGET_FRAMES:
        mfcc = np.pad(mfcc, ((0, TARGET_FRAMES - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:TARGET_FRAMES, :]
    return ((mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)).astype(np.float32)

# Load dataset
X, y = [], []
for label_idx, word in enumerate(TARGET_WORDS):
    files = list((dataset_dir / word).glob("*.wav"))[:400] # 400 samples per class
    print(f" -> Loading {len(files)} samples for class: '{word}'")
    for f in files:
        X.append(extract_mfcc(str(f)))
        y.append(label_idx)

X = np.array(X)
y = np.array(y)

# Train / Test split
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))
X_train, X_val = X[indices[:split]], X[indices[split:]]
y_train, y_val = y[indices[:split]], y[indices[split:]]

print(f"\n[INFO] Train Shape: {X_train.shape} | Val Shape: {X_val.shape}")

# Define 1D Depthwise Separable CNN Model
def build_1d_ds_cnn(input_shape=(32, 13), num_classes=10):
    inputs = layers.Input(shape=input_shape, name="audio_input")
    
    # Layer 1: Standard 1D Convolution
    x = layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Layer 2: 1D Depthwise Separable Block 1
    x = layers.SeparableConv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Layer 3: 1D Depthwise Separable Block 2
    x = layers.SeparableConv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Output Actuation Layer
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="keyword_actuation")(x)
    
    return models.Model(inputs=inputs, outputs=outputs, name="1D_DS_CNN_EdgeKWS")

model = build_1d_ds_cnn()
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# Save baseline Keras model & labels
os.makedirs("outputs", exist_ok=True)
model.save("outputs/baseline_ds_cnn.keras")
np.save("outputs/ps10_classes.npy", np.array(TARGET_WORDS))

print("\n[SUCCESS] Step 3 Complete: Model trained and saved to outputs/baseline_ds_cnn.keras")
