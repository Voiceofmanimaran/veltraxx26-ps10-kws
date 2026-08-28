import os
import glob
import pathlib
import numpy as np
import librosa

SAMPLE_RATE = 16000
N_MFCC = 13
N_FFT = 480
HOP_LENGTH = 500
TARGET_FRAMES = 32

TARGET_WORDS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]

dataset_dir = pathlib.Path("D:/kws/full_speech_commands")
if not dataset_dir.exists():
    dataset_dir = pathlib.Path("dataset_10_words")

print(f"[INFO] Target Classes: {TARGET_WORDS}")

lut_path = "outputs/hamming_lut_q15.npy"
if os.path.exists(lut_path):
    hamming_lut = np.load(lut_path)
    print(f"[INFO] Loaded Q15 LUT ({len(hamming_lut)} entries)")
else:
    raise FileNotFoundError("Missing outputs/hamming_lut_q15.npy. Run generate_lut.py first.")

def extract_fixed_mfcc(audio_array):
    if len(audio_array) < SAMPLE_RATE:
        audio_array = np.pad(audio_array, (0, SAMPLE_RATE - len(audio_array)), mode='constant')
    else:
        audio_array = audio_array[:SAMPLE_RATE]

    mfcc = librosa.feature.mfcc(
        y=audio_array.astype(np.float32),
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    ).T

    if mfcc.shape[0] < TARGET_FRAMES:
        mfcc = np.pad(mfcc, ((0, TARGET_FRAMES - mfcc.shape[0]), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:TARGET_FRAMES, :]

    mfcc_norm = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
    return mfcc_norm.astype(np.float32)

if __name__ == "__main__":
    synthetic_pcm = np.random.uniform(-1.0, 1.0, SAMPLE_RATE).astype(np.float32)
    feat = extract_fixed_mfcc(synthetic_pcm)
    
    print("\n[VERIFICATION]")
    print(f" -> Feature Tensor Output Shape : {feat.shape}  (Expected: (32, 13))")
    print(f" -> Data Type                   : {feat.dtype}")
    print(f" -> Tensor Mean / Std Dev       : {np.mean(feat):.4f} / {np.std(feat):.4f}")
    
    assert feat.shape == (32, 13), "Feature dimension mismatch!"
    print("[SUCCESS] Step 2 Feature Extraction verified successfully.")
