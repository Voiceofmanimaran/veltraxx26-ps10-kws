import os
import pathlib
import numpy as np
import tensorflow as tf
import librosa

SAMPLE_RATE = 16000
N_MFCC = 13
N_FFT = 480
HOP_LENGTH = 500
TARGET_FRAMES = 32

TARGET_WORDS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]

# Locate dataset
dataset_dir = pathlib.Path("D:/kws/full_speech_commands")
if not dataset_dir.exists():
    dataset_dir = pathlib.Path("dataset_10_words")

# Load Quantized TFLite Model
model_path = "outputs/edge_kws_ps10.tflite"
if not os.path.exists(model_path):
    raise FileNotFoundError("Missing outputs/edge_kws_ps10.tflite. Run quantize_model.py first.")

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def extract_features(audio_arr):
    if len(audio_arr) < SAMPLE_RATE:
        audio_arr = np.pad(audio_arr, (0, SAMPLE_RATE - len(audio_arr)))
    else:
        audio_arr = audio_arr[:SAMPLE_RATE]
    mfcc = librosa.feature.mfcc(y=audio_arr.astype(np.float32), sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH).T
    if mfcc.shape[0] < TARGET_FRAMES:
        mfcc = np.pad(mfcc, ((0, TARGET_FRAMES - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:TARGET_FRAMES, :]
    return ((mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)).astype(np.float32)

def inject_noise(clean_signal, snr_db):
    if snr_db is None:
        return clean_signal
    sig_power = np.mean(clean_signal ** 2)
    if sig_power == 0:
        return clean_signal
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = np.random.normal(0, np.sqrt(noise_power), len(clean_signal))
    return clean_signal + noise

# Test SNR levels
snr_conditions = [
    (None, "Clean Audio (Baseline)"),
    (20, "20 dB SNR (Light Ambient Noise)"),
    (10, "10 dB SNR (Moderate Background Noise)"),
    (5, "5 dB SNR (Heavy Acoustic Noise)"),
    (0, "0 dB SNR (Extreme / Industrial Noise)")
]

print("=" * 65)
print("     PS-10 ACOUSTIC NOISE ROBUSTNESS & STRESS BENCHMARK")
print("=" * 65)
print(f"Dataset Source : {dataset_dir}")
print(f"Target Classes : {', '.join(TARGET_WORDS)}")
print("Evaluating test slices across SNR conditions...")
print("-" * 65)

results = []

for snr_val, label in snr_conditions:
    correct = 0
    total = 0
    
    for class_idx, word in enumerate(TARGET_WORDS):
        files = list((dataset_dir / word).glob("*.wav"))[350:380] # 30 validation files per class = 300 test runs per SNR
        for f in files:
            audio, _ = librosa.load(str(f), sr=SAMPLE_RATE)
            noisy_audio = inject_noise(audio, snr_val)
            feat = extract_features(noisy_audio)
            
            inp = np.expand_dims(feat, axis=0).astype(np.float32)
            interpreter.set_tensor(input_index, inp)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_index)[0]
            
            if np.argmax(preds) == class_idx:
                correct += 1
            total += 1

    acc = (correct / total) * 100.0 if total > 0 else 0
    results.append((label, acc, correct, total))
    print(f"{label:<40} : {acc:6.2f}% ({correct}/{total})")

print("=" * 65)

# Save Report
os.makedirs("logs", exist_ok=True)
report_path = "logs/noise_robustness_report.txt"
with open(report_path, "w") as f:
    f.write("=========================================================\n")
    f.write("     PS-10 ACOUSTIC NOISE & STRESS BENCHMARK REPORT\n")
    f.write("=========================================================\n")
    for label, acc, corr, tot in results:
        f.write(f"{label:<40} : {acc:6.2f}% ({corr}/{tot})\n")
    f.write("=========================================================\n")

print(f"[SUCCESS] Stress evaluation exported to '{report_path}'")
