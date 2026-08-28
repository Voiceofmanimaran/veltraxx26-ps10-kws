import os
import time
import numpy as np
import tensorflow as tf
import librosa

NUM_RUNS = 50
SAMPLE_RATE = 16000
LATENCY_CEILING_MS = 50.0

TARGET_WORDS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]

# Load Quantized Model
model_path = "outputs/edge_kws_ps10.tflite"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Missing {model_path}. Run quantize_model.py first.")

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# Warm-up run: Compiles Librosa JIT cache & initializes TFLite XNNPACK kernels
dummy_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
_ = librosa.feature.mfcc(y=dummy_audio, sr=SAMPLE_RATE, n_mfcc=13, n_fft=480, hop_length=500)
dummy_input = np.zeros((1, 32, 13), dtype=np.float32)
interpreter.set_tensor(input_index, dummy_input)
interpreter.invoke()

t_feat_list = []
t_infer_list = []
t_total_list = []

print("=" * 60)
print("     PS-10 LATENCY & TIMING PROFILING SUITE")
print("=" * 60)
print(f"Target Vocabulary   : {', '.join(TARGET_WORDS)}")
print(f"Benchmark Runs      : {NUM_RUNS} iterations")
print(f"Latency Constraint  : <= {LATENCY_CEILING_MS} ms")
print("Profiling steady-state execution in progress...")

for i in range(NUM_RUNS):
    synthetic_audio = np.random.uniform(-1.0, 1.0, SAMPLE_RATE).astype(np.float32)

    # 1. Measure Feature Extraction Latency
    t0 = time.perf_counter()
    mfcc = librosa.feature.mfcc(y=synthetic_audio, sr=SAMPLE_RATE, n_mfcc=13, n_fft=480, hop_length=500).T
    if mfcc.shape[0] < 32:
        mfcc = np.pad(mfcc, ((0, 32 - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:32, :]
    mfcc_norm = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
    t_feat = (time.perf_counter() - t0) * 1000.0

    # 2. Measure INT8 Inference Latency
    t1 = time.perf_counter()
    inp = np.expand_dims(mfcc_norm, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, inp)
    interpreter.invoke()
    _ = interpreter.get_tensor(output_index)[0]
    t_infer = (time.perf_counter() - t1) * 1000.0

    t_feat_list.append(t_feat)
    t_infer_list.append(t_infer)
    t_total_list.append(t_feat + t_infer)

avg_feat = np.mean(t_feat_list)
avg_infer = np.mean(t_infer_list)
avg_total = np.mean(t_total_list)
p100_total = np.max(t_total_list)
p95_total = np.percentile(t_total_list, 95)

status = "PASSED [OPTIMAL]" if p100_total <= LATENCY_CEILING_MS else "FAILED"

report = f"""=========================================================
      PS-10 LATENCY & TIMING PROFILING REPORT
=========================================================
Target Classes                   : {', '.join(TARGET_WORDS)}
Benchmark Frames Evaluated       : {NUM_RUNS}
Model Architecture               : 1D Depthwise Separable CNN
Arithmetic Mode                  : Integer Quantized (INT8)
---------------------------------------------------------
Avg Feature Extraction (LUT/MFCC): {avg_feat:.2f} ms
Avg 1D DS-CNN INT8 Inference     : {avg_infer:.2f} ms
Avg End-to-End Execution Latency : {avg_total:.2f} ms
95th Percentile Latency (P95)    : {p95_total:.2f} ms
Worst-Case Peak Latency (P100)   : {p100_total:.2f} ms
PS-10 Latency Ceiling Constraint : <= {LATENCY_CEILING_MS:.2f} ms
Status                           : {status}
=========================================================
"""

print("\n" + report)

os.makedirs("logs", exist_ok=True)
with open("logs/latency_profile.txt", "w") as f:
    f.write(report)

print("[SUCCESS] Benchmark report exported to 'logs/latency_profile.txt'")
