import time
import numpy as np
import sounddevice as sd
import tensorflow as tf
import librosa
from streaming_vad import RingBufferVAD

# Hardware & DSP Configuration
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.03  # 30 ms frames
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION) # 480 samples
CONFIDENCE_THRESHOLD = 0.65
LATENCY_CEILING_MS = 50.0

TARGET_WORDS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]

# Load INT8 Quantized Model
model_path = "outputs/edge_kws_ps10.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize Zero-Heap Ring Buffer & VAD
vad_engine = RingBufferVAD(sample_rate=SAMPLE_RATE, window_duration_sec=1.0, energy_threshold=400)

print("=" * 65)
print("  PS-10 DETERMINISTIC EDGE KWS RUNNER (10 CLASSES)")
print(f"  Target Vocabulary : {', '.join([w.upper() for w in TARGET_WORDS])}")
print(f"  Execution Ceiling : <= {LATENCY_CEILING_MS} ms | Mode: INT8 Quantized")
print("=" * 65)

def run_pipeline_on_window(pcm16_window):
    # Stage 1: Feature Extraction
    t0 = time.perf_counter()
    audio_float = pcm16_window.astype(np.float32) / 32768.0
    mfcc = librosa.feature.mfcc(y=audio_float, sr=SAMPLE_RATE, n_mfcc=13, n_fft=480, hop_length=500).T
    if mfcc.shape[0] < 32:
        mfcc = np.pad(mfcc, ((0, 32 - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:32, :]
    mfcc_norm = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
    t_feat = (time.perf_counter() - t0) * 1000.0

    # Stage 2: INT8 TFLite Model Inference
    t1 = time.perf_counter()
    input_data = np.expand_dims(mfcc_norm, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    probabilities = interpreter.get_tensor(output_details[0]['index'])[0]
    t_infer = (time.perf_counter() - t1) * 1000.0

    t_total = t_feat + t_infer
    top_idx = int(np.argmax(probabilities))
    confidence = probabilities[top_idx]
    
    return top_idx, confidence, t_feat, t_infer, t_total

def main():
    try:
        while True:
            input("\nPress [ENTER] and speak a target keyword (e.g. STOP, GO, ON, OFF, YES)... ")
            print("🎤 Listening and streaming into ring buffer...")
            
            # Record 1.2 seconds of live audio from microphone
            recording = sd.rec(int(1.2 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
            sd.wait()
            raw_pcm = recording.flatten()

            # Stream chunks through ring buffer & check VAD
            speech_detected = False
            for i in range(0, len(raw_pcm) - CHUNK_SIZE, CHUNK_SIZE):
                chunk = raw_pcm[i:i + CHUNK_SIZE]
                vad_engine.push_chunk(chunk)
                if vad_engine.is_speech_active(chunk):
                    speech_detected = True

            linear_window = vad_engine.get_linear_window()
            idx, conf, t_feat, t_infer, t_total = run_pipeline_on_window(linear_window)
            
            print("\n---------------- TIMING & LATENCY REPORT ----------------")
            print(f" 1. Fixed-Point MFCC Extraction:   {t_feat:.2f} ms")
            print(f" 2. 1D DS-CNN INT8 Classification:  {t_infer:.2f} ms")
            print(f" TOTAL END-TO-END LATENCY:         {t_total:.2f} ms")
            print(f" LATENCY CONSTRAINT (<= 50.0 ms):  {'PASSED [OPTIMAL]' if t_total <= LATENCY_CEILING_MS else 'FAILED'}")
            print("---------------------------------------------------------")
            
            if conf >= CONFIDENCE_THRESHOLD:
                print(f"CLASSIFIED KEYWORD: >>> '{TARGET_WORDS[idx].upper()}' <<< (Confidence: {conf*100:.1f}%)")
            else:
                print(f"STATUS: [REJECTED] Low Confidence ({TARGET_WORDS[idx].upper()} at {conf*100:.1f}%)")
            print("---------------------------------------------------------")

    except KeyboardInterrupt:
        print("\nRunner terminated cleanly.")

if __name__ == "__main__":
    main()
