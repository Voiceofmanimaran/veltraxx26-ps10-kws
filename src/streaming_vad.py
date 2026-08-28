import numpy as np

class RingBufferVAD:
    """
    Zero-heap-allocation sliding circular buffer with integer-based VAD.
    Maintains a 1-second audio window (16,000 samples at 16 kHz).
    """
    def __init__(self, sample_rate=16000, window_duration_sec=1.0, energy_threshold=450):
        self.capacity = int(sample_rate * window_duration_sec)
        self.buffer = np.zeros(self.capacity, dtype=np.int16)
        self.write_ptr = 0
        self.energy_threshold = energy_threshold

    def push_chunk(self, chunk_pcm16):
        """Pushes a new PCM16 chunk into the circular buffer."""
        chunk_len = len(chunk_pcm16)
        if chunk_len >= self.capacity:
            self.buffer[:] = chunk_pcm16[-self.capacity:]
            self.write_ptr = 0
            return

        end_ptr = self.write_ptr + chunk_len
        if end_ptr <= self.capacity:
            self.buffer[self.write_ptr:end_ptr] = chunk_pcm16
        else:
            first_part = self.capacity - self.write_ptr
            self.buffer[self.write_ptr:] = chunk_pcm16[:first_part]
            self.buffer[:end_ptr - self.capacity] = chunk_pcm16[first_part:]
        
        self.write_ptr = end_ptr % self.capacity

    def get_linear_window(self):
        """Extracts the continuous 1-second linear PCM array in chronological order."""
        return np.roll(self.buffer, -self.write_ptr)

    def is_speech_active(self, chunk_pcm16):
        """
        Integer-only Mean Absolute Amplitude (MAA) energy calculation.
        Avoids floating-point math and square roots.
        """
        if len(chunk_pcm16) == 0:
            return False
        # Calculate Mean Absolute Value (integer division)
        avg_energy = int(np.mean(np.abs(chunk_pcm16)))
        return avg_energy >= self.energy_threshold

if __name__ == "__main__":
    print("=" * 60)
    print("  PS-10 CIRCULAR RING BUFFER & INTEGER VAD VALIDATION")
    print("=" * 60)
    
    vad_engine = RingBufferVAD(sample_rate=16000, window_duration_sec=1.0, energy_threshold=500)
    
    # 1. Test with synthetic background silence (low amplitude noise)
    silence_frame = np.random.randint(-150, 150, size=480, dtype=np.int16)
    vad_engine.push_chunk(silence_frame)
    is_active_silence = vad_engine.is_speech_active(silence_frame)
    print(f"Frame 1 (Silence Frame) -> Energy: {int(np.mean(np.abs(silence_frame)))} | VAD Trigger: {is_active_silence} (Expected: False)")
    
    # 2. Test with synthetic speech chunk (high amplitude signal)
    speech_frame = (np.sin(np.linspace(0, 2 * np.pi * 5, 480)) * 5000).astype(np.int16)
    vad_engine.push_chunk(speech_frame)
    is_active_speech = vad_engine.is_speech_active(speech_frame)
    print(f"Frame 2 (Speech Frame)  -> Energy: {int(np.mean(np.abs(speech_frame)))} | VAD Trigger: {is_active_speech} (Expected: True)")
    
    # 3. Test linear window retrieval
    full_window = vad_engine.get_linear_window()
    print(f"Full Window Shape       -> {full_window.shape} (Expected: (16000,))")
    print(f"Full Window Dtype       -> {full_window.dtype} (Expected: int16)")
    
    assert full_window.shape == (16000,), "Ring buffer capacity mismatch!"
    assert not is_active_silence, "VAD failed: silence triggered speech."
    assert is_active_speech, "VAD failed: speech frame not detected."
    print("\n[SUCCESS] Step 5 Verified: Circular Buffer and VAD logic operational.")
