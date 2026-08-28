import numpy as np
import os

FRAME_SIZE = 480

# 1. Precompute 480-point standard Hamming Window coefficients in float
hamming_float = np.hamming(FRAME_SIZE)

# 2. Scale to Signed Q15 Fixed-Point integer range [-32768, 32767]
hamming_q15 = np.round(hamming_float * 32767).astype(np.int16)

os.makedirs("outputs", exist_ok=True)
os.makedirs("src", exist_ok=True)

# 3. Export Python-accessible binary NumPy array
np.save("outputs/hamming_lut_q15.npy", hamming_q15)

# 4. Export pure C/C++ Header file for embedded deployment
header_path = "src/hamming_lut.h"
with open(header_path, "w") as f:
    f.write("// ====================================================================\n")
    f.write("//  Q15 Fixed-Point Hamming Window Look-Up Table (PS-10 Edge KWS)\n")
    f.write(f"//  Frame Size: {FRAME_SIZE} samples (30 ms at 16 kHz)\n")
    f.write("// ====================================================================\n\n")
    f.write("#ifndef HAMMING_LUT_H\n")
    f.write("#define HAMMING_LUT_H\n\n")
    f.write("#include <stdint.h>\n\n")
    f.write(f"static const int16_t HAMMING_LUT_Q15[{FRAME_SIZE}] = {{\n    ")
    
    for i, val in enumerate(hamming_q15):
        f.write(f"{val}, ")
        if (i + 1) % 12 == 0 and (i + 1) != FRAME_SIZE:
            f.write("\n    ")
            
    f.write("\n};\n\n")
    f.write("#endif // HAMMING_LUT_H\n")

print(f"[SUCCESS] Q15 Hamming LUT generated with {FRAME_SIZE} entries.")
print(f" -> Binary LUT: outputs/hamming_lut_q15.npy")
print(f" -> Embedded C Header: {header_path}")
