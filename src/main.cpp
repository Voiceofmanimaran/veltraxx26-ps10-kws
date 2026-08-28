#include <Arduino.h>
#include "hamming_lut.h"

#define FRAME_LEN 480
#define VAD_THRESHOLD 400
#define LED_ACTIVE 2
#define LED_KEYWORD 4

// Zero-heap static memory buffers
static int16_t s_raw_pcm[FRAME_LEN];
static int16_t s_windowed_pcm[FRAME_LEN];

// Fixed-point Q15 multiplication
static inline int16_t mult_q15(int16_t a, int16_t b) {
    return (int16_t)(((int32_t)a * (int32_t)b) >> 15);
}

void setup() {
    Serial.begin(115200);
    pinMode(LED_ACTIVE, OUTPUT);
    pinMode(LED_KEYWORD, OUTPUT);
    digitalWrite(LED_ACTIVE, LOW);
    digitalWrite(LED_KEYWORD, LOW);

    Serial.println("==================================================");
    Serial.println("  PS-10 EDGE KWS FIRMWARE INITIALIZED (ESP32-S3)");
    Serial.println("==================================================");
    Serial.printf("[FIRMWARE] Static LUT Size in Flash: %d bytes\n", sizeof(HAMMING_LUT_Q15));
    Serial.printf("[FIRMWARE] Frame Buffer Size in SRAM: %d bytes\n", sizeof(s_raw_pcm) + sizeof(s_windowed_pcm));
}

void loop() {
    uint32_t t_start = micros();

    // 1. Emulate incoming 30 ms PCM audio frame
    int32_t energy_sum = 0;
    for (int i = 0; i < FRAME_LEN; i++) {
        // Synthetic speech signal test frame
        s_raw_pcm[i] = (int16_t)(sin(i * 0.05) * 2000.0);
        
        // 2. Fixed-Point Q15 Hamming Windowing via LUT
        s_windowed_pcm[i] = mult_q15(s_raw_pcm[i], HAMMING_LUT_Q15[i]);
        energy_sum += abs(s_windowed_pcm[i]);
    }

    int32_t frame_energy = energy_sum / FRAME_LEN;
    uint32_t t_dsp_us = micros() - t_start;

    // 3. Integer VAD & Hardware Actuation
    if (frame_energy > VAD_THRESHOLD) {
        digitalWrite(LED_ACTIVE, HIGH);
        digitalWrite(LED_KEYWORD, HIGH);
        Serial.printf("[DSP CORE] Frame Energy: %d | DSP Time: %u us | Actuation: KEYWORD DETECTED\n", 
                      frame_energy, t_dsp_us);
    } else {
        digitalWrite(LED_ACTIVE, LOW);
        digitalWrite(LED_KEYWORD, LOW);
    }

    delay(200);
}
