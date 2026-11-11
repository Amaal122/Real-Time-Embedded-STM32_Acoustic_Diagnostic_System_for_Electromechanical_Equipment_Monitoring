#ifndef CAR_SIGNAL_PROCESSING_H
#define CAR_SIGNAL_PROCESSING_H

#include <stddef.h>
#include <stdint.h>

#include "global_random_forest_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CAR_SIGNAL_SAMPLE_RATE_HZ       16000u
#define CAR_SIGNAL_CLIP_SECONDS         6u
#define CAR_SIGNAL_WINDOW_SAMPLES       16000u
#define CAR_SIGNAL_HOP_SAMPLES          8000u
#define CAR_SIGNAL_WINDOWS_PER_CLIP     11u
#define CAR_SIGNAL_FEATURE_COUNT        65u

#define CAR_SIGNAL_CLIP_SAMPLES \
    (CAR_SIGNAL_SAMPLE_RATE_HZ * CAR_SIGNAL_CLIP_SECONDS)

#define CAR_SIGNAL_FILTER_PAD_SAMPLES   27u
#define CAR_SIGNAL_FILTERED_CAPACITY \
    (CAR_SIGNAL_CLIP_SAMPLES + 2u * CAR_SIGNAL_FILTER_PAD_SAMPLES)

#define CAR_SIGNAL_WPT_MAX_COEFFICIENTS 16096u

/*
 * Working memory is about 690 KiB. Put this object in the board's external
 * SDRAM (after FMC/SDRAM initialization), not on the task stack or small SRAM.
 */
typedef struct {
    float filtered[CAR_SIGNAL_FILTERED_CAPACITY];
    float wpt_a[CAR_SIGNAL_WPT_MAX_COEFFICIENTS];
    float wpt_b[CAR_SIGNAL_WPT_MAX_COEFFICIENTS];
    float fft_real[CAR_SIGNAL_WINDOW_SAMPLES];
    float fft_imag[CAR_SIGNAL_WINDOW_SAMPLES];
    float hann[CAR_SIGNAL_WINDOW_SAMPLES];
} car_signal_workspace_t;

void car_signal_workspace_init(car_signal_workspace_t *workspace);

/*
 * Process exactly 6 seconds of signed PCM16 audio sampled at 16 kHz.
 * The function applies the training Butterworth filter, computes the 65
 * features for 11 one-second windows at 50% overlap, averages features over
 * the clip, then calls the global Random Forest classifier.
 *
 * Returns the classifier's class index, or a negative error code. The
 * optional probabilities array receives one score per CAR_RF_CLASS_COUNT.
 */
int car_signal_classify_pcm16_6s(
    const int16_t *pcm_16khz,
    size_t sample_count,
    car_signal_workspace_t *workspace,
    float probabilities[CAR_RF_CLASS_COUNT]);

#ifdef __cplusplus
}
#endif

#endif /* CAR_SIGNAL_PROCESSING_H */
