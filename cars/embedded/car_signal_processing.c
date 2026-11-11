#include "car_signal_processing.h"

#include <float.h>
#include <math.h>
#include <string.h>

#define PI_F 3.14159265358979323846f
#define DB4_FILTER_LENGTH 8u
#define WPT_LEVEL 4u
#define FFT_SIZE CAR_SIGNAL_WINDOW_SAMPLES
#define FFT_BINS (FFT_SIZE / 2u + 1u)

/*
 * scipy.signal.butter(4, [300, 7000], btype="bandpass", fs=16000,
 *                     output="sos")
 *
 * SOS rows are [b0,b1,b2,a0,a1,a2], with a0 normalized to 1.
 */
static const float butterworth_sos[4][6] = {
    {0.5087922728570657f,  1.0175845457141315f,  0.5087922728570657f,
     1.0f,                  1.3552723175395416f,  0.47175903772333777f},
    {1.0f,                  2.0f,                 1.0f,
     1.0f,                  1.6143892221725957f,  0.748322681570716f},
    {1.0f,                 -2.0f,                 1.0f,
     1.0f,                 -1.7881592401165678f,  0.8011591971830078f},
    {1.0f,                 -2.0f,                 1.0f,
     1.0f,                 -1.9019338754287527f,  0.9152977402371221f},
};

/* PyWavelets db4 decomposition filters, symmetric mode. */
static const float db4_dec_lo[DB4_FILTER_LENGTH] = {
    -0.010597401785069032f, 0.0328830116668852f,
     0.030841381835560764f,-0.18703481171909309f,
    -0.027983769416859854f, 0.6308807679298589f,
     0.7148465705529157f,   0.2303778133088965f,
};

static const float db4_dec_hi[DB4_FILTER_LENGTH] = {
    -0.2303778133088965f,   0.7148465705529157f,
    -0.6308807679298589f,  -0.027983769416859854f,
     0.18703481171909309f,  0.030841381835560764f,
    -0.0328830116668852f,  -0.010597401785069032f,
};

/*
 * PyWavelets get_level(4, order="freq") order, expressed as indices into
 * the natural binary path order (a=0, d=1).
 */
static const uint8_t wpt_frequency_order[16] = {
    0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8,
};

typedef struct {
    float z1[4];
    float z2[4];
} sos_state_t;

static size_t dwt_output_length(size_t input_length)
{
    return (input_length + DB4_FILTER_LENGTH - 1u) / 2u;
}

static size_t symmetric_index(int64_t index, size_t length)
{
    const int64_t n = (int64_t)length;
    while (index < 0 || index >= n) {
        if (index < 0) {
            index = -index - 1;
        } else {
            index = 2 * n - 1 - index;
        }
    }
    return (size_t)index;
}

/* Match PyWavelets' symmetric-mode downsampling convolution. */
static void dwt_symmetric(const float *input,
                          size_t input_length,
                          const float filter[DB4_FILTER_LENGTH],
                          float *output,
                          size_t output_length)
{
    for (size_t out = 0; out < output_length; ++out) {
        const int64_t center = (int64_t)(2u * out + 1u);
        float sum = 0.0f;
        for (size_t tap = 0; tap < DB4_FILTER_LENGTH; ++tap) {
            const int64_t index = center - (int64_t)tap;
            sum += filter[tap] * input[symmetric_index(index, input_length)];
        }
        output[out] = sum;
    }
}

static void wavelet_packet_level4(car_signal_workspace_t *workspace,
                                  const float *input,
                                  float relative_rms[16],
                                  float node_kurtosis[16],
                                  float node_entropy[16])
{
    float *current = workspace->wpt_a;
    float *next = workspace->wpt_b;
    size_t current_length = CAR_SIGNAL_WINDOW_SAMPLES;
    size_t node_count = 1u;

    memcpy(current, input, current_length * sizeof(float));

    for (size_t level = 0; level < WPT_LEVEL; ++level) {
        const size_t next_length = dwt_output_length(current_length);
        for (size_t node = 0; node < node_count; ++node) {
            const float *node_input = current + node * current_length;
            float *low = next + (2u * node) * next_length;
            float *high = next + (2u * node + 1u) * next_length;
            dwt_symmetric(node_input, current_length, db4_dec_lo, low, next_length);
            dwt_symmetric(node_input, current_length, db4_dec_hi, high, next_length);
        }
        {
            float *swap = current;
            current = next;
            next = swap;
        }
        current_length = next_length;
        node_count *= 2u;
    }

    {
        double energies[16];
        double total_energy = 0.0;
        for (size_t frequency_node = 0; frequency_node < 16u; ++frequency_node) {
            const size_t natural_node = wpt_frequency_order[frequency_node];
            const float *coefficients = current + natural_node * current_length;
            double energy = 0.0;
            for (size_t i = 0; i < current_length; ++i) {
                const double value = coefficients[i];
                energy += value * value;
            }
            energies[frequency_node] = energy;
            total_energy += energy;
        }

        for (size_t node = 0; node < 16u; ++node) {
            const size_t natural_node = wpt_frequency_order[node];
            const float *coefficients = current + natural_node * current_length;
            double mean = 0.0;
            double m2 = 0.0;
            double m4 = 0.0;
            const double coefficient_energy = energies[node];

            relative_rms[node] = total_energy > DBL_EPSILON
                ? (float)sqrt(energies[node] / total_energy) : 0.0f;

            for (size_t i = 0; i < current_length; ++i)
                mean += coefficients[i];
            mean /= (double)current_length;

            for (size_t i = 0; i < current_length; ++i) {
                const double centered = (double)coefficients[i] - mean;
                const double square = centered * centered;
                m2 += square;
                m4 += square * square;
            }
            m2 /= (double)current_length;
            m4 /= (double)current_length;

            if (current_length > 3u && m2 > DBL_EPSILON) {
                const double biased_excess = m4 / (m2 * m2) - 3.0;
                const double corrected = ((double)(current_length - 1u) /
                    ((double)(current_length - 2u) * (double)(current_length - 3u))) *
                    ((double)(current_length + 1u) * biased_excess + 6.0);
                node_kurtosis[node] = (float)corrected;
            } else {
                node_kurtosis[node] = 0.0f;
            }

            if (coefficient_energy <= DBL_EPSILON) {
                node_entropy[node] = 0.0f;
            } else {
                double entropy = 0.0;
                for (size_t i = 0; i < current_length; ++i) {
                    const double value = coefficients[i];
                    const double probability = (value * value) / coefficient_energy;
                    if (probability > 0.0)
                        entropy -= probability * (log(probability) / log(2.0));
                }
                node_entropy[node] = (float)entropy;
            }
        }
    }
}

static void fft_radix2(float *real, float *imag, size_t length)
{
    size_t reversed = 0u;
    for (size_t i = 1u; i < length; ++i) {
        size_t bit = length >> 1u;
        while ((reversed & bit) != 0u) {
            reversed ^= bit;
            bit >>= 1u;
        }
        reversed ^= bit;
        if (i < reversed) {
            const float real_swap = real[i];
            const float imag_swap = imag[i];
            real[i] = real[reversed];
            imag[i] = imag[reversed];
            real[reversed] = real_swap;
            imag[reversed] = imag_swap;
        }
    }

    for (size_t block = 2u; block <= length; block <<= 1u) {
        const size_t half = block >> 1u;
        const float angle = -2.0f * PI_F / (float)block;
        const float step_real = cosf(angle);
        const float step_imag = sinf(angle);
        for (size_t base = 0u; base < length; base += block) {
            float twiddle_real = 1.0f;
            float twiddle_imag = 0.0f;
            for (size_t j = 0u; j < half; ++j) {
                const size_t even = base + j;
                const size_t odd = even + half;
                const float odd_real = real[odd] * twiddle_real - imag[odd] * twiddle_imag;
                const float odd_imag = real[odd] * twiddle_imag + imag[odd] * twiddle_real;
                const float even_real = real[even];
                const float even_imag = imag[even];
                real[even] = even_real + odd_real;
                imag[even] = even_imag + odd_imag;
                real[odd] = even_real - odd_real;
                imag[odd] = even_imag - odd_imag;
                {
                    const float next_real = twiddle_real * step_real - twiddle_imag * step_imag;
                    twiddle_imag = twiddle_real * step_imag + twiddle_imag * step_real;
                    twiddle_real = next_real;
                }
            }
        }
    }
}

static void spectral_features(car_signal_workspace_t *workspace,
                              const float *window,
                              float output[5])
{
    double total_power = 0.0;
    double weighted_frequency = 0.0;
    double weighted_frequency_square = 0.0;
    double log_power_sum = 0.0;
    double peak_power = 0.0;
    const double epsilon = DBL_EPSILON;

    for (size_t i = 0; i < FFT_SIZE; ++i) {
        workspace->fft_real[i] = window[i] * workspace->hann[i];
        workspace->fft_imag[i] = 0.0f;
    }
    fft_radix2(workspace->fft_real, workspace->fft_imag, FFT_SIZE);

    for (size_t k = 0; k < FFT_BINS; ++k) {
        const double re = workspace->fft_real[k];
        const double im = workspace->fft_imag[k];
        const double power = re * re + im * im;
        const double frequency = (double)k * (double)CAR_SIGNAL_SAMPLE_RATE_HZ /
                                 (double)FFT_SIZE;
        total_power += power;
        weighted_frequency += frequency * power;
        weighted_frequency_square += frequency * frequency * power;
        log_power_sum += log(power + epsilon);
        if (power > peak_power)
            peak_power = power;
    }

    if (total_power <= epsilon) {
        output[0] = 0.0f;
        output[1] = 0.0f;
        output[2] = 0.0f;
        output[3] = 0.0f;
        output[4] = 0.0f;
        return;
    }

    {
        const double centroid = weighted_frequency / total_power;
        const double variance = weighted_frequency_square / total_power - centroid * centroid;
        const double bandwidth = sqrt(variance > 0.0 ? variance : 0.0);
        const double target = 0.85 * total_power;
        double cumulative = 0.0;
        size_t rolloff_index = FFT_BINS - 1u;
        for (size_t k = 0; k < FFT_BINS; ++k) {
            const double re = workspace->fft_real[k];
            const double im = workspace->fft_imag[k];
            cumulative += re * re + im * im;
            if (cumulative >= target) {
                rolloff_index = k;
                break;
            }
        }
        output[0] = (float)centroid;
        output[1] = (float)bandwidth;
        output[2] = (float)((double)rolloff_index * CAR_SIGNAL_SAMPLE_RATE_HZ / FFT_SIZE);
        output[3] = (float)(exp(log_power_sum / (double)FFT_BINS) /
                            ((total_power / (double)FFT_BINS) + epsilon));
        output[4] = (float)(peak_power / total_power);
    }
}

static void extract_window_features(car_signal_workspace_t *workspace,
                                    const float *window,
                                    float features[CAR_SIGNAL_FEATURE_COUNT])
{
    double mean = 0.0;
    double centered_m2 = 0.0;
    double centered_m3 = 0.0;
    double centered_m4 = 0.0;
    double sum_squares = 0.0;
    double sum_absolute = 0.0;
    double maximum_absolute = 0.0;
    float minimum = window[0];
    float maximum = window[0];
    size_t crossings = 0u;
    const size_t n = CAR_SIGNAL_WINDOW_SAMPLES;

    for (size_t i = 0; i < n; ++i) {
        const double value = window[i];
        const double absolute = fabs(value);
        mean += value;
        sum_squares += value * value;
        sum_absolute += absolute;
        if (absolute > maximum_absolute)
            maximum_absolute = absolute;
        if (window[i] > maximum)
            maximum = window[i];
        if (window[i] < minimum)
            minimum = window[i];
        if (i > 0u && ((window[i] < 0.0f) != (window[i - 1u] < 0.0f)))
            ++crossings;
    }
    mean /= (double)n;

    for (size_t i = 0; i < n; ++i) {
        const double centered = (double)window[i] - mean;
        const double square = centered * centered;
        centered_m2 += square;
        centered_m3 += square * centered;
        centered_m4 += square * square;
    }

    {
        const double m2 = centered_m2 / (double)n;
        const double m3 = centered_m3 / (double)n;
        const double m4 = centered_m4 / (double)n;
        const double rms = sqrt(sum_squares / (double)n);
        const double epsilon = DBL_EPSILON;
        double kurtosis = 0.0;
        double skewness = 0.0;

        if (n > 3u && m2 > epsilon) {
            const double biased_excess = m4 / (m2 * m2) - 3.0;
            kurtosis = ((double)(n - 1u) / ((double)(n - 2u) * (double)(n - 3u))) *
                       ((double)(n + 1u) * biased_excess + 6.0);
            skewness = sqrt((double)n * (double)(n - 1u)) /
                       (double)(n - 2u) * m3 / pow(m2, 1.5);
        }

        /* First 12 temporal features: keep the training column order. */
        features[0] = (float)mean;
        features[1] = (float)sqrt(m2);
        features[2] = maximum;
        features[3] = minimum;
        features[4] = (float)rms;
        features[5] = (float)kurtosis;
        features[6] = (float)skewness;
        features[7] = (float)(maximum_absolute / (rms + epsilon));
        features[8] = (float)sum_squares;
        features[9] = (float)crossings / (float)(n - 1u);
        features[10] = (float)(sum_absolute / (double)n);
        features[11] = maximum - minimum;
    }

    /* Spectral features are indices 12..16. */
    spectral_features(workspace, window, &features[12]);

    /* WPT features are 16 relative RMS, then 16 kurtosis, then 16 entropy. */
    {
        float relative_rms[16];
        float packet_kurtosis[16];
        float packet_entropy[16];
        wavelet_packet_level4(workspace, window, relative_rms,
                              packet_kurtosis, packet_entropy);
        for (size_t i = 0; i < 16u; ++i) {
            features[17u + i] = relative_rms[i];
            features[33u + i] = packet_kurtosis[i];
            features[49u + i] = packet_entropy[i];
        }
    }
}

static void sos_initial_state(float initial[4][2], float input_level)
{
    double level = input_level;
    for (size_t section = 0; section < 4u; ++section) {
        const double b0 = butterworth_sos[section][0];
        const double b1 = butterworth_sos[section][1];
        const double b2 = butterworth_sos[section][2];
        const double a1 = butterworth_sos[section][4];
        const double a2 = butterworth_sos[section][5];
        const double gain = (b0 + b1 + b2) / (1.0 + a1 + a2);
        const double steady_output = level * gain;
        initial[section][0] = (float)(steady_output - b0 * level);
        initial[section][1] = (float)(b2 * level - a2 * steady_output);
        level = steady_output;
    }
}

static void sos_filter_inplace(float *samples, size_t length)
{
    float initial[4][2];
    float z1[4];
    float z2[4];
    sos_initial_state(initial, samples[0]);
    for (size_t section = 0; section < 4u; ++section) {
        z1[section] = initial[section][0];
        z2[section] = initial[section][1];
    }

    for (size_t i = 0; i < length; ++i) {
        float value = samples[i];
        for (size_t section = 0; section < 4u; ++section) {
            const float *sos = butterworth_sos[section];
            const float output = sos[0] * value + z1[section];
            z1[section] = sos[1] * value - sos[4] * output + z2[section];
            z2[section] = sos[2] * value - sos[5] * output;
            value = output;
        }
        samples[i] = value;
    }
}

static void reverse_samples(float *samples, size_t length)
{
    for (size_t i = 0; i < length / 2u; ++i) {
        const size_t opposite = length - 1u - i;
        const float temporary = samples[i];
        samples[i] = samples[opposite];
        samples[opposite] = temporary;
    }
}

static void butterworth_sosfiltfilt(car_signal_workspace_t *workspace,
                                   const int16_t *pcm)
{
    const size_t pad = CAR_SIGNAL_FILTER_PAD_SAMPLES;
    const size_t n = CAR_SIGNAL_CLIP_SAMPLES;
    const size_t extended_length = CAR_SIGNAL_FILTERED_CAPACITY;
    float *signal = workspace->filtered;

    for (size_t i = 0; i < pad; ++i) {
        const size_t source = pad - i;
        signal[i] = 2.0f * ((float)pcm[0] / 32768.0f) -
                    ((float)pcm[source] / 32768.0f);
    }
    for (size_t i = 0; i < n; ++i)
        signal[pad + i] = (float)pcm[i] / 32768.0f;
    for (size_t i = 0; i < pad; ++i) {
        const size_t source = n - 2u - i;
        signal[pad + n + i] = 2.0f * ((float)pcm[n - 1u] / 32768.0f) -
                              ((float)pcm[source] / 32768.0f);
    }

    sos_filter_inplace(signal, extended_length);
    reverse_samples(signal, extended_length);
    sos_filter_inplace(signal, extended_length);
    reverse_samples(signal, extended_length);
    memmove(signal, signal + pad, n * sizeof(float));
}

void car_signal_workspace_init(car_signal_workspace_t *workspace)
{
    if (workspace == NULL)
        return;
    for (size_t i = 0; i < CAR_SIGNAL_WINDOW_SAMPLES; ++i) {
        workspace->hann[i] = 0.5f * (1.0f - cosf(2.0f * PI_F * (float)i /
                                                (float)(CAR_SIGNAL_WINDOW_SAMPLES - 1u)));
    }
}

int car_signal_classify_pcm16_6s(const int16_t *pcm_16khz,
                                 size_t sample_count,
                                 car_signal_workspace_t *workspace,
                                 float probabilities[CAR_RF_CLASS_COUNT])
{
    car_rf_clip_accumulator_t accumulator;
    float features[CAR_SIGNAL_FEATURE_COUNT];
    int result;

    if (pcm_16khz == NULL || workspace == NULL)
        return -1;
    if (sample_count != CAR_SIGNAL_CLIP_SAMPLES)
        return -2;
    if (CAR_RF_FEATURE_COUNT != CAR_SIGNAL_FEATURE_COUNT ||
        CAR_RF_CLASS_COUNT == 0u)
        return -3;
    car_signal_workspace_init(workspace);

    butterworth_sosfiltfilt(workspace, pcm_16khz);
    car_rf_clip_reset(&accumulator);

    for (size_t window = 0; window < CAR_SIGNAL_WINDOWS_PER_CLIP; ++window) {
        const size_t start = window * CAR_SIGNAL_HOP_SAMPLES;
        extract_window_features(workspace, workspace->filtered + start, features);
        if (car_rf_clip_add_window(&accumulator, features) != 0)
            return -4;
    }

    result = car_rf_clip_finish(&accumulator, probabilities);
    return result;
}
