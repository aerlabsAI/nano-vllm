#pragma once

#include <cmath>

// ============================================================================
// Normalization Operations
// ============================================================================

namespace Ops {

// RMS Normalization
// Normalizes input using Root Mean Square
inline void rms_norm(float *out, const float *in, const float *weight, int size, float eps = 1e-5f)
{
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += in[i] * in[i];
    }
    float rms = 1.0f / sqrtf(sum / size + eps);
    for (int i = 0; i < size; i++) {
        out[i] = in[i] * rms * weight[i];
    }
}

} // namespace Ops
