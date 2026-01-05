#pragma once

#include <cmath>

// ============================================================================
// Activation Functions
// ============================================================================

namespace Ops {

// Softmax Activation
// Converts logits to probabilities
inline void softmax(float *x, int size)
{
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val)
            max_val = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// SwiGLU Activation
// Combination of Swish and Gated Linear Unit
inline void swiglu(float *hb, const float *h_gate, const float *h_up, int hidden_dim)
{
    for (int i = 0; i < hidden_dim; i++) {
        float val = h_gate[i] / (1.0f + expf(-h_gate[i])); // Silu
        hb[i]     = val * h_up[i];
    }
}

} // namespace Ops
