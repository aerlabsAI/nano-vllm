#pragma once

#include <cmath>

// ============================================================================
// Positional Encoding Operations
// ============================================================================

namespace Ops {

// Rotary Position Embedding (RoPE)
// Apply rotary embeddings to query and key tensors
inline void apply_rope(float *q, float *k, int pos, int head_dim, int n_heads, int n_kv_heads, float theta)
{
    for (int i = 0; i < head_dim; i += 2) {
        float freq = 1.0f / powf(theta, (float)i / head_dim);
        float val  = pos * freq;
        float fcr  = cosf(val);
        float fci  = sinf(val);

        // Apply to Query
        for (int h = 0; h < n_heads; h++) {
            float *vec = q + h * head_dim;
            float  v0  = vec[i];
            float  v1  = vec[i + 1];
            vec[i]     = v0 * fcr - v1 * fci;
            vec[i + 1] = v0 * fci + v1 * fcr;
        }

        // Apply to Key
        for (int h = 0; h < n_kv_heads; h++) {
            float *vec = k + h * head_dim;
            float  v0  = vec[i];
            float  v1  = vec[i + 1];
            vec[i]     = v0 * fcr - v1 * fci;
            vec[i + 1] = v0 * fci + v1 * fcr;
        }
    }
}

} // namespace Ops
