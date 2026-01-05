#pragma once

// ============================================================================
// Linear Operations
// ============================================================================

namespace Ops {

// Matrix Multiplication
// out[i] = dot(in, weight[i])
// weight is typically stored as [out_dim, in_dim]
inline void matmul(float *out, const float *in, const float *weight, int in_dim, int out_dim)
{
    for (int i = 0; i < out_dim; i++) {
        float        val   = 0.0f;
        const float *w_row = weight + i * in_dim;
        for (int j = 0; j < in_dim; j++) {
            val += in[j] * w_row[j];
        }
        out[i] = val;
    }
}

} // namespace Ops
