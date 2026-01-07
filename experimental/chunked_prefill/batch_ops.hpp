#pragma once

#include <cmath>
#include <cstring>

namespace BatchOps {

inline void batch_matmul(float *out, const float *in, const float *weight, int batch_size, int in_dim, int out_dim)
{
    for (int b = 0; b < batch_size; b++) {
        const float *in_row  = in + b * in_dim;
        float       *out_row = out + b * out_dim;

        for (int i = 0; i < out_dim; i++) {
            float        val   = 0.0f;
            const float *w_row = weight + i * in_dim;
            for (int j = 0; j < in_dim; j++) {
                val += in_row[j] * w_row[j];
            }
            out_row[i] = val;
        }
    }
}

inline void
batch_rope(float *q, float *k, int start_pos, int batch_size, int head_dim, int n_heads, int n_kv_heads, float theta)
{
    for (int b = 0; b < batch_size; b++) {
        int    pos = start_pos + b;
        float *q_b = q + b * n_heads * head_dim;
        float *k_b = k + b * n_kv_heads * head_dim;

        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(theta, (float)i / head_dim);
            float val  = pos * freq;
            float fcr  = cosf(val);
            float fci  = sinf(val);

            for (int h = 0; h < n_heads; h++) {
                float *vec = q_b + h * head_dim;
                float  v0  = vec[i];
                float  v1  = vec[i + 1];
                vec[i]     = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }

            for (int h = 0; h < n_kv_heads; h++) {
                float *vec = k_b + h * head_dim;
                float  v0  = vec[i];
                float  v1  = vec[i + 1];
                vec[i]     = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }
    }
}

inline void batch_rms_norm(float *out, const float *in, const float *weight, int batch_size, int dim)
{
    for (int b = 0; b < batch_size; b++) {
        const float *in_row  = in + b * dim;
        float       *out_row = out + b * dim;

        float ss = 0.0f;
        for (int i = 0; i < dim; i++) {
            ss += in_row[i] * in_row[i];
        }
        ss /= dim;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);

        for (int i = 0; i < dim; i++) {
            out_row[i] = in_row[i] * ss * weight[i];
        }
    }
}

} // namespace BatchOps
