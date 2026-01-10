#pragma once

#include <cmath>
#include <cstring>

// ============================================================================
// Attention Implementations
// ============================================================================

namespace Attention {

// ============================================================================
// Standard Attention (Original contiguous memory approach)
// ============================================================================

inline void standard_attention(float       *out,
                               const float *q,
                               const float *key_cache,   // Already offset to layer
                               const float *value_cache, // Already offset to layer
                               float       *att_scores,
                               int          pos,
                               int          head_dim,
                               int          n_heads,
                               int          n_kv_heads,
                               int          max_seq_len)
{
    int   kv_mul = n_heads / n_kv_heads;
    float scale  = 1.0f / sqrtf(head_dim);

    // Reset output
    std::memset(out, 0, n_heads * head_dim * sizeof(float));

    for (int h = 0; h < n_heads; h++) {
        const float *q_head   = q + h * head_dim;
        float       *att_head = att_scores + h * max_seq_len;
        int          kv_h     = h / kv_mul;

        // Score: Q * K^T
        for (int t = 0; t <= pos; t++) {
            const float *k_head = key_cache + t * n_kv_heads * head_dim + kv_h * head_dim;
            float        score  = 0.0f;
            for (int i = 0; i < head_dim; i++) {
                score += q_head[i] * k_head[i];
            }
            score *= scale;
            att_head[t] = score;
        }

        // Softmax
        float max_val = -1e10;
        for (int t = 0; t <= pos; t++) {
            if (att_head[t] > max_val)
                max_val = att_head[t];
        }

        float sum = 0.0f;
        for (int t = 0; t <= pos; t++) {
            att_head[t] = expf(att_head[t] - max_val);
            sum += att_head[t];
        }

        for (int t = 0; t <= pos; t++) {
            att_head[t] /= sum;
        }

        // Weighted sum: softmax(Q*K^T) * V
        float *out_head = out + h * head_dim;
        for (int t = 0; t <= pos; t++) {
            const float *v_head = value_cache + t * n_kv_heads * head_dim + kv_h * head_dim;
            float        prob   = att_head[t];
            for (int i = 0; i < head_dim; i++) {
                out_head[i] += prob * v_head[i];
            }
        }
    }
}

// ============================================================================
// Paged Attention (Block-based memory approach)
// ============================================================================

inline void paged_attention(float       *out,
                            const float *q,
                            const float *key_cache,
                            const float *value_cache,
                            const int   *block_table,
                            float       *att_scores,
                            int          num_tokens,
                            int          block_size,
                            int          head_dim,
                            int          n_heads,
                            int          n_kv_heads)
{
    int   kv_mul = n_heads / n_kv_heads;
    float scale  = 1.0f / sqrtf(head_dim);

    // Reset output
    std::memset(out, 0, n_heads * head_dim * sizeof(float));

    int num_blocks = (num_tokens + block_size - 1) / block_size;

    for (int h = 0; h < n_heads; h++) {
        const float *q_head   = q + h * head_dim;
        float       *att_head = att_scores + h * num_tokens;
        int          kv_h     = h / kv_mul;

        // Score: Q * K^T (using block table)
        for (int t = 0; t < num_tokens; t++) {
            int logical_block  = t / block_size;
            int block_offset   = t % block_size;
            int physical_block = block_table[logical_block];

            // KV cache layout: [num_physical_blocks, block_size, n_kv_heads, head_dim]
            const float *k_head = key_cache + physical_block * block_size * n_kv_heads * head_dim
                                + block_offset * n_kv_heads * head_dim + kv_h * head_dim;

            float score = 0.0f;
            for (int i = 0; i < head_dim; i++) {
                score += q_head[i] * k_head[i];
            }
            score *= scale;
            att_head[t] = score;
        }

        // Softmax
        float max_val = -1e10;
        for (int t = 0; t < num_tokens; t++) {
            if (att_head[t] > max_val)
                max_val = att_head[t];
        }

        float sum = 0.0f;
        for (int t = 0; t < num_tokens; t++) {
            att_head[t] = expf(att_head[t] - max_val);
            sum += att_head[t];
        }

        for (int t = 0; t < num_tokens; t++) {
            att_head[t] /= sum;
        }

        // Weighted sum: softmax(Q*K^T) * V (using block table)
        float *out_head = out + h * head_dim;
        for (int t = 0; t < num_tokens; t++) {
            int logical_block  = t / block_size;
            int block_offset   = t % block_size;
            int physical_block = block_table[logical_block];

            // Value cache layout: [num_physical_blocks, block_size, n_kv_heads, head_dim]
            const float *v_head = value_cache + physical_block * block_size * n_kv_heads * head_dim
                                + block_offset * n_kv_heads * head_dim + kv_h * head_dim;

            float prob = att_head[t];
            for (int i = 0; i < head_dim; i++) {
                out_head[i] += prob * v_head[i];
            }
        }
    }
}

} // namespace Attention
