#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

#include "../../include/core/model.hpp"
#include "../../include/ops/activation.hpp"
#include "batch_ops.hpp"
#include "chunking.hpp"

struct ChunkedRunState
{
    int max_chunk_size = 0;

    std::vector<float> x_batch;
    std::vector<float> xb_batch;
    std::vector<float> xb2_batch;
    std::vector<float> hb_batch;
    std::vector<float> hb2_batch;
    std::vector<float> q_batch;
    std::vector<float> k_batch;
    std::vector<float> v_batch;
    std::vector<float> att_batch;

    void resize(int chunk_size, int dim, int hidden_dim, int n_heads, int max_seq_len)
    {
        max_chunk_size = chunk_size;
        x_batch.resize(chunk_size * dim);
        xb_batch.resize(chunk_size * dim);
        xb2_batch.resize(chunk_size * dim);
        hb_batch.resize(chunk_size * hidden_dim);
        hb2_batch.resize(chunk_size * hidden_dim);
        q_batch.resize(chunk_size * dim);
        k_batch.resize(chunk_size * dim);
        v_batch.resize(chunk_size * dim);
        att_batch.resize(chunk_size * n_heads * max_seq_len);
    }
};

class LlamaModelChunked : public LlamaModel
{
public:
    ChunkedRunState chunk_state;

    void forward_chunk(const std::vector<int> &chunk_tokens, int start_pos)
    {
        int chunk_size = static_cast<int>(chunk_tokens.size());

        if (chunk_state.max_chunk_size < chunk_size) {
            chunk_state.resize(chunk_size, config.dim, config.hidden_dim, config.n_heads, config.max_seq_len);
        }

        float *x   = chunk_state.x_batch.data();
        float *xb  = chunk_state.xb_batch.data();
        float *xb2 = chunk_state.xb2_batch.data();
        float *hb  = chunk_state.hb_batch.data();
        float *hb2 = chunk_state.hb2_batch.data();
        float *q   = chunk_state.q_batch.data();
        float *k   = chunk_state.k_batch.data();
        float *v   = chunk_state.v_batch.data();

        for (int b = 0; b < chunk_size; b++) {
            const float *embedding = weights.token_embedding_table.data() + chunk_tokens[b] * config.dim;
            std::memcpy(x + b * config.dim, embedding, config.dim * sizeof(float));
        }

        for (int layer = 0; layer < config.n_layers; layer++) {
            auto &l = weights.layers[layer];

            BatchOps::batch_rms_norm(xb, x, l.rms_att_weight.data(), chunk_size, config.dim);

            BatchOps::batch_matmul(q, xb, l.wq.data(), chunk_size, config.dim, config.n_heads * config.head_dim);
            BatchOps::batch_matmul(k, xb, l.wk.data(), chunk_size, config.dim, config.n_kv_heads * config.head_dim);
            BatchOps::batch_matmul(v, xb, l.wv.data(), chunk_size, config.dim, config.n_kv_heads * config.head_dim);

            BatchOps::batch_rope(
                q, k, start_pos, chunk_size, config.head_dim, config.n_heads, config.n_kv_heads, config.rope_theta);

            int layer_offset = layer * config.max_seq_len * config.n_kv_heads * config.head_dim;
            for (int b = 0; b < chunk_size; b++) {
                int    pos         = start_pos + b;
                int    pos_offset  = pos * config.n_kv_heads * config.head_dim;
                float *k_cache_ptr = state.key_cache.data() + layer_offset + pos_offset;
                float *v_cache_ptr = state.value_cache.data() + layer_offset + pos_offset;

                float *k_b = k + b * config.n_kv_heads * config.head_dim;
                float *v_b = v + b * config.n_kv_heads * config.head_dim;

                std::memcpy(k_cache_ptr, k_b, config.n_kv_heads * config.head_dim * sizeof(float));
                std::memcpy(v_cache_ptr, v_b, config.n_kv_heads * config.head_dim * sizeof(float));
            }

            chunked_attention(layer, chunk_size, start_pos, xb2);

            BatchOps::batch_matmul(xb, xb2, l.wo.data(), chunk_size, config.n_heads * config.head_dim, config.dim);

            for (int b = 0; b < chunk_size; b++) {
                for (int i = 0; i < config.dim; i++) {
                    x[b * config.dim + i] += xb[b * config.dim + i];
                }
            }

            BatchOps::batch_rms_norm(xb, x, l.rms_ffn_weight.data(), chunk_size, config.dim);
            BatchOps::batch_matmul(hb, xb, l.w_gate.data(), chunk_size, config.dim, config.hidden_dim);
            BatchOps::batch_matmul(hb2, xb, l.w_up.data(), chunk_size, config.dim, config.hidden_dim);

            for (int b = 0; b < chunk_size; b++) {
                Ops::swiglu(hb + b * config.hidden_dim,
                            hb + b * config.hidden_dim,
                            hb2 + b * config.hidden_dim,
                            config.hidden_dim);
            }

            BatchOps::batch_matmul(xb, hb, l.w_down.data(), chunk_size, config.hidden_dim, config.dim);

            for (int b = 0; b < chunk_size; b++) {
                for (int i = 0; i < config.dim; i++) {
                    x[b * config.dim + i] += xb[b * config.dim + i];
                }
            }
        }

        for (int b = 0; b < chunk_size; b++) {
            float *x_b = x + b * config.dim;
            std::memcpy(state.x.data(), x_b, config.dim * sizeof(float));
            Ops::rms_norm(state.x.data(), state.x.data(), weights.rms_final_weight.data(), config.dim);
            Ops::matmul(state.logits.data(), state.x.data(), weights.lm_head.data(), config.dim, config.vocab_size);
        }
    }

    ChunkedPrefill::PrefillMetrics prefill_chunked(const std::vector<int> &tokens, int chunk_size)
    {
        auto chunks = ChunkedPrefill::create_chunks(tokens, chunk_size);

        auto                           start = std::chrono::high_resolution_clock::now();
        std::vector<double>            chunk_times;
        ChunkedPrefill::PrefillMetrics metrics;

        for (const auto &chunk : chunks) {
            auto chunk_start = std::chrono::high_resolution_clock::now();

            forward_chunk(chunk.tokens, chunk.start_pos);

            auto   chunk_end  = std::chrono::high_resolution_clock::now();
            double chunk_time = std::chrono::duration<double, std::milli>(chunk_end - chunk_start).count();
            chunk_times.push_back(chunk_time);
        }

        auto end = std::chrono::high_resolution_clock::now();

        metrics.total_time_ms     = std::chrono::duration<double, std::milli>(end - start).count();
        metrics.num_chunks        = static_cast<int>(chunks.size());
        metrics.total_tokens      = static_cast<int>(tokens.size());
        metrics.chunk_size        = chunk_size;
        metrics.avg_chunk_time_ms = 0.0;
        for (double t : chunk_times)
            metrics.avg_chunk_time_ms += t;
        metrics.avg_chunk_time_ms /= std::max(1, metrics.num_chunks);

        return metrics;
    }

private:
    void chunked_attention(int layer, int chunk_size, int start_pos, float *out)
    {
        float *att = chunk_state.att_batch.data();
        float *q   = chunk_state.q_batch.data();

        int   kv_mul       = config.n_heads / config.n_kv_heads;
        float scale        = 1.0f / sqrtf(config.head_dim);
        int   layer_offset = layer * config.max_seq_len * config.n_kv_heads * config.head_dim;

        std::memset(out, 0, chunk_size * config.n_heads * config.head_dim * sizeof(float));

        for (int b = 0; b < chunk_size; b++) {
            int curr_pos = start_pos + b;

            for (int h = 0; h < config.n_heads; h++) {
                float *q_head   = q + b * config.n_heads * config.head_dim + h * config.head_dim;
                float *att_head = att + b * config.n_heads * config.max_seq_len + h * config.max_seq_len;
                int    kv_h     = h / kv_mul;

                for (int t = 0; t <= curr_pos; t++) {
                    float *k_head = state.key_cache.data() + layer_offset + t * config.n_kv_heads * config.head_dim
                                  + kv_h * config.head_dim;
                    float score = 0.0f;
                    for (int i = 0; i < config.head_dim; i++) {
                        score += q_head[i] * k_head[i];
                    }
                    score *= scale;
                    att_head[t] = score;
                }

                float max_val = -1e10f;
                for (int t = 0; t <= curr_pos; t++) {
                    if (att_head[t] > max_val)
                        max_val = att_head[t];
                }

                float sum = 0.0f;
                for (int t = 0; t <= curr_pos; t++) {
                    att_head[t] = expf(att_head[t] - max_val);
                    sum += att_head[t];
                }

                for (int t = 0; t <= curr_pos; t++) {
                    att_head[t] /= sum;
                }

                float *out_head = out + b * config.n_heads * config.head_dim + h * config.head_dim;
                for (int t = 0; t <= curr_pos; t++) {
                    float *v_head = state.value_cache.data() + layer_offset + t * config.n_kv_heads * config.head_dim
                                  + kv_h * config.head_dim;
                    float prob = att_head[t];
                    for (int i = 0; i < config.head_dim; i++) {
                        out_head[i] += prob * v_head[i];
                    }
                }
            }
        }
    }
};
