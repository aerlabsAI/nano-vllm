#pragma once

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ops/activation.hpp"
#include "ops/linear.hpp"
#include "ops/normalization.hpp"
#include "ops/positional.hpp"
#include "utils/logger.hpp"

// ============================================================================
// Llama Model Configuration & Data Structures
// ============================================================================

struct Config
{
    int dim;         // Transformer dimension
    int hidden_dim;  // FFN hidden dimension
    int n_layers;    // Number of layers
    int n_heads;     // Number of query heads
    int n_kv_heads;  // Number of key/value heads (can be < n_heads for GQA)
    int vocab_size;  // Vocabulary size
    int max_seq_len; // Maximum sequence length

    // Derived/Constants
    int   head_dim;
    float rope_theta = 10000.0f;
};

// Holds the weights of the model.
// We use std::vector for automatic memory management.
struct TransformerWeights
{
    // Embedding
    std::vector<float> token_embedding_table; // [vocab_size, dim]

    // Layers
    struct Layer
    {
        std::vector<float> rms_att_weight; // [dim]
        std::vector<float> wq;             // [dim, n_heads * head_dim]
        std::vector<float> wk;             // [dim, n_kv_heads * head_dim]
        std::vector<float> wv;             // [dim, n_kv_heads * head_dim]
        std::vector<float> wo;             // [n_heads * head_dim, dim]
        std::vector<float> rms_ffn_weight; // [dim]
        std::vector<float> w_gate;         // [dim, hidden_dim]
        std::vector<float> w_up;           // [dim, hidden_dim]
        std::vector<float> w_down;         // [hidden_dim, dim]
    };

    std::vector<Layer> layers;

    // Final RMSNorm
    std::vector<float> rms_final_weight; // [dim]

    // Output Head (optional if shared)
    std::vector<float> lm_head; // [vocab_size, dim]
    bool               weights_shared = false;
};

// Runtime state buffers
struct RunState
{
    // Current hidden states
    std::vector<float> x;      // [dim]
    std::vector<float> xb;     // [dim]
    std::vector<float> xb2;    // [dim]
    std::vector<float> hb;     // [hidden_dim]
    std::vector<float> hb2;    // [hidden_dim]
    std::vector<float> q;      // [dim]
    std::vector<float> k;      // [dim]
    std::vector<float> v;      // [dim]
    std::vector<float> att;    // [n_heads, seq_len]
    std::vector<float> logits; // [vocab_size]

    // KV Cache
    // Layout: [n_layers, max_seq_len, n_kv_heads, head_dim]
    std::vector<float> key_cache;
    std::vector<float> value_cache;
};

// ============================================================================
// Llama Model
// ============================================================================

class LlamaModel
{
public:
    Config             config;
    TransformerWeights weights;
    RunState           state;

    void load(const std::string &path)
    {
        LOG_INFO("Loading model: ", path);
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open model file");
        }

        // Read config
        file.read(reinterpret_cast<char *>(&config), 7 * sizeof(int));
        // Calculate derived
        config.head_dim = config.dim / config.n_heads;

        LOG_INFO("Config: dim=",
                 config.dim,
                 " layers=",
                 config.n_layers,
                 " heads=",
                 config.n_heads,
                 " vocab=",
                 config.vocab_size);

        // Allocate weights
        resize_weights();

        // Read weights
        read_weights(file);

        // Allocate run state
        resize_run_state();
    }

    void forward(int token, int pos)
    {
        // 1. Embedding
        const float *content_row = weights.token_embedding_table.data() + token * config.dim;
        std::memcpy(state.x.data(), content_row, config.dim * sizeof(float));

        // 2. Layers
        for (int i = 0; i < config.n_layers; i++) {
            auto &l = weights.layers[i];

            // RMSNorm
            Ops::rms_norm(state.xb.data(), state.x.data(), l.rms_att_weight.data(), config.dim);

            // QKV Matmul
            Ops::matmul(state.q.data(), state.xb.data(), l.wq.data(), config.dim, config.n_heads * config.head_dim);
            Ops::matmul(state.k.data(), state.xb.data(), l.wk.data(), config.dim, config.n_kv_heads * config.head_dim);
            Ops::matmul(state.v.data(), state.xb.data(), l.wv.data(), config.dim, config.n_kv_heads * config.head_dim);

            // RoPE
            Ops::apply_rope(state.q.data(),
                            state.k.data(),
                            pos,
                            config.head_dim,
                            config.n_heads,
                            config.n_kv_heads,
                            config.rope_theta);

            // Save to KV Cache
            int    layer_offset = i * config.max_seq_len * config.n_kv_heads * config.head_dim;
            int    pos_offset   = pos * config.n_kv_heads * config.head_dim;
            float *k_cache_ptr  = state.key_cache.data() + layer_offset + pos_offset;
            float *v_cache_ptr  = state.value_cache.data() + layer_offset + pos_offset;

            std::memcpy(k_cache_ptr, state.k.data(), config.n_kv_heads * config.head_dim * sizeof(float));
            std::memcpy(v_cache_ptr, state.v.data(), config.n_kv_heads * config.head_dim * sizeof(float));

            // Multi-head Attention
            attention(i, pos, state.xb2.data()); // writes to xb2

            // Output Projection
            Ops::matmul(state.xb.data(), state.xb2.data(), l.wo.data(), config.n_heads * config.head_dim, config.dim);

            // Residual
            for (int j = 0; j < config.dim; j++)
                state.x[j] += state.xb[j];

            // FFN
            Ops::rms_norm(state.xb.data(), state.x.data(), l.rms_ffn_weight.data(), config.dim);
            Ops::matmul(state.hb.data(), state.xb.data(), l.w_gate.data(), config.dim, config.hidden_dim);
            Ops::matmul(state.hb2.data(), state.xb.data(), l.w_up.data(), config.dim, config.hidden_dim);
            Ops::swiglu(state.hb.data(), state.hb.data(), state.hb2.data(), config.hidden_dim);
            Ops::matmul(state.xb.data(), state.hb.data(), l.w_down.data(), config.hidden_dim, config.dim);

            // Residual
            for (int j = 0; j < config.dim; j++)
                state.x[j] += state.xb[j];
        }

        // Final RMSNorm
        Ops::rms_norm(state.x.data(), state.x.data(), weights.rms_final_weight.data(), config.dim);

        // Classifier
        Ops::matmul(state.logits.data(), state.x.data(), weights.lm_head.data(), config.dim, config.vocab_size);
    }

private:
    void resize_weights()
    {
        weights.token_embedding_table.resize(config.vocab_size * config.dim);
        weights.layers.resize(config.n_layers);
        for (auto &l : weights.layers) {
            l.rms_att_weight.resize(config.dim);
            l.wq.resize(config.dim * config.n_heads * config.head_dim);
            l.wk.resize(config.dim * config.n_kv_heads * config.head_dim);
            l.wv.resize(config.dim * config.n_kv_heads * config.head_dim);
            l.wo.resize(config.n_heads * config.head_dim * config.dim);
            l.rms_ffn_weight.resize(config.dim);
            l.w_gate.resize(config.dim * config.hidden_dim);
            l.w_up.resize(config.dim * config.hidden_dim);
            l.w_down.resize(config.hidden_dim * config.dim);
        }
        weights.rms_final_weight.resize(config.dim);
        weights.lm_head.resize(config.vocab_size * config.dim);
    }

    // Reads data assuming flat float structure
    void read_weights(std::ifstream &file)
    {
        auto read_tensor = [&](std::vector<float> &vec) {
            file.read(reinterpret_cast<char *>(vec.data()), vec.size() * sizeof(float));
        };

        read_tensor(weights.token_embedding_table);

        // llama2.c file format stores weights grouped by parameter type, not by layer.
        // We must iterate over layers for each parameter type.

        for (int i = 0; i < config.n_layers; i++)
            read_tensor(weights.layers[i].rms_att_weight);
        for (int i = 0; i < config.n_layers; i++)
            read_tensor(weights.layers[i].wq);
        for (int i = 0; i < config.n_layers; i++)
            read_tensor(weights.layers[i].wk);
        for (int i = 0; i < config.n_layers; i++)
            read_tensor(weights.layers[i].wv);
        for (int i = 0; i < config.n_layers; i++)
            read_tensor(weights.layers[i].wo);
        for (int i = 0; i < config.n_layers; i++)
            read_tensor(weights.layers[i].rms_ffn_weight);

        for (int i = 0; i < config.n_layers; i++)
            read_tensor(weights.layers[i].w_gate);
        for (int i = 0; i < config.n_layers; i++)
            read_tensor(weights.layers[i].w_down);
        for (int i = 0; i < config.n_layers; i++)
            read_tensor(weights.layers[i].w_up);

        read_tensor(weights.rms_final_weight);

        // Handling shared weights vs non-shared
        long current = file.tellg();
        file.seekg(0, std::ios::end);
        long end       = file.tellg();
        long remaining = end - current;
        file.seekg(current, std::ios::beg);

        if (remaining >= weights.lm_head.size() * sizeof(float)) {
            read_tensor(weights.lm_head);
        }
        else {
            // Shared weights
            weights.weights_shared = true;
            weights.lm_head        = weights.token_embedding_table;
            LOG_INFO("Weights shared: lm_head <- token_embedding");
        }
    }

    void resize_run_state()
    {
        state.x.resize(config.dim);
        state.xb.resize(config.dim);
        state.xb2.resize(config.dim);
        state.hb.resize(config.hidden_dim);
        state.hb2.resize(config.hidden_dim);
        state.q.resize(config.dim);
        state.k.resize(config.dim);
        state.v.resize(config.dim);
        state.att.resize(config.n_heads * config.max_seq_len);
        state.logits.resize(config.vocab_size);

        // KV Cache
        size_t cache_size = static_cast<size_t>(config.n_layers) * static_cast<size_t>(config.max_seq_len)
                          * static_cast<size_t>(config.n_kv_heads) * static_cast<size_t>(config.head_dim);

        constexpr size_t MAX_CACHE_ELEMENTS = 25'000'000'000ULL; // ~100GB in floats
        if (cache_size > MAX_CACHE_ELEMENTS) {
            throw std::runtime_error("KV cache size exceeds limit");
        }

        state.key_cache.resize(cache_size);
        state.value_cache.resize(cache_size);
    }

    void attention(int layer, int pos, float *out)
    {
        float *att        = state.att.data();
        float *q          = state.q.data();
        int    head_dim   = config.head_dim;
        int    n_heads    = config.n_heads;
        int    n_kv_heads = config.n_kv_heads;

        int   kv_mul = n_heads / n_kv_heads;
        float scale  = 1.0f / sqrtf(head_dim);

        // Reset out
        std::memset(out, 0, n_heads * head_dim * sizeof(float));

        int layer_offset = layer * config.max_seq_len * n_kv_heads * head_dim;

        for (int h = 0; h < n_heads; h++) {
            float *q_head   = q + h * head_dim;
            float *att_head = att + h * config.max_seq_len;
            int    kv_h     = h / kv_mul;

            // Score
            for (int t = 0; t <= pos; t++) {
                float *k_head = state.key_cache.data() + layer_offset + t * n_kv_heads * head_dim + kv_h * head_dim;
                float  score  = 0.0f;
                for (int i = 0; i < head_dim; i++) {
                    score += q_head[i] * k_head[i];
                }
                score *= scale;
                att_head[t] = score;
            }

            // Softmax
            float max_val = -1e10; // -INFINITY
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

            // Weighted Sum
            float *out_head = out + h * head_dim;
            for (int t = 0; t <= pos; t++) {
                float *v_head = state.value_cache.data() + layer_offset + t * n_kv_heads * head_dim + kv_h * head_dim;
                float  prob   = att_head[t];
                for (int i = 0; i < head_dim; i++) {
                    out_head[i] += prob * v_head[i];
                }
            }
        }
    }
};
