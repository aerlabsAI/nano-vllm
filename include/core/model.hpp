#pragma once

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/attention.hpp"
#include "ops/activation.hpp"
#include "ops/linear.hpp"
#include "ops/normalization.hpp"
#include "ops/positional.hpp"
#include "scheduler/block_manager.hpp"
#include "scheduler/request.hpp"
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

    // PagedAttention configuration
    bool use_paged_attention = false; // Enable/disable PagedAttention
    int  block_size          = 16;    // Block size for PagedAttention (in tokens)
    int  num_blocks          = 256;   // Total number of physical blocks

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

    // Standard KV Cache (contiguous memory)
    // Layout: [n_layers, max_seq_len, n_kv_heads, head_dim]
    std::vector<float> key_cache;
    std::vector<float> value_cache;

    // Paged KV Cache (block-based memory)
    // Layout: [n_layers, num_blocks, block_size, n_kv_heads, head_dim]
    std::vector<float> paged_key_cache;
    std::vector<float> paged_value_cache;
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

    // PagedAttention components
    BlockManager                 *block_manager = nullptr;
    std::vector<std::vector<int>> block_tables; // [n_layers][logical_blocks]

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
            if (config.use_paged_attention) {
                // PagedAttention: Block-based KV cache
                // Allocate new block if needed
                if (pos % config.block_size == 0) {
                    int new_block = block_manager->allocate_block();
                    if (new_block == -1) {
                        throw std::runtime_error("Out of memory: no free blocks");
                    }
                    block_tables[i].push_back(new_block);
                }

                // Get current block
                int logical_block  = pos / config.block_size;
                int block_offset   = pos % config.block_size;
                int physical_block = block_tables[i][logical_block];

                // Calculate cache pointers
                // Layout: [n_layers, num_blocks, block_size, n_kv_heads, head_dim]
                size_t layer_cache_offset =
                    i * config.num_blocks * config.block_size * config.n_kv_heads * config.head_dim;
                size_t block_cache_offset = physical_block * config.block_size * config.n_kv_heads * config.head_dim;
                size_t pos_cache_offset   = block_offset * config.n_kv_heads * config.head_dim;

                float *k_cache_ptr =
                    state.paged_key_cache.data() + layer_cache_offset + block_cache_offset + pos_cache_offset;
                float *v_cache_ptr =
                    state.paged_value_cache.data() + layer_cache_offset + block_cache_offset + pos_cache_offset;

                std::memcpy(k_cache_ptr, state.k.data(), config.n_kv_heads * config.head_dim * sizeof(float));
                std::memcpy(v_cache_ptr, state.v.data(), config.n_kv_heads * config.head_dim * sizeof(float));
            }
            else {
                // Standard: Contiguous KV cache
                int    layer_offset = i * config.max_seq_len * config.n_kv_heads * config.head_dim;
                int    pos_offset   = pos * config.n_kv_heads * config.head_dim;
                float *k_cache_ptr  = state.key_cache.data() + layer_offset + pos_offset;
                float *v_cache_ptr  = state.value_cache.data() + layer_offset + pos_offset;

                std::memcpy(k_cache_ptr, state.k.data(), config.n_kv_heads * config.head_dim * sizeof(float));
                std::memcpy(v_cache_ptr, state.v.data(), config.n_kv_heads * config.head_dim * sizeof(float));
            }

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

    void initialize_paged_attention()
    {
        if (!config.use_paged_attention) {
            return;
        }

        if (block_manager != nullptr) {
            delete block_manager;
            block_manager = nullptr;
        }

        block_manager = new BlockManager(config.num_blocks, config.block_size);

        block_tables.resize(config.n_layers);
        for (auto &table : block_tables) {
            table.clear();
        }

        size_t paged_cache_size = static_cast<size_t>(config.n_layers) * static_cast<size_t>(config.num_blocks)
                                * static_cast<size_t>(config.block_size) * static_cast<size_t>(config.n_kv_heads)
                                * static_cast<size_t>(config.head_dim);

        state.paged_key_cache.resize(paged_cache_size);
        state.paged_value_cache.resize(paged_cache_size);

        LOG_SUCCESS("PagedAttention initialized: ",
                    config.num_blocks,
                    " blocks × ",
                    config.block_size,
                    " tokens = ",
                    config.num_blocks * config.block_size,
                    " total capacity");
    }

    // Forward pass with per-request KV cache isolation
    // Enables continuous batching by using req->block_tables instead of global block_tables
    void forward_with_request(int token, int pos, Request *req)
    {
        if (req->block_tables.empty()) {
            req->block_tables.resize(config.n_layers);
        }

        const float *content_row = weights.token_embedding_table.data() + token * config.dim;
        std::memcpy(state.x.data(), content_row, config.dim * sizeof(float));

        for (int i = 0; i < config.n_layers; i++) {
            auto &l = weights.layers[i];

            Ops::rms_norm(state.xb.data(), state.x.data(), l.rms_att_weight.data(), config.dim);

            Ops::matmul(state.q.data(), state.xb.data(), l.wq.data(), config.dim, config.n_heads * config.head_dim);
            Ops::matmul(state.k.data(), state.xb.data(), l.wk.data(), config.dim, config.n_kv_heads * config.head_dim);
            Ops::matmul(state.v.data(), state.xb.data(), l.wv.data(), config.dim, config.n_kv_heads * config.head_dim);

            Ops::apply_rope(state.q.data(),
                            state.k.data(),
                            pos,
                            config.head_dim,
                            config.n_heads,
                            config.n_kv_heads,
                            config.rope_theta);

            // Allocate new block when reaching block boundary
            if (pos % config.block_size == 0) {
                int new_block = block_manager->allocate_block_for_request(req->id);
                if (new_block == -1) {
                    throw std::runtime_error("Out of memory: no free blocks for request");
                }
                req->block_tables[i].push_back(new_block);
            }

            // Map logical block to physical block using per-request block table
            int logical_block  = pos / config.block_size;
            int block_offset   = pos % config.block_size;
            int physical_block = req->block_tables[i][logical_block];

            // Calculate KV cache pointers: [layer][block][position][head][dim]
            size_t layer_cache_offset = i * config.num_blocks * config.block_size * config.n_kv_heads * config.head_dim;
            size_t block_cache_offset = physical_block * config.block_size * config.n_kv_heads * config.head_dim;
            size_t pos_cache_offset   = block_offset * config.n_kv_heads * config.head_dim;

            float *k_cache_ptr =
                state.paged_key_cache.data() + layer_cache_offset + block_cache_offset + pos_cache_offset;
            float *v_cache_ptr =
                state.paged_value_cache.data() + layer_cache_offset + block_cache_offset + pos_cache_offset;

            std::memcpy(k_cache_ptr, state.k.data(), config.n_kv_heads * config.head_dim * sizeof(float));
            std::memcpy(v_cache_ptr, state.v.data(), config.n_kv_heads * config.head_dim * sizeof(float));

            attention_with_request(i, pos, state.xb2.data(), req);

            Ops::matmul(state.xb.data(), state.xb2.data(), l.wo.data(), config.n_heads * config.head_dim, config.dim);

            for (int j = 0; j < config.dim; j++)
                state.x[j] += state.xb[j];

            Ops::rms_norm(state.xb.data(), state.x.data(), l.rms_ffn_weight.data(), config.dim);
            Ops::matmul(state.hb.data(), state.xb.data(), l.w_gate.data(), config.dim, config.hidden_dim);
            Ops::matmul(state.hb2.data(), state.xb.data(), l.w_up.data(), config.dim, config.hidden_dim);
            Ops::swiglu(state.hb.data(), state.hb.data(), state.hb2.data(), config.hidden_dim);
            Ops::matmul(state.xb.data(), state.hb.data(), l.w_down.data(), config.hidden_dim, config.dim);

            for (int j = 0; j < config.dim; j++)
                state.x[j] += state.xb[j];
        }

        Ops::rms_norm(state.x.data(), state.x.data(), weights.rms_final_weight.data(), config.dim);

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
        if (config.use_paged_attention) {
            int num_tokens = pos + 1;

            size_t layer_cache_offset =
                layer * config.num_blocks * config.block_size * config.n_kv_heads * config.head_dim;
            const float *paged_key_ptr   = state.paged_key_cache.data() + layer_cache_offset;
            const float *paged_value_ptr = state.paged_value_cache.data() + layer_cache_offset;

            Attention::paged_attention(out,
                                       state.q.data(),
                                       paged_key_ptr,
                                       paged_value_ptr,
                                       block_tables[layer].data(),
                                       state.att.data(),
                                       num_tokens,
                                       config.block_size,
                                       config.head_dim,
                                       config.n_heads,
                                       config.n_kv_heads);
        }
        else {
            int layer_offset = layer * config.max_seq_len * config.n_kv_heads * config.head_dim;

            Attention::standard_attention(out,
                                          state.q.data(),
                                          state.key_cache.data() + layer_offset,
                                          state.value_cache.data() + layer_offset,
                                          state.att.data(),
                                          pos,
                                          config.head_dim,
                                          config.n_heads,
                                          config.n_kv_heads,
                                          config.max_seq_len);
        }
    }

    void attention_with_request(int layer, int pos, float *out, Request *req)
    {
        int num_tokens = pos + 1;

        size_t layer_cache_offset = layer * config.num_blocks * config.block_size * config.n_kv_heads * config.head_dim;
        const float *paged_key_ptr   = state.paged_key_cache.data() + layer_cache_offset;
        const float *paged_value_ptr = state.paged_value_cache.data() + layer_cache_offset;

        Attention::paged_attention(out,
                                   state.q.data(),
                                   paged_key_ptr,
                                   paged_value_ptr,
                                   req->block_tables[layer].data(),
                                   state.att.data(),
                                   num_tokens,
                                   config.block_size,
                                   config.head_dim,
                                   config.n_heads,
                                   config.n_kv_heads);
    }
};
