#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "argparser.hpp"

// ============================================================================
// 1. Configuration & Data Structures
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
    // Note: To keep it simple and flat like the C version, we could have arrays of vectors
    // or flat vectors. Let's use flat vectors for each layer component to match the file structure.
    // Actually, storing them as a vector of structs is cleaner in C++.

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
    // Flattened: layer * (max_seq_len * n_kv_heads * head_dim) + pos * (...)
    std::vector<float> key_cache;
    std::vector<float> value_cache;
};

// ============================================================================
// 2. Utils & Math Operations
// ============================================================================

namespace Ops {

void rms_norm(float *out, const float *in, const float *weight, int size, float eps = 1e-5f)
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

void matmul(float *out, const float *in, const float *weight, int in_dim, int out_dim)
{
    // out[i] = dot(in, weight[i])
    // weight is typically stored as [out_dim, in_dim]
    // We implement a naive loop here.
    // Optimization (e.g., OpenMP) can be added later.

    for (int i = 0; i < out_dim; i++) {
        float        val   = 0.0f;
        const float *w_row = weight + i * in_dim;
        for (int j = 0; j < in_dim; j++) {
            val += in[j] * w_row[j];
        }
        out[i] = val;
    }
}

void softmax(float *x, int size)
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

void swiglu(float *hb, const float *h_gate, const float *h_up, int hidden_dim)
{
    for (int i = 0; i < hidden_dim; i++) {
        float val = h_gate[i] / (1.0f + expf(-h_gate[i])); // Silu
        hb[i]     = val * h_up[i];
    }
}

void apply_rope(float *q, float *k, int pos, int head_dim, int n_heads, int n_kv_heads, float theta)
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

// ============================================================================
// 3. Tokenizer
// ============================================================================

class Tokenizer
{
public:
    struct TokenIndex
    {
        std::string str;
        int         id;
        bool        operator<(const TokenIndex &other) const { return str < other.str; }
    };

    Tokenizer(const std::string &path, int vocab_size)
        : vocab_size(vocab_size)
    {
        load(path);
    }

    void load(const std::string &path)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open tokenizer: " + path);
        }

        int max_token_length_val;
        file.read(reinterpret_cast<char *>(&max_token_length_val), sizeof(int));
        this->max_token_length = max_token_length_val;

        vocab.resize(vocab_size);
        vocab_scores.resize(vocab_size);

        for (int i = 0; i < vocab_size; i++) {
            file.read(reinterpret_cast<char *>(&vocab_scores[i]), sizeof(float));
            int len;
            file.read(reinterpret_cast<char *>(&len), sizeof(int));
            std::string word(len, '\0');
            file.read(&word[0], len);
            vocab[i] = word;
        }

        // Build sorted vocab for fast lookup
        for (int i = 0; i < vocab_size; i++) {
            sorted_vocab.push_back({vocab[i], i});
        }
        std::sort(sorted_vocab.begin(), sorted_vocab.end());
    }

    std::string decode(int token) const
    {
        if (token < 0 || token >= vocab_size)
            return "";
        std::string piece = vocab[token];

        // Handle raw byte tokens like <0x01>
        if (piece.rfind("<0x", 0) == 0 && piece.size() == 6) {
            int byte_val = std::stoi(piece.substr(3, 2), nullptr, 16);
            return std::string(1, (char)byte_val);
        }
        return piece;
    }

    std::vector<int> encode(const std::string &text, bool bos = true, bool eos = false)
    {
        std::vector<int> tokens;
        if (bos)
            tokens.push_back(1); // BOS

        if (!text.empty()) {
            // Prepend dummy prefix if needed (simplified)
            int dummy = str_lookup(" ");
            if (dummy != -1)
                tokens.push_back(dummy);
        }

        // Basic UTF-8 parsing to bytes
        for (char c : text) {
            std::string s(1, c);
            int         id = str_lookup(s);
            if (id != -1) {
                tokens.push_back(id);
            }
            else {
                // Byte fallback
                // For now, map to <0xXX> or just ignore/fallback logic
                // The C implementation had a specific fallback.
                // For simplicity, we skip complex fallback logic here.
            }
        }

        // Merge pairs
        while (true) {
            float best_score = -1e10;
            int   best_id    = -1;
            int   best_idx   = -1;

            for (size_t i = 0; i < tokens.size() - 1; i++) {
                std::string merged = vocab[tokens[i]] + vocab[tokens[i + 1]];
                int         id     = str_lookup(merged);
                if (id != -1 && vocab_scores[id] > best_score) {
                    best_score = vocab_scores[id];
                    best_id    = id;
                    best_idx   = i;
                }
            }

            if (best_idx == -1)
                break;

            tokens[best_idx] = best_id;
            tokens.erase(tokens.begin() + best_idx + 1);
        }

        if (eos)
            tokens.push_back(2); // EOS
        return tokens;
    }

private:
    int                      vocab_size;
    int                      max_token_length;
    std::vector<std::string> vocab;
    std::vector<float>       vocab_scores;
    std::vector<TokenIndex>  sorted_vocab;

    int str_lookup(const std::string &str) const
    {
        TokenIndex query = {str, 0};
        auto       it    = std::lower_bound(sorted_vocab.begin(), sorted_vocab.end(), query);
        if (it != sorted_vocab.end() && it->str == str) {
            return it->id;
        }
        return -1;
    }
};

// ============================================================================
// 4. Sampler
// ============================================================================

class Sampler
{
public:
    Sampler(int vocab_size, float temp, float topp, unsigned long long seed)
        : vocab_size(vocab_size)
        , temperature(temp)
        , topp(topp)
        , rng(seed)
    {
    }

    int sample(float *logits)
    {
        // 1. Temperature
        if (temperature == 0.0f) {
            return std::distance(logits, std::max_element(logits, logits + vocab_size));
        }

        for (int i = 0; i < vocab_size; i++) {
            logits[i] /= temperature;
        }

        // 2. Softmax
        Ops::softmax(logits, vocab_size);

        // 3. Top-p (Nucleus) or Argmax or Random
        // Simple random sample from distribution
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float                                 r = dist(rng);

        if (topp > 0.0f && topp < 1.0f) {
            // Sort indices by probability
            struct ProbIndex
            {
                float p;
                int   i;
            };
            std::vector<ProbIndex> probs(vocab_size);
            for (int i = 0; i < vocab_size; i++)
                probs[i] = {logits[i], i};

            std::sort(probs.begin(), probs.end(), [](const ProbIndex &a, const ProbIndex &b) { return a.p > b.p; });

            float cum_prob = 0.0f;
            int   last_idx = vocab_size - 1;
            for (int i = 0; i < vocab_size; i++) {
                cum_prob += probs[i].p;
                if (cum_prob > topp) {
                    last_idx = i;
                    break;
                }
            }

            float r_scaled = r * cum_prob;
            float cdf      = 0.0f;
            for (int i = 0; i <= last_idx; i++) {
                cdf += probs[i].p;
                if (r_scaled < cdf)
                    return probs[i].i;
            }
            return probs[last_idx].i;
        }
        else {
            float cdf = 0.0f;
            for (int i = 0; i < vocab_size; i++) {
                cdf += logits[i];
                if (r < cdf)
                    return i;
            }
            return vocab_size - 1;
        }
    }

private:
    int          vocab_size;
    float        temperature;
    float        topp;
    std::mt19937 rng;
};

// ============================================================================
// 5. Model Engine
// ============================================================================

class LlamaModel
{
public:
    Config             config;
    TransformerWeights weights;
    RunState           state;

    void load(const std::string &path)
    {
        std::cout << "Loading model: " << path << std::endl;
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open model file");
        }

        // Read config
        file.read(reinterpret_cast<char *>(&config), 7 * sizeof(int));
        // Calculate derived
        config.head_dim = config.dim / config.n_heads;

        std::cout << "Config: dim=" << config.dim << " layers=" << config.n_layers << " heads=" << config.n_heads
                  << " vocab=" << config.vocab_size << std::endl;

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
        // We peek or trust file size. But simplest is to read.
        // If file ends early, it might be shared.
        // Standard llama2.c (stories15M) is not shared usually, but tinyllamas might be.
        // We try to read.
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
            // We can just copy embedding table or point to it.
            // Since we use vector, we copy.
            weights.lm_head = weights.token_embedding_table;
            std::cout << "Weights shared: lm_head <- token_embedding" << std::endl;
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

// ============================================================================
// 6. Main
// ============================================================================

int main(int argc, char **argv)
{
    // Setup argument parser
    ArgParser parser("nano-vllm: A minimal vLLM implementation in C++");
    parser.add_positional("model_path", "Path to the model file");
    parser.add_option<float>("-t", "Temperature for sampling", 1.0f);
    parser.add_option<float>("-p", "Top-p (nucleus) sampling parameter", 0.9f);
    parser.add_option<int>("-n", "Number of steps to generate", 256);
    parser.add_option<std::string>("-i", "Input prompt");

    if (!parser.parse(argc, argv)) {
        parser.print_usage();
        return 1;
    }

    // Get parsed arguments
    std::string model_path  = parser.get_positional();
    float       temperature = parser.get<float>("-t");
    float       topp        = parser.get<float>("-p");
    int         steps       = parser.get<int>("-n");
    std::string prompt      = parser.get<std::string>("-i");

    // 1. Load Model
    LlamaModel model;
    try {
        model.load(model_path);
    }
    catch (const std::exception &e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return 1;
    }

    // 2. Load Tokenizer (Assumes tokenizer.bin is in same dir as model)
    std::string tokenizer_path = model_path;
    size_t      last_slash     = tokenizer_path.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        tokenizer_path = tokenizer_path.substr(0, last_slash + 1) + "tokenizer.bin";
    }
    else {
        tokenizer_path = "tokenizer.bin";
    }

    Tokenizer tokenizer(tokenizer_path, model.config.vocab_size);
    Sampler   sampler(model.config.vocab_size, temperature, topp, std::time(nullptr));

    // 3. Encode Prompt
    std::vector<int> tokens = tokenizer.encode(prompt, true, false);

    std::cout << "\nGenerating...\n" << std::endl;
    std::cout << prompt;
    std::cout.flush();

    // 4. Generation Loop
    int pos = 0;

    // Prefill
    for (size_t i = 0; i < tokens.size() - 1; i++) {
        model.forward(tokens[i], pos);
        pos++;
    }
    int token = tokens.back();

    // Decode
    long start_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();

    for (int s = 0; s < steps; s++) {
        // Forward
        model.forward(token, pos);

        // Sample
        int next_token = sampler.sample(model.state.logits.data());

        // Output
        std::string piece = tokenizer.decode(next_token);
        std::cout << piece;
        std::cout.flush();

        // Advance
        token = next_token;
        pos++;

        if (pos >= model.config.max_seq_len)
            break;
    }

    long end_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();

    std::cout << "\n\nDone. (" << (double)(end_time - start_time) / 1000.0 << "s)" << std::endl;

    return 0;
}
