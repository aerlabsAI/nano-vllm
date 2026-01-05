#pragma once

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils/logger.hpp"

// ============================================================================
// Tokenizer - BPE (Byte Pair Encoding) Implementation
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
        LOG_INFO("Loading tokenizer: ", path);
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            LOG_ERROR("Failed to open tokenizer: ", path);
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
