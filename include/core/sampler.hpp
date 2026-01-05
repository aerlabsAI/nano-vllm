#pragma once

#include <algorithm>
#include <random>
#include <vector>

#include "ops/activation.hpp"

// ============================================================================
// Sampler - Temperature and Top-p Sampling
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
