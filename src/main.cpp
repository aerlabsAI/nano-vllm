#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "core/model.hpp"
#include "core/sampler.hpp"
#include "core/tokenizer.hpp"
#include "utils/argparser.hpp"
#include "utils/logger.hpp"
#include "utils/path.hpp"

// ============================================================================
// Program Arguments Configuration
// ============================================================================

// NOTE:
// Macro to simplify tuple declaration (used once)
#define ARGS_LIST path, prompt, temperature, topp, steps, without_paged_attn

class Arguments : public ArgConfig<Arguments>
{
public:
    // NOTE: When adding a new argument, add it to BOTH places:
    // 1. Declaration below (supports multiple flags with {"-t", "--temperature"} syntax)
    //    Use nullptr as third parameter for REQUIRED arguments
    Arg<std::string> path{"path", "Path to model directory or model.bin file"};
    Arg<std::string> prompt{{"-i", "--prompt"}, "Input prompt", nullptr}; // REQUIRED
    Arg<float>       temperature{{"-t", "--temperature"}, "Temperature for sampling", 1.0f};
    Arg<float>       topp{{"-p", "--top-p"}, "Top-p (nucleus) sampling parameter", 0.9f};
    Arg<int>         steps{{"-n", "--steps"}, "Number of steps to generate", 256};
    Arg<bool>        without_paged_attn{"--without-paged-attn", "Disable PagedAttention", false};

    // 2. ARGS_LIST macro above (must match above order)
    decltype(std::tie(ARGS_LIST)) args_tuple = std::tie(ARGS_LIST);
};

#undef ARGS_LIST

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv)
{
    // Parse arguments using declarative configuration
    Arguments args;
    ArgParser parser("nano-vllm: A minimal vLLM implementation in C++");

    if (!args.parse(parser, argc, argv)) {
        return 1;
    }

    // 1. Resolve model and tokenizer paths
    std::string model_path, tokenizer_path;
    try {
        auto paths     = resolve_model_paths(args.path);
        model_path     = paths.first;
        tokenizer_path = paths.second;
    }
    catch (const std::exception &e) {
        LOG_ERROR("Failed to resolve paths: ", e.what());
        return 1;
    }

    // 2. Load Model
    LlamaModel model;
    try {
        model.load(model_path);

        // Configure PagedAttention based on CLI flag
        model.config.use_paged_attention = !args.without_paged_attn;

        if (model.config.use_paged_attention) {
            LOG_INFO("Using PagedAttention (block_size=", model.config.block_size, ")");
            model.initialize_paged_attention();
        }
        else {
            LOG_INFO("Using Standard Attention");
        }

        LOG_SUCCESS("Model loaded successfully");
    }
    catch (const std::exception &e) {
        LOG_ERROR("Error loading model: ", e.what());
        return 1;
    }

    // 3. Load Tokenizer
    Tokenizer tokenizer(tokenizer_path, model.config.vocab_size);
    LOG_SUCCESS("Tokenizer loaded successfully");

    Sampler sampler(model.config.vocab_size, args.temperature, args.topp, std::time(nullptr));

    // 3. Encode Prompt
    std::vector<int> tokens = tokenizer.encode(args.prompt, true, false);
    LOG_INFO("Encoded prompt into ", tokens.size(), " tokens");
    LOG_INFO("Starting generation with temperature=", args.temperature, " topp=", args.topp, " steps=", args.steps);
    LOG_INFO("Generating...");

    std::cout << "\n" << args.prompt.value;
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

    for (int s = 0; s < args.steps; s++) {
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

    double elapsed = (double)(end_time - start_time) / 1000.0;
    std::cout << std::endl;
    LOG_SUCCESS("Generation completed in ", elapsed, " seconds");

    return 0;
}
