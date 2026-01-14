#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "../../include/core/sampler.hpp"
#include "../../include/core/tokenizer.hpp"
#include "../../include/utils/argparser.hpp"
#include "../../include/utils/logger.hpp"
#include "../../include/utils/path.hpp"
#include "model_chunked.hpp"

// Arguments configuration using ArgConfig
struct ChunkedPrefillArgs : public ArgConfig<ChunkedPrefillArgs>
{
    Arg<std::string> model_path      = {"path", "Path to model directory or model.bin file"};
    Arg<float>       temperature     = {{"-t", "--temperature"}, "Temperature for sampling", 1.0f};
    Arg<float>       topp            = {{"-p", "--top-p"}, "Top-p (nucleus) sampling parameter", 0.9f};
    Arg<int>         steps           = {{"-n", "--steps"}, "Number of steps to generate", 256};
    Arg<int>         chunk_size      = {"--chunk-size", "Chunk size for prefill", 16};
    Arg<std::string> prompt          = {{"-i", "--input"}, "Input prompt", nullptr};
    Arg<bool>        benchmark       = {"--benchmark", "Show detailed metrics", false};

    auto args_tuple = std::tie(model_path, temperature, topp, steps, chunk_size, prompt, benchmark);
};

int main(int argc, char **argv)
{
    ChunkedPrefillArgs args;
    ArgParser          parser("nano-vllm: Chunked Prefill Implementation");

    if (!args.parse(parser, argc, argv)) {
        return 1;
    }

    std::string input_path = args.model_path.get();

    std::string model_path, tokenizer_path;
    try {
        auto paths     = resolve_model_paths(input_path);
        model_path     = paths.first;
        tokenizer_path = paths.second;
    }
    catch (const std::exception &e) {
        LOG_ERROR("Failed to resolve paths: ", e.what());
        return 1;
    }

    LlamaModelChunked model;
    try {
        model.load(model_path);
        LOG_SUCCESS("Model loaded successfully");
    }
    catch (const std::exception &e) {
        LOG_ERROR("Error loading model: ", e.what());
        return 1;
    }

    Tokenizer tokenizer(tokenizer_path, model.config.vocab_size);
    LOG_SUCCESS("Tokenizer loaded successfully");

    Sampler sampler(model.config.vocab_size, args.temperature, args.topp, std::time(nullptr));

    std::vector<int> tokens = tokenizer.encode(args.prompt, true, false);
    LOG_INFO("Encoded prompt into ", tokens.size(), " tokens");
    LOG_INFO("Chunk size: ", args.chunk_size.get());

    std::cout << "\n" << args.prompt.get();
    std::cout.flush();

    auto prefill_tokens = std::vector<int>(tokens.begin(), tokens.end() - 1);
    auto metrics        = model.prefill_chunked(prefill_tokens, args.chunk_size);

    if (args.benchmark) {
        LOG_INFO("=== Prefill Metrics ===");
        LOG_INFO("Total tokens: ", metrics.total_tokens);
        LOG_INFO("Num chunks: ", metrics.num_chunks);
        LOG_INFO("Chunk size: ", metrics.chunk_size);
        LOG_INFO("Total time: ", metrics.total_time_ms, " ms");
        LOG_INFO("Avg chunk time: ", metrics.avg_chunk_time_ms, " ms");
        LOG_INFO("Throughput: ", metrics.tokens_per_second(), " tokens/sec");
    }

    int token = tokens.back();
    int pos   = static_cast<int>(tokens.size()) - 1;

    auto decode_start = std::chrono::high_resolution_clock::now();

    for (int s = 0; s < args.steps; s++) {
        model.forward(token, pos);

        int next_token = sampler.sample(model.state.logits.data());

        std::string piece = tokenizer.decode(next_token);
        std::cout << piece;
        std::cout.flush();

        token = next_token;
        pos++;

        if (pos >= model.config.max_seq_len)
            break;
    }

    auto decode_end  = std::chrono::high_resolution_clock::now();
    auto decode_time = std::chrono::duration<double>(decode_end - decode_start).count();

    std::cout << std::endl;
    LOG_SUCCESS("Generation completed");

    if (args.benchmark) {
        LOG_INFO("=== Decode Metrics ===");
        LOG_INFO("Decode time: ", decode_time, " seconds");
        LOG_INFO("Total time: ", (metrics.total_time_ms / 1000.0 + decode_time), " seconds");
    }

    return 0;
}
