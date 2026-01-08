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

int main(int argc, char **argv)
{
    ArgParser parser("nano-vllm: Chunked Prefill Implementation");
    parser.add_positional("path", "Path to model directory or model.bin file");
    parser.add_option<float>("-t", "Temperature for sampling", 1.0f);
    parser.add_option<float>("-p", "Top-p (nucleus) sampling parameter", 0.9f);
    parser.add_option<int>("-n", "Number of steps to generate", 256);
    parser.add_option<int>("--chunk-size", "Chunk size for prefill", 16);
    parser.add_option<std::string>("-i", "Input prompt");
    parser.add_flag("--benchmark", "Show detailed metrics");

    if (!parser.parse(argc, argv)) {
        parser.print_usage();
        return 1;
    }

    std::string input_path  = parser.get_positional();
    float       temperature = parser.get<float>("-t");
    float       topp        = parser.get<float>("-p");
    int         steps       = parser.get<int>("-n");
    int         chunk_size  = parser.get<int>("--chunk-size");
    std::string prompt      = parser.get<std::string>("-i");
    bool        benchmark   = parser.has_flag("--benchmark");

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

    Sampler sampler(model.config.vocab_size, temperature, topp, std::time(nullptr));

    std::vector<int> tokens = tokenizer.encode(prompt, true, false);
    LOG_INFO("Encoded prompt into ", tokens.size(), " tokens");
    LOG_INFO("Chunk size: ", chunk_size);

    std::cout << "\n" << prompt;
    std::cout.flush();

    auto prefill_tokens = std::vector<int>(tokens.begin(), tokens.end() - 1);
    auto metrics        = model.prefill_chunked(prefill_tokens, chunk_size);

    if (benchmark) {
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

    for (int s = 0; s < steps; s++) {
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

    if (benchmark) {
        LOG_INFO("=== Decode Metrics ===");
        LOG_INFO("Decode time: ", decode_time, " seconds");
        LOG_INFO("Total time: ", (metrics.total_time_ms / 1000.0 + decode_time), " seconds");
    }

    return 0;
}
