#include <string>

#include "core/model.hpp"
#include "core/runner.hpp"
#include "core/tokenizer.hpp"
#include "utils/argparser.hpp"
#include "utils/benchmark_result.hpp"
#include "utils/comparison.hpp"
#include "utils/logger.hpp"
#include "utils/path.hpp"

// ============================================================================
// Program Arguments Configuration
// ============================================================================

#define ARGS_LIST path, prompt, input_json, max_batch_size, async_mode, temperature, topp, steps, without_paged_attn, save_results

class Arguments : public ArgConfig<Arguments>
{
public:
    Arg<std::string> path{"path", "Path to model directory or model.bin file"};
    Arg<std::string> prompt{{"-i", "--prompt"}, "Input prompt", ""};
    Arg<std::string> input_json{"--input-json", "Path to JSON file with benchmark requests", ""};
    Arg<int>         max_batch_size{{"-b", "--max-batch-size"}, "Maximum batch size for continuous batching", 1};
    Arg<bool>        async_mode{"--async", "Enable async request submission (simulate dynamic arrivals)", false};
    Arg<float>       temperature{{"-t", "--temperature"}, "Temperature for sampling", 1.0f};
    Arg<float>       topp{{"-p", "--top-p"}, "Top-p (nucleus) sampling parameter", 0.9f};
    Arg<int>         steps{{"-n", "--steps"}, "Number of steps to generate", 256};
    Arg<bool>        without_paged_attn{"--without-paged-attn", "Disable PagedAttention", false};
    Arg<std::string> save_results{"--save-results", "Save benchmark results to JSON file", ""};

    decltype(std::tie(ARGS_LIST)) args_tuple = std::tie(ARGS_LIST);
};

#undef ARGS_LIST

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv)
{
    // ---- Comparison mode (no model needed) ----
    // Pre-scan argv since comparison mode doesn't use the positional path arg
    std::string compare_a, compare_b, output_format = "table";
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--compare-a" && i + 1 < argc)      { compare_a = argv[++i]; }
        else if (arg == "--compare-b" && i + 1 < argc)  { compare_b = argv[++i]; }
        else if (arg == "--output-format" && i + 1 < argc) { output_format = argv[++i]; }
    }

    if (!compare_a.empty() && !compare_b.empty()) {
        try {
            BenchmarkResult a = BenchmarkResult::load(compare_a);
            BenchmarkResult b = BenchmarkResult::load(compare_b);
            Comparison cmp(a, b);

            if (output_format == "json") {
                std::cout << cmp.to_json();
            }
            else {
                cmp.print_table();
            }
            return 0;
        }
        catch (const std::exception &e) {
            LOG_ERROR("Comparison failed: ", e.what());
            return 1;
        }
    }

    // ---- Inference mode ----
    Arguments args;
    ArgParser parser("nano-vllm: A minimal vLLM implementation in C++");

    if (!args.parse(parser, argc, argv)) {
        return 1;
    }

    bool has_prompt     = !args.prompt.value.empty();
    bool has_input_json = !args.input_json.value.empty();

    if (!has_prompt && !has_input_json) {
        LOG_ERROR("Either --prompt or --input-json must be provided");
        parser.print_usage();
        return 1;
    }

    if (has_prompt && has_input_json) {
        LOG_ERROR("Cannot use both --prompt and --input-json");
        return 1;
    }

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

    LlamaModel model;
    try {
        model.load(model_path);
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

    Tokenizer tokenizer(tokenizer_path, model.config.vocab_size);
    LOG_SUCCESS("Tokenizer loaded successfully");

    BenchmarkResult result;
    try {
        if (has_input_json) {
            result = run_json_benchmark(model, tokenizer, args.input_json, args.max_batch_size, args.async_mode);
        }
        else {
            result = run_single_prompt(model, tokenizer, args.prompt, args.temperature, args.topp, args.steps);
        }
    }
    catch (const std::exception &e) {
        LOG_ERROR("Benchmark failed: ", e.what());
        return 1;
    }

    if (!args.save_results.value.empty()) {
        if (result.save(args.save_results)) {
            LOG_SUCCESS("Results saved to ", args.save_results.value);
        }
        else {
            LOG_ERROR("Failed to save results to ", args.save_results.value);
            return 1;
        }
    }

    return 0;
}
