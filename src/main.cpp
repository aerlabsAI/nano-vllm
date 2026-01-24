#include <string>

#include "core/model.hpp"
#include "core/runner.hpp"
#include "core/tokenizer.hpp"
#include "utils/argparser.hpp"
#include "utils/logger.hpp"
#include "utils/path.hpp"

// ============================================================================
// Program Arguments Configuration
// ============================================================================

#define ARGS_LIST path, prompt, input_json, temperature, topp, steps, without_paged_attn

class Arguments : public ArgConfig<Arguments>
{
public:
    Arg<std::string> path{"path", "Path to model directory or model.bin file"};
    Arg<std::string> prompt{{"-i", "--prompt"}, "Input prompt", ""};
    Arg<std::string> input_json{"--input-json", "Path to JSON file with benchmark requests", ""};
    Arg<float>       temperature{{"-t", "--temperature"}, "Temperature for sampling", 1.0f};
    Arg<float>       topp{{"-p", "--top-p"}, "Top-p (nucleus) sampling parameter", 0.9f};
    Arg<int>         steps{{"-n", "--steps"}, "Number of steps to generate", 256};
    Arg<bool>        without_paged_attn{"--without-paged-attn", "Disable PagedAttention", false};

    decltype(std::tie(ARGS_LIST)) args_tuple = std::tie(ARGS_LIST);
};

#undef ARGS_LIST

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv)
{
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

    if (has_input_json) {
        return run_json_benchmark(model, tokenizer, args.input_json);
    }
    else {
        return run_single_prompt(model, tokenizer, args.prompt, args.temperature, args.topp, args.steps);
    }
}
