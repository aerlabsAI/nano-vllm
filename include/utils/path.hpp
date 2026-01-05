#pragma once

#include <filesystem>
#include <string>
#include <utility>

#include "logger.hpp"

namespace fs = std::filesystem;

// ============================================================================
// Path Resolution Functions
// ============================================================================

// Resolve model and tokenizer paths from user input
// If path is a directory, look for model.bin and tokenizer.bin inside
// If path is a file, use it as model and look for tokenizer.bin in same directory
inline std::pair<std::string, std::string> resolve_model_paths(const std::string &input_path)
{
    fs::path    p(input_path);
    std::string model_path;
    std::string tokenizer_path;

    if (fs::is_directory(p)) {
        // TODO: json, safetensors, ...

        // Input is a directory
        model_path     = (p / "model.bin").string();
        tokenizer_path = (p / "tokenizer.bin").string();

        if (!fs::exists(model_path)) {
            LOG_ERROR("model.bin not found in directory: ", input_path);
            throw std::runtime_error("model.bin not found in: " + input_path);
        }
        if (!fs::exists(tokenizer_path)) {
            LOG_ERROR("tokenizer.bin not found in directory: ", input_path);
            throw std::runtime_error("tokenizer.bin not found in: " + input_path);
        }

        LOG_INFO("Found model.bin and tokenizer.bin in: ", input_path);
    }
    else if (fs::exists(p) && fs::is_regular_file(p)) {
        // Input is a file
        model_path             = p.string();
        fs::path parent        = p.parent_path();
        fs::path tokenizer_bin = parent / "tokenizer.bin";

        if (parent.empty()) {
            tokenizer_bin = "tokenizer.bin";
        }

        tokenizer_path = tokenizer_bin.string();

        if (!fs::exists(tokenizer_path)) {
            LOG_WARNING("tokenizer.bin not found in: ", parent.string(), ", trying current directory");
            tokenizer_path = "tokenizer.bin";
        }
    }
    else {
        LOG_ERROR("Path does not exist: ", input_path);
        throw std::runtime_error("Path does not exist: " + input_path);
    }

    return {model_path, tokenizer_path};
}
