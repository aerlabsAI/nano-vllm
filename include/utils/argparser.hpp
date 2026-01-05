#pragma once

#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "logger.hpp"

// ============================================================================
// Argument Parser for nano-vllm (argparse-style)
// ============================================================================

class ArgParser
{
public:
    using OptionValue = std::variant<int, float, std::string>;

    ArgParser(const std::string &description = "")
        : description_(description)
    {
    }

    // Add positional argument (e.g., model_path)
    void add_positional(const std::string &name, const std::string &help)
    {
        positional_name_ = name;
        positional_help_ = help;
    }

    // Add optional argument with default value
    template <typename T> void add_option(const std::string &flag, const std::string &help, T default_value)
    {
        Option opt;
        opt.default_value = default_value;
        opt.value         = default_value;
        opt.help          = help;
        opt.required      = false;
        opt.is_flag       = false;
        opt.type_name     = get_type_name<T>();
        options_[flag]    = opt;
    }

    // Add required argument (no default value)
    template <typename T> void add_option(const std::string &flag, const std::string &help)
    {
        Option opt;
        opt.default_value = get_default_for_type<T>();
        opt.value         = std::nullopt;
        opt.help          = help;
        opt.required      = true;
        opt.is_flag       = false;
        opt.type_name     = get_type_name<T>();
        options_[flag]    = opt;
    }

    // Add boolean flag (no value required)
    void add_flag(const std::string &flag, const std::string &help) { flags_[flag] = {help, false}; }

    // Parse command-line arguments
    bool parse(int argc, char **argv)
    {
        program_name_ = argv[0];

        if (argc < 2 && !positional_name_.empty()) {
            return false;
        }

        // Parse positional argument
        if (!positional_name_.empty()) {
            positional_value_ = argv[1];
        }

        // Parse optional arguments and flags
        for (int i = 2; i < argc; i++) {
            std::string arg = argv[i];

            // Check if it's a flag
            auto flag_it = flags_.find(arg);
            if (flag_it != flags_.end()) {
                flag_it->second.value = true;
                continue;
            }

            // Check if it's an option
            auto it = options_.find(arg);
            if (it != options_.end()) {
                if (i + 1 >= argc) {
                    LOG_ERROR("Option ", arg, " requires a value");
                    return false;
                }

                try {
                    it->second.value = parse_value(argv[++i], it->second.default_value);
                }
                catch (const std::exception &e) {
                    LOG_ERROR("Invalid value for ", arg, ": ", e.what());
                    return false;
                }
            }
        }

        // Check required arguments
        std::vector<std::string> missing;
        for (const auto &[flag, opt] : options_) {
            if (opt.required && !opt.value.has_value()) {
                missing.push_back(flag);
            }
        }

        if (!missing.empty()) {
            std::string missing_str = "Missing required arguments: ";
            for (size_t i = 0; i < missing.size(); i++) {
                missing_str += missing[i];
                if (i < missing.size() - 1)
                    missing_str += ", ";
            }
            LOG_ERROR(missing_str);
            return false;
        }

        return true;
    }

    // Get positional argument value
    std::string get_positional() const { return positional_value_; }

    // Get optional argument value
    template <typename T> T get(const std::string &flag) const
    {
        auto it = options_.find(flag);
        if (it == options_.end()) {
            throw std::runtime_error("Unknown option: " + flag);
        }
        if (!it->second.value.has_value()) {
            throw std::runtime_error("Option " + flag + " was not provided");
        }
        return std::get<T>(it->second.value.value());
    }

    // Get flag value
    bool get_flag(const std::string &flag) const
    {
        auto it = flags_.find(flag);
        if (it == flags_.end()) {
            throw std::runtime_error("Unknown flag: " + flag);
        }
        return it->second.value;
    }

    // Print usage information
    void print_usage() const
    {
        std::cout << "Usage: " << program_name_;

        if (!positional_name_.empty()) {
            std::cout << " <" << positional_name_ << ">";
        }

        std::cout << " [options]" << std::endl;

        if (!description_.empty()) {
            std::cout << "\n" << description_ << std::endl;
        }

        if (!positional_name_.empty()) {
            std::cout << "\nPositional arguments:" << std::endl;
            std::cout << "  " << positional_name_ << "\t\t" << positional_help_ << std::endl;
        }

        if (!options_.empty()) {
            std::cout << "\nOptional arguments:" << std::endl;
            for (const auto &[flag, opt] : options_) {
                std::cout << "  " << flag << " <" << opt.type_name << ">\t" << opt.help;
                if (opt.required) {
                    std::cout << " [REQUIRED]";
                }
                else {
                    std::cout << " (default: " << value_to_string(opt.default_value) << ")";
                }
                std::cout << std::endl;
            }
        }

        if (!flags_.empty()) {
            std::cout << "\nFlags:" << std::endl;
            for (const auto &[flag, flag_info] : flags_) {
                std::cout << "  " << flag << "\t\t" << flag_info.help << std::endl;
            }
        }
    }

private:
    struct Option
    {
        std::optional<OptionValue> value;
        OptionValue                default_value;
        std::string                help;
        std::string                type_name;
        bool                       required;
        bool                       is_flag;
    };

    struct Flag
    {
        std::string help;
        bool        value;
    };

    std::string                   program_name_;
    std::string                   description_;
    std::string                   positional_name_;
    std::string                   positional_help_;
    std::string                   positional_value_;
    std::map<std::string, Option> options_;
    std::map<std::string, Flag>   flags_;

    // Get type name string
    template <typename T> std::string get_type_name() const
    {
        if constexpr (std::is_same_v<T, int>) {
            return "int";
        }
        else if constexpr (std::is_same_v<T, float>) {
            return "float";
        }
        else if constexpr (std::is_same_v<T, std::string>) {
            return "string";
        }
        return "unknown";
    }

    // Get default value for type (used for required options)
    template <typename T> OptionValue get_default_for_type() const
    {
        if constexpr (std::is_same_v<T, int>) {
            return 0;
        }
        else if constexpr (std::is_same_v<T, float>) {
            return 0.0f;
        }
        else if constexpr (std::is_same_v<T, std::string>) {
            return std::string("");
        }
    }

    // Parse string to appropriate type based on default value type
    OptionValue parse_value(const std::string &str, const OptionValue &default_val)
    {
        return std::visit(
            [&str](auto &&arg) -> OptionValue {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, int>) {
                    return std::stoi(str);
                }
                else if constexpr (std::is_same_v<T, float>) {
                    return std::stof(str);
                }
                else if constexpr (std::is_same_v<T, std::string>) {
                    return str;
                }
            },
            default_val);
    }

    // Convert value to string for printing
    std::string value_to_string(const OptionValue &val) const
    {
        return std::visit(
            [](auto &&arg) -> std::string {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, std::string>) {
                    return "\"" + arg + "\"";
                }
                else {
                    return std::to_string(arg);
                }
            },
            val);
    }
};
