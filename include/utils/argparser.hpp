#pragma once

#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "logger.hpp"

// Forward declaration
class ArgParser;

// ============================================================================
// Argument Type Enum
// ============================================================================

enum class ArgType { Positional, Option, Flag };

// ============================================================================
// Argument Wrapper Class
// ============================================================================

template <typename T> struct Arg
{
    T                        value;
    std::vector<std::string> flags; // Support multiple flags (e.g., {"-t", "--temperature"})
    std::string              help;
    T                        default_value;
    ArgType                  type;
    bool                     required = false;

    // Constructor for positional argument
    Arg(const std::string &name, const std::string &help_text)
        : flags({name})
        , help(help_text)
        , type(ArgType::Positional)
        , value(T{})
        , default_value(T{})
        , required(false)
    {
    }

    // Constructor for option with single flag (with default value)
    Arg(const std::string &flag_name, const std::string &help_text, T default_val)
        : flags({flag_name})
        , help(help_text)
        , default_value(default_val)
        , value(default_val)
        , type(ArgType::Option)
        , required(false)
    {
    }

    // Constructor for option with multiple flags (with default value)
    Arg(const std::initializer_list<std::string> &flag_names, const std::string &help_text, T default_val)
        : flags(flag_names)
        , help(help_text)
        , default_value(default_val)
        , value(default_val)
        , type(ArgType::Option)
        , required(false)
    {
    }

    // Constructor for REQUIRED option with single flag (no default value)
    Arg(const std::string &flag_name, const std::string &help_text, std::nullptr_t)
        : flags({flag_name})
        , help(help_text)
        , default_value(T{})
        , value(T{})
        , type(ArgType::Option)
        , required(true)
    {
    }

    // Constructor for REQUIRED option with multiple flags (no default value)
    Arg(const std::initializer_list<std::string> &flag_names, const std::string &help_text, std::nullptr_t)
        : flags(flag_names)
        , help(help_text)
        , default_value(T{})
        , value(T{})
        , type(ArgType::Option)
        , required(true)
    {
    }

    // Implicit conversion to T
    operator T() const { return value; }

    // Explicit getter
    T get() const { return value; }

    // Get primary flag (first one)
    std::string get_primary_flag() const { return flags.empty() ? "" : flags[0]; }
};

// Specialization for boolean flags
template <> struct Arg<bool>
{
    bool                     value;
    std::vector<std::string> flags; // Support multiple flags (e.g., {"-v", "--verbose"})
    std::string              help;
    bool                     default_value;
    ArgType                  type = ArgType::Flag;

    // Constructor with single flag
    Arg(const std::string &flag_name, const std::string &help_text, bool default_val = false)
        : flags({flag_name})
        , help(help_text)
        , default_value(default_val)
        , value(default_val)
        , type(ArgType::Flag)
    {
    }

    // Constructor with multiple flags (e.g., {"-v", "--verbose"})
    Arg(const std::initializer_list<std::string> &flag_names, const std::string &help_text, bool default_val = false)
        : flags(flag_names)
        , help(help_text)
        , default_value(default_val)
        , value(default_val)
        , type(ArgType::Flag)
    {
    }

    operator bool() const { return value; }
    bool get() const { return value; }

    // Get primary flag (first one)
    std::string get_primary_flag() const { return flags.empty() ? "" : flags[0]; }
};

// ============================================================================
// Argument Configuration Base Class (CRTP) - Forward Declaration
// ============================================================================

template <typename Derived> class ArgConfig
{
protected:
    std::vector<std::function<void(ArgParser &)>> configurators_;

    // Register a single argument
    template <typename T> void register_arg(Arg<T> &arg);
    void                       register_arg(Arg<bool> &arg);

    // Register multiple arguments at once
    template <typename... Args> void register_args(Args &...args) { (register_arg(args), ...); }

    // Helper to register arguments from a tuple
    template <typename Tuple> void register_from_tuple(Tuple &args_tuple)
    {
        std::apply([this](auto &...args) { register_args(args...); }, args_tuple);
    }

    // Helper to print a single argument
    template <typename T> void print_arg(const Arg<T> &arg)
    {
        std::string flags_str;
        if (!arg.flags.empty()) {
            flags_str = arg.flags[0];
            for (size_t i = 1; i < arg.flags.size(); i++) {
                flags_str += ", " + arg.flags[i];
            }
        }

        if (arg.required) {
            LOG_INFO("  ", flags_str, ": ", arg.value);
        }
        else {
            bool is_default = (arg.value == arg.default_value);
            LOG_INFO("  ", flags_str, ": ", arg.value, is_default ? " (default)" : "");
        }
    }

    // Specialization for bool
    void print_arg(const Arg<bool> &arg)
    {
        std::string flags_str;
        if (!arg.flags.empty()) {
            flags_str = arg.flags[0];
            for (size_t i = 1; i < arg.flags.size(); i++) {
                flags_str += ", " + arg.flags[i];
            }
        }

        bool is_default = (arg.value == arg.default_value);
        LOG_INFO("  ", flags_str, ": ", arg.value ? "true" : "false", is_default ? " (default)" : "");
    }

    // Helper to print all arguments from a tuple
    template <typename Tuple> void print_all_args(Tuple &args_tuple)
    {
        LOG_INFO("=== Parsed Arguments ===");
        std::apply([this](auto &...args) { (print_arg(args), ...); }, args_tuple);
        LOG_INFO("========================");
    }

public:
    virtual ~ArgConfig() = default;
    bool parse(ArgParser &parser, int argc, char **argv);
};

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
    template <typename T> void add_positional(const std::string &name, const std::string &help, T *value_ptr)
    {
        positional_name_ = name;
        positional_help_ = help;
        if constexpr (std::is_same_v<T, std::string>) {
            positional_ptr_ = value_ptr;
        }
    }

    // Add optional argument with multiple flags (e.g., {"-t", "--temperature"})
    template <typename T>
    void add_option(const std::vector<std::string> &flags, const std::string &help, T default_value, T *value_ptr)
    {
        if (flags.empty())
            return;

        std::string primary_flag = flags[0];
        Option      opt;
        opt.default_value = default_value;
        opt.value         = default_value;
        opt.help          = help;
        opt.required      = false;
        opt.is_flag       = false;
        opt.type_name     = get_type_name<T>();
        opt.flags         = flags; // Store all flags
        opt.extract_fn    = [this, primary_flag, value_ptr]() {
            if (value_ptr) {
                *value_ptr = this->get<T>(primary_flag);
            }
        };

        // Register all flags to point to same option
        for (const auto &flag : flags) {
            options_[flag] = opt;
        }
    }

    // Add required argument with multiple flags
    template <typename T> void add_option(const std::vector<std::string> &flags, const std::string &help, T *value_ptr)
    {
        if (flags.empty())
            return;

        std::string primary_flag = flags[0];
        Option      opt;
        opt.default_value = get_default_for_type<T>();
        opt.value         = std::nullopt;
        opt.help          = help;
        opt.required      = true;
        opt.is_flag       = false;
        opt.type_name     = get_type_name<T>();
        opt.flags         = flags;
        opt.extract_fn    = [this, primary_flag, value_ptr]() {
            if (value_ptr) {
                *value_ptr = this->get<T>(primary_flag);
            }
        };

        for (const auto &flag : flags) {
            options_[flag] = opt;
        }
    }

    // Add boolean flag with multiple flags (e.g., {"-v", "--verbose"})
    void add_flag(const std::vector<std::string> &flags, const std::string &help, bool default_value, bool *value_ptr)
    {
        if (flags.empty())
            return;

        std::string primary_flag = flags[0];
        Flag        flag_info    = {help,
                                    default_value,
                                    default_value,
                                    [this, primary_flag, value_ptr]() {
                              if (value_ptr) {
                                  *value_ptr = this->get_flag(primary_flag);
                              }
                          },
                                    flags};

        for (const auto &flag : flags) {
            flags_[flag] = flag_info;
        }
    }

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
                    auto parsed_value = parse_value(argv[++i], it->second.default_value);
                    // Set value for all aliases of this option
                    for (const auto &flag : it->second.flags) {
                        options_[flag].value = parsed_value;
                    }
                }
                catch (const std::exception &e) {
                    LOG_ERROR("Invalid value for ", arg, ": ", e.what());
                    return false;
                }
            }
        }

        // Check required arguments (only check primary flags to avoid duplicates)
        std::vector<std::string> missing;
        std::set<std::string>    checked;
        for (const auto &[flag, opt] : options_) {
            if (opt.required && !opt.value.has_value()) {
                // Only add primary flag (first one) to avoid duplicates
                std::string primary = opt.flags.empty() ? flag : opt.flags[0];
                if (checked.find(primary) == checked.end()) {
                    // Show all aliases for clarity
                    std::string all_flags;
                    if (!opt.flags.empty()) {
                        for (size_t i = 0; i < opt.flags.size(); i++) {
                            all_flags += opt.flags[i];
                            if (i < opt.flags.size() - 1)
                                all_flags += "/";
                        }
                    }
                    else {
                        all_flags = flag;
                    }
                    missing.push_back(all_flags);
                    checked.insert(primary);
                }
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

        // Auto-extract values to target pointers
        if (positional_ptr_) {
            *positional_ptr_ = positional_value_;
        }
        for (auto &[flag, opt] : options_) {
            if (opt.extract_fn) {
                opt.extract_fn();
            }
        }
        for (auto &[flag, flag_info] : flags_) {
            if (flag_info.extract_fn) {
                flag_info.extract_fn();
            }
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

    // Set program name (used by ArgConfig before parsing)
    void set_program_name(const std::string &name) { program_name_ = name; }

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
            std::set<std::string> printed_options;
            for (const auto &[flag, opt] : options_) {
                // Skip if already printed (for multi-flag options)
                if (printed_options.count(opt.flags.empty() ? flag : opt.flags[0]))
                    continue;

                // Print all aliases together
                std::cout << "  ";
                if (!opt.flags.empty()) {
                    for (size_t i = 0; i < opt.flags.size(); i++) {
                        std::cout << opt.flags[i];
                        if (i < opt.flags.size() - 1)
                            std::cout << ", ";
                    }
                    printed_options.insert(opt.flags[0]);
                }
                else {
                    std::cout << flag;
                    printed_options.insert(flag);
                }

                std::cout << " <" << opt.type_name << ">\t" << opt.help;
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
            std::set<std::string> printed_flags;
            for (const auto &[flag, flag_info] : flags_) {
                // Skip if already printed
                if (printed_flags.count(flag_info.flags.empty() ? flag : flag_info.flags[0]))
                    continue;

                // Print all aliases together
                std::cout << "  ";
                if (!flag_info.flags.empty()) {
                    for (size_t i = 0; i < flag_info.flags.size(); i++) {
                        std::cout << flag_info.flags[i];
                        if (i < flag_info.flags.size() - 1)
                            std::cout << ", ";
                    }
                    printed_flags.insert(flag_info.flags[0]);
                }
                else {
                    std::cout << flag;
                    printed_flags.insert(flag);
                }

                std::cout << "\t\t" << flag_info.help << " (default: " << (flag_info.default_value ? "true" : "false")
                          << ")" << std::endl;
            }
        }

        // Always show help option
        std::cout << "\nHelp:" << std::endl;
        std::cout << "  -h, --help\t\tShow this help message and exit" << std::endl;
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
        std::function<void()>      extract_fn;
        std::vector<std::string>   flags; // All flag aliases
    };

    struct Flag
    {
        std::string              help;
        bool                     value;         // Current value
        bool                     default_value; // Default value
        std::function<void()>    extract_fn;
        std::vector<std::string> flags; // All flag aliases
    };

    std::string                   program_name_;
    std::string                   description_;
    std::string                   positional_name_;
    std::string                   positional_help_;
    std::string                   positional_value_;
    std::string                  *positional_ptr_ = nullptr;
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

// ============================================================================
// ArgConfig Method Definitions
// ============================================================================

template <typename Derived> template <typename T> void ArgConfig<Derived>::register_arg(Arg<T> &arg)
{
    configurators_.push_back([&arg](ArgParser &parser) {
        if (arg.type == ArgType::Positional) {
            parser.add_positional(arg.get_primary_flag(), arg.help, &arg.value);
        }
        else if (arg.type == ArgType::Option) {
            if (arg.required) {
                parser.add_option(arg.flags, arg.help, &arg.value);
            }
            else {
                parser.add_option(arg.flags, arg.help, arg.default_value, &arg.value);
            }
        }
    });
}

template <typename Derived> void ArgConfig<Derived>::register_arg(Arg<bool> &arg)
{
    configurators_.push_back(
        [&arg](ArgParser &parser) { parser.add_flag(arg.flags, arg.help, arg.default_value, &arg.value); });
}

template <typename Derived> bool ArgConfig<Derived>::parse(ArgParser &parser, int argc, char **argv)
{
    // Auto-detect and register args_tuple from derived class
    auto &derived = static_cast<Derived &>(*this);
    register_from_tuple(derived.args_tuple);

    for (auto &configurator : configurators_) {
        configurator(parser);
    }

    // Set program name first
    parser.set_program_name(argc > 0 ? argv[0] : "");

    // Check for --help or -h first
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            parser.print_usage();
            return false;
        }
    }

    if (!parser.parse(argc, argv)) {
        parser.print_usage();
        return false;
    }

    // Print all parsed arguments
    print_all_args(derived.args_tuple);

    return true;
}
