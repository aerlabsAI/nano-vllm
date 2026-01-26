#pragma once

#include <cctype>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "scheduler/request.hpp"

// ============================================================================
// Simple JSON Parser - Minimal implementation for benchmark input
// ============================================================================

namespace json {

struct JsonObject;
using JsonObjectPtr = std::unique_ptr<JsonObject>;
using JsonArray     = std::vector<JsonObject>;
using JsonValue     = std::variant<std::nullptr_t, bool, double, std::string, JsonArray, JsonObjectPtr>;

struct JsonObject
{
    std::unordered_map<std::string, JsonValue> data;

    bool has(const std::string &key) const { return data.find(key) != data.end(); }

    std::string get_string(const std::string &key, const std::string &default_val = "") const
    {
        auto it = data.find(key);
        if (it == data.end())
            return default_val;
        if (auto *val = std::get_if<std::string>(&it->second))
            return *val;
        return default_val;
    }

    double get_number(const std::string &key, double default_val = 0.0) const
    {
        auto it = data.find(key);
        if (it == data.end())
            return default_val;
        if (auto *val = std::get_if<double>(&it->second))
            return *val;
        return default_val;
    }

    int get_int(const std::string &key, int default_val = 0) const
    {
        return static_cast<int>(get_number(key, static_cast<double>(default_val)));
    }

    float get_float(const std::string &key, float default_val = 0.0f) const
    {
        return static_cast<float>(get_number(key, static_cast<double>(default_val)));
    }

    bool get_bool(const std::string &key, bool default_val = false) const
    {
        auto it = data.find(key);
        if (it == data.end())
            return default_val;
        if (auto *val = std::get_if<bool>(&it->second))
            return *val;
        return default_val;
    }

    const JsonArray &get_array(const std::string &key) const
    {
        static JsonArray empty;
        auto             it = data.find(key);
        if (it == data.end())
            return empty;
        if (auto *val = std::get_if<JsonArray>(&it->second))
            return *val;
        return empty;
    }

    const JsonObject &get_object(const std::string &key) const
    {
        static JsonObject empty;
        auto              it = data.find(key);
        if (it == data.end())
            return empty;
        if (auto *val = std::get_if<JsonObjectPtr>(&it->second))
            return **val;
        return empty;
    }
};

class JsonParser
{
public:
    JsonObject parse(const std::string &json_str)
    {
        pos_ = 0;
        str_ = json_str;
        skip_whitespace();
        return parse_object();
    }

    JsonObject parse_file(const std::string &filepath)
    {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open JSON file: " + filepath);
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        return parse(buffer.str());
    }

private:
    std::string str_;
    size_t      pos_ = 0;

    char current() const { return pos_ < str_.size() ? str_[pos_] : '\0'; }
    char advance() { return str_[pos_++]; }
    bool at_end() const { return pos_ >= str_.size(); }

    void skip_whitespace()
    {
        while (!at_end() && std::isspace(current()))
            advance();
    }

    void expect(char c)
    {
        skip_whitespace();
        if (current() != c) {
            throw std::runtime_error(std::string("Expected '") + c + "' but got '" + current() + "'");
        }
        advance();
    }

    std::string parse_string()
    {
        skip_whitespace();
        expect('"');
        std::string result;
        while (!at_end() && current() != '"') {
            if (current() == '\\') {
                advance();
                switch (current()) {
                case '"':
                    result += '"';
                    break;
                case '\\':
                    result += '\\';
                    break;
                case 'n':
                    result += '\n';
                    break;
                case 't':
                    result += '\t';
                    break;
                case 'r':
                    result += '\r';
                    break;
                default:
                    result += current();
                }
            }
            else {
                result += current();
            }
            advance();
        }
        expect('"');
        return result;
    }

    double parse_number()
    {
        skip_whitespace();
        size_t start = pos_;
        if (current() == '-')
            advance();
        while (!at_end()
               && (std::isdigit(current()) || current() == '.' || current() == 'e' || current() == 'E'
                   || current() == '+' || current() == '-'))
            advance();
        return std::stod(str_.substr(start, pos_ - start));
    }

    JsonValue parse_value()
    {
        skip_whitespace();
        char c = current();

        if (c == '"') {
            return parse_string();
        }
        else if (c == '{') {
            return std::make_unique<JsonObject>(parse_object());
        }
        else if (c == '[') {
            return parse_array();
        }
        else if (c == 't' || c == 'f') {
            return parse_bool();
        }
        else if (c == 'n') {
            return parse_null();
        }
        else if (std::isdigit(c) || c == '-') {
            return parse_number();
        }
        throw std::runtime_error(std::string("Unexpected character: ") + c);
    }

    JsonObject parse_object()
    {
        JsonObject obj;
        expect('{');
        skip_whitespace();

        if (current() == '}') {
            advance();
            return obj;
        }

        while (true) {
            std::string key = parse_string();
            skip_whitespace();
            expect(':');
            JsonValue value = parse_value();
            obj.data[key]   = std::move(value);

            skip_whitespace();
            if (current() == '}') {
                advance();
                break;
            }
            expect(',');
        }
        return obj;
    }

    JsonArray parse_array()
    {
        JsonArray arr;
        expect('[');
        skip_whitespace();

        if (current() == ']') {
            advance();
            return arr;
        }

        while (true) {
            skip_whitespace();
            if (current() == '{') {
                arr.push_back(parse_object());
            }
            else {
                // For simplicity, we only support arrays of objects
                throw std::runtime_error("Only arrays of objects are supported");
            }

            skip_whitespace();
            if (current() == ']') {
                advance();
                break;
            }
            expect(',');
        }
        return arr;
    }

    bool parse_bool()
    {
        skip_whitespace();
        if (str_.substr(pos_, 4) == "true") {
            pos_ += 4;
            return true;
        }
        else if (str_.substr(pos_, 5) == "false") {
            pos_ += 5;
            return false;
        }
        throw std::runtime_error("Expected 'true' or 'false'");
    }

    std::nullptr_t parse_null()
    {
        skip_whitespace();
        if (str_.substr(pos_, 4) == "null") {
            pos_ += 4;
            return nullptr;
        }
        throw std::runtime_error("Expected 'null'");
    }
};

// ============================================================================
// Benchmark Input Parser - Parse requests from JSON file
// ============================================================================

inline std::vector<Request> parse_benchmark_input(const std::string &filepath)
{
    JsonParser           parser;
    JsonObject           root     = parser.parse_file(filepath);
    const JsonArray     &requests = root.get_array("requests");
    std::vector<Request> result;

    int request_id = 0;
    for (const auto &req_obj : requests) {
        std::string prompt      = req_obj.get_string("prompt", "");
        float       temperature = req_obj.get_float("temperature", 1.0f);
        float       top_p       = req_obj.get_float("top_p", 0.9f);
        int         max_tokens  = req_obj.get_int("max_tokens", 256);

        if (prompt.empty()) {
            throw std::runtime_error("Request " + std::to_string(request_id) + " has empty prompt");
        }

        SamplingParams params(temperature, top_p, max_tokens);
        result.emplace_back(request_id++, prompt, params);
    }

    return result;
}

} // namespace json
