#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>

#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define WHITE   "\033[37m"
#define GRAY    "\033[90m"

// TODO: VLLM_LOGGER_LEVEL
class Logger
{
private:
    static std::string getCurrentTime()
    {
        auto now    = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);

        std::stringstream ss;
        ss << std::put_time(localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }

    static std::string getFilename(const char *file)
    {
        std::string fullPath(file);
        // src/ 이후의 경로만 반환
        size_t srcPos = fullPath.find("/src/");
        if (srcPos != std::string::npos) {
            return fullPath.substr(srcPos + 1); // /src/ → src/
        }
        return fullPath;
    }

    static void log(std::ostream      &stream,
                    const std::string &level,
                    const std::string &color,
                    const std::string &message,
                    const char        *file,
                    int                line)
    {
        stream << WHITE << "[" << BLUE << getCurrentTime() << WHITE << "] [" << GREEN << getFilename(file) << ":"
               << std::to_string(line) << WHITE << "] " << RESET << level << " " << color << message << RESET
               << std::endl;
    }

    template <typename T> static void addToStream(std::stringstream &ss, T &&value)
    {
        if constexpr (std::is_same_v<std::decay_t<T>, std::string>) {
            ss << value;
        }
        else if constexpr (std::is_same_v<std::decay_t<T>, const char *> || std::is_same_v<std::decay_t<T>, char *>) {
            ss << std::string(value);
        }
        else if constexpr (std::is_array_v<std::remove_reference_t<T>>) {
            ss << "[";
            for (size_t i = 0; i < std::extent_v<std::remove_reference_t<T>>; ++i) {
                if (i > 0)
                    ss << ", ";
                ss << value[i];
            }
            ss << "]";
        }
        else if constexpr (std::is_floating_point_v<std::decay_t<T>>) {
            ss << std::fixed << std::setprecision(6) << value;
        }
        else if constexpr (std::is_arithmetic_v<std::decay_t<T>>) {
            ss << value;
        }
        else {
            ss << "[unsupported type]";
        }
    }

    template <typename... Args> static std::string buildMessage(Args &&...args)
    {
        std::stringstream ss;
        (addToStream(ss, std::forward<Args>(args)), ...);
        return ss.str();
    }

public:
    template <typename... Args> static void info(const char *file, int line, Args &&...args)
    {
        log(std::cout, "ℹ️", CYAN, buildMessage(std::forward<Args>(args)...), file, line);
    }

    template <typename... Args> static void success(const char *file, int line, Args &&...args)
    {
        log(std::cout, "✅", GREEN, buildMessage(std::forward<Args>(args)...), file, line);
    }

    template <typename... Args> static void warning(const char *file, int line, Args &&...args)
    {
        log(std::cout, "⚠️", YELLOW, buildMessage(std::forward<Args>(args)...), file, line);
    }

    template <typename... Args> static void error(const char *file, int line, Args &&...args)
    {
        log(std::cerr, "❌", RED, buildMessage(std::forward<Args>(args)...), file, line);
    }
};

// CPU Logger macros
#define LOG_INFO(...)    Logger::info(__FILE__, __LINE__, __VA_ARGS__)
#define LOG_SUCCESS(...) Logger::success(__FILE__, __LINE__, __VA_ARGS__)
#define LOG_WARNING(...) Logger::warning(__FILE__, __LINE__, __VA_ARGS__)
#define LOG_ERROR(...)   Logger::error(__FILE__, __LINE__, __VA_ARGS__)
