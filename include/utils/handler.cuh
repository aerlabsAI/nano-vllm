#pragma once

#include <cstdlib>
#include <string>

#include "logger.hpp"

class Handler
{
public:
    static void checkError(cudaError_t error, const char *file, int line, const char *func = nullptr)
    {
        if (error != cudaSuccess) {
            std::string message = "CUDA ERROR";
            if (func) {
                message += " " + std::string(func);
            }
            message += "\n  " + std::string(cudaGetErrorString(error)) + " (code: " + std::to_string(error) + ")";

            Logger::error(file, line, message);
            std::exit(EXIT_FAILURE);
        }
        else if (func) {
            Logger::success(file, line, func);
        }
    }
};

// CUDA macro
#define CUDA_CHECK(error) Handler::checkError(error, __FILE__, __LINE__, #error)
