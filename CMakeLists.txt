cmake_minimum_required(VERSION 3.20)

project(nano-vllm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimization flags
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-O3 -ffast-math -march=native)
endif()

# Main Executable
add_executable(src_main src/main.cpp)

# Include directories
target_include_directories(src_main PRIVATE include)
