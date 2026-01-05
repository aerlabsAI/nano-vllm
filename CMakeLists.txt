# NOTE: Ref: https://github.com/NVIDIA/cuda-samples

cmake_minimum_required(VERSION 3.20)

option(USE_DEBUG "Use debug mode" OFF)

set(CMAKE_CXX_COMPILER "clang++")
project(nano-vllm LANGUAGES C CXX)

if(USE_DEBUG)
  set(CMAKE_BUILD_TYPE "Debug")
  message(STATUS "Debug mode enabled")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# macOS SDK path for Homebrew LLVM compatibility
if(APPLE)
  execute_process(
    COMMAND xcrun --show-sdk-path
    OUTPUT_VARIABLE MACOS_SDK_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(MACOS_SDK_PATH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isysroot ${MACOS_SDK_PATH}")
    message(STATUS "Using macOS SDK: ${MACOS_SDK_PATH}")
  endif()
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CPU optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -march=native")

include_directories(include)

file(GLOB_RECURSE ALL_CPP_SOURCES "src/*.cpp")
foreach(cpp_source ${ALL_CPP_SOURCES})
  get_filename_component(executable_name ${cpp_source} NAME_WE)
  add_executable(${executable_name} ${cpp_source})
endforeach()
