<<<<<<< HEAD
# NOTE: Ref: https://github.com/NVIDIA/cuda-samples
=======
# NOTE:
# Ref: https://github.com/NVIDIA/cuda-samples
>>>>>>> 938f31b (Fix CMakelists)

cmake_minimum_required(VERSION 3.20)

option(USE_NVCC "Use NVCC instead of Clang++ for CUDA" OFF)
option(USE_DEBUG "Use CUDA debug mode" OFF)

set(CMAKE_CXX_COMPILER "clang++")
project(nano-vllm LANGUAGES C CXX)

# Try to find CUDA Toolkit
find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
  enable_language(CUDA)
  if(USE_NVCC)
    set(CMAKE_CUDA_COMPILER "nvcc")
    message(STATUS "Using NVCC compiler")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
    if(USE_DEBUG)
      set(CMAKE_BUILD_TYPE "Debug")
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
      message(STATUS "NVCC debug mode enabled")
    else()
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
    endif()
  else()
    set(CMAKE_CUDA_COMPILER "clang++")
    message(STATUS "Using Clang++ for CUDA")
    if(USE_DEBUG)
      set(CMAKE_BUILD_TYPE "Debug")
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --cuda-noopt-device-debug")
      message(STATUS "Clang++ debug mode enabled")
    endif()
  endif()
else()
<<<<<<< HEAD
  message(
    WARNING
      "CUDA Toolkit not found. Building C++ only (CUDA files will be skipped).")
=======
  message(WARNING "CUDA Toolkit not found. Building C++ only (CUDA files will be skipped).")
>>>>>>> 938f31b (Fix CMakelists)
  if(USE_DEBUG)
    set(CMAKE_BUILD_TYPE "Debug")
  endif()
endif()
# find_package(OpenGL REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# macOS SDK path for Homebrew LLVM compatibility
if(APPLE)
  execute_process(
    COMMAND xcrun --show-sdk-path
    OUTPUT_VARIABLE MACOS_SDK_PATH
<<<<<<< HEAD
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
=======
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
>>>>>>> 938f31b (Fix CMakelists)
  if(MACOS_SDK_PATH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isysroot ${MACOS_SDK_PATH}")
    message(STATUS "Using macOS SDK: ${MACOS_SDK_PATH}")
  endif()
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CPU optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -march=native")

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 90) # (75 80 86 87 89 90 100 110 120)

include_directories(include)

file(GLOB_RECURSE ALL_CPP_SOURCES "src/*.cpp" "example/*.cpp")
foreach(cpp_source ${ALL_CPP_SOURCES})
  get_filename_component(executable_name ${cpp_source} NAME_WE)
  get_filename_component(source_dir ${cpp_source} DIRECTORY)
  if(source_dir MATCHES "/src$" OR source_dir MATCHES "/src/")
    set(PREFIX "src_")
  elseif(source_dir MATCHES "/example$" OR source_dir MATCHES "/example/")
    set(PREFIX "example_")
  else()
    set(PREFIX "")
  endif()
  add_executable(${PREFIX}${executable_name} ${cpp_source})
endforeach()

if(CUDAToolkit_FOUND)
  file(GLOB_RECURSE ALL_CU_SOURCES "src/*.cu" "example/*.cu")
  foreach(cu_source ${ALL_CU_SOURCES})
    get_filename_component(executable_name ${cu_source} NAME_WE)
    get_filename_component(source_dir ${cu_source} DIRECTORY)
    if(source_dir MATCHES "/src$" OR source_dir MATCHES "/src/")
      set(PREFIX "src_")
    elseif(source_dir MATCHES "/example$" OR source_dir MATCHES "/example/")
      set(PREFIX "example_")
    else()
      set(PREFIX "")
    endif()
    add_executable(${PREFIX}${executable_name} ${cu_source})
  endforeach()
<<<<<<< HEAD
endif()
=======
endif()
>>>>>>> 938f31b (Fix CMakelists)
