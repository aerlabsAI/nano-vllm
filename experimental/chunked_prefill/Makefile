SHELL := /bin/bash

CXX := clang++
CXX_FLAGS := -std=c++20 -O3 -I ../../include -I .

.PHONY: all clean

all: build/main_chunked

build/main_chunked: main_chunked.cpp model_chunked.hpp chunking.hpp
	@mkdir -p build
	$(CXX) $(CXX_FLAGS) -o $@ main_chunked.cpp

clean:
	rm -rf build/
