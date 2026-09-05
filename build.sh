#!/usr/bin/env bash
set -e

cmake --preset release
cmake --build --preset release
./build/release/llm_tests