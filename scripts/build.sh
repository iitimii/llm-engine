#!/usr/bin/env bash
# set -e
set -euo pipefail

cd "$(dirname "$0")/.." 

#release
# cmake --preset release
# cmake --build --preset release
# ./build/release/llm_tests

# #debug
# cmake --preset debug
# cmake --build --preset debug
# ./build/debug/llm_tests

# #ci
# cmake --preset ci
# cmake --build --preset ci
# ./build/ci/llm_tests

cmake --preset ci # Release + LLM_WERROR=ON
cmake --build --preset ci -j
ctest --test-dir build/ci --output-on-failure
echo "build + tests green"