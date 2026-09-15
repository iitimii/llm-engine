#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.." 

cmake --preset ci # Release + LLM_WERROR=ON
cmake --build --preset ci -j
ctest --test-dir build/ci --output-on-failure

echo "build + tests green"