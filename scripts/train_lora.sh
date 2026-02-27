#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" src/train_lora_mt5.py --config configs/lora_mt5_small.yaml "$@"
