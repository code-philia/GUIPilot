#!/bin/bash
set -e

echo "Starting End-to-End Smoke Test..."

# 1. Install the package in editable mode
pip install -e .

# 2. Install dependencies (if any missing)
pip install opencv-python numpy supervision scikit-learn pillow

# 3. Set DATASET_PATH and PYTHONPATH
export DATASET_PATH="./datasets/new"
export PYTHONPATH=$PYTHONPATH:.

# 4. Run the smoke test
python tests/e2e_smoke_test.py

# 5. Check if evaluation.csv was created
if [ -f "evaluation.csv" ]; then
    echo "E2E Smoke Test Passed!"
else
    echo "E2E Smoke Test Failed: evaluation.csv not found."
    exit 1
fi
