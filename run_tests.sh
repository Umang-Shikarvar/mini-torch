#!/bin/bash
# run_tests.sh
# Bash script to run both test files

pytest tests/test_minitorch_vs_torch.py
pytest tests/test_tensor_transpose.py
