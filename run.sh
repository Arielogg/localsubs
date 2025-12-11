#!/bin/bash
# Startup script for realtime-subtitles that sets CUDA library paths

# Set library path to include venv's CUDA libraries
export LD_LIBRARY_PATH="./venv/lib/python3.12/site-packages/nvidia/cudnn/lib:./venv/lib/python3.12/site-packages/nvidia/cublas/lib:./venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"

# Run the application
./venv/bin/python main.py
