#!/bin/bash

# Build the docker image
echo "Building Docker image..."
docker build --progress=plain -t dphsir-env .

# Run inference inside the docker container
# We mount the current directory to /workspace to easily access data and results
# and we use the --gpus all flag to enable GPU support

echo "Running inference with GRUNet..."
docker run --rm --gpus all -v "$(pwd):/workspace" dphsir-env python run_inference.py "$@"
