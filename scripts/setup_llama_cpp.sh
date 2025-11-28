#!/bin/bash
# Script to setup llama.cpp for GGUF export

set -e

echo "Setting up llama.cpp..."

if [ -d "llama.cpp" ]; then
    echo "llama.cpp directory already exists. Pulling latest changes..."
    cd llama.cpp
    git pull
else
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
fi

echo "Building llama.cpp..."
make

echo "Installing python dependencies..."
pip install -r requirements.txt

echo "✓ llama.cpp setup complete!"
