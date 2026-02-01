#!/bin/bash
# Setup script for Fish Speech TTS virtualenv
# This creates a separate venv due to potential dependency conflicts

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv-fish"

echo "═══════════════════════════════════════════════════════════════════"
echo "  Setting up Fish Speech TTS Environment"
echo "═══════════════════════════════════════════════════════════════════"

# Check for Python 3.10+
PYTHON_CMD=""
for cmd in python3.10 python3.11 python3.12 python3; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Python 3.10+ is required but not found"
    exit 1
fi

echo "Using Python: $PYTHON_CMD ($($PYTHON_CMD --version))"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating virtual environment..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
fi

# Activate and install
echo ""
echo "Installing Fish Speech dependencies..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU or CUDA depending on system)
echo ""
echo "Installing PyTorch..."
if command -v nvidia-smi &>/dev/null; then
    echo "CUDA detected, installing CUDA version..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "No CUDA detected, installing CPU version..."
    pip install torch torchaudio
fi

# Install Fish Speech
echo ""
echo "Installing Fish Speech..."
pip install fish-speech

# Install additional dependencies
pip install scipy numpy rich

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Fish Speech setup complete!"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "To use Fish Speech TTS, the model will be downloaded on first use."
echo "Model size: ~2.5GB"
echo ""
