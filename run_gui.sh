#!/bin/bash
# Polyglot GUI launcher
# This script sets up the correct environment and launches the GUI

cd "$(dirname "$0")"

# Set environment variables for Qt/PyTorch compatibility
export QT_MAC_WANTS_LAYER=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Use Python 3.10
/opt/homebrew/bin/python3.10 gui.py "$@"
