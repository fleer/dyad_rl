#!/bin/bash

#exit on error
set -e

# ensure uv is available
if ! command -v uv &>/dev/null; then
  echo "Error: 'uv' is not installed or not on PATH. Install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

# create and activate the virtual environment
if [ -d ".venv" ]; then
  echo "Virtual environment already exists. Skipping creation."
else
  echo "Creating virtual environment..."
  uv venv --python=3.11
fi
source .venv/bin/activate

# build the C libraries
mkdir -p puzzle_env/rlp/lib
cd puzzle_env/rlp/lib
cmake ../../puzzles
TMP_MAKEFLAGS=$MAKEFLAGS
export MAKEFLAGS='-j 1'
if grep -q "^icons:" Makefile; then
  make icons
else
  echo "Skipping 'make icons' (icons target not available on this platform)."
fi
export MAKEFLAGS=$TMP_MAKEFLAGS
make
cd ../../..

echo "Which PyTorch backend should be installed?"
echo "  1) cpu (default)"
echo "  2) amd (ROCm)"
echo "  3) cuda"
read -rp "Enter choice [1-3]: " backend_choice

case "$backend_choice" in
2) PYTORCH_EXTRA="amd" ;;
3) PYTORCH_EXTRA="cuda" ;;
"" | 1) PYTORCH_EXTRA="cpu" ;;
*)
  echo "Invalid choice '$backend_choice'. Defaulting to 'cpu'." >&2
  PYTORCH_EXTRA="cpu"
  ;;
esac

echo "Installing with extras: $PYTORCH_EXTRA"
uv sync --extras "$PYTORCH_EXTRA"
