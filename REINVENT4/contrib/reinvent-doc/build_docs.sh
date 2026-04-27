#!/usr/bin/env bash
# Build REINVENT4 documentation with Sphinx
# Run from the repo root: bash doc/build_docs.sh

set -e

DOC_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$DOC_DIR/_build/html"

echo "==> Installing doc dependencies..."
pip install -r "$DOC_DIR/requirements.txt"

echo "==> Building HTML..."
sphinx-build -b html "$DOC_DIR" "$BUILD_DIR"

echo ""
echo "Done. Open: $BUILD_DIR/index.html"
