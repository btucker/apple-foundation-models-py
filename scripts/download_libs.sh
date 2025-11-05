#!/bin/bash
set -e

# Script to download latest libai release and prepare for Python binding

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LIB_DIR="$PROJECT_ROOT/lib"
INCLUDE_DIR="$PROJECT_ROOT/include"
TEMP_DIR=$(mktemp -d)

echo "Downloading latest libai release..."
cd "$TEMP_DIR"
curl -L https://github.com/6over3/libai/releases/latest/download/libai.tar.gz | tar xz

echo "Copying libraries..."
mkdir -p "$LIB_DIR"

# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    LIB_ARCH="arm64"
elif [ "$ARCH" = "x86_64" ]; then
    LIB_ARCH="x86_64"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

# Find and copy the static library for current architecture
LIB_PATH=$(find . -name "libai.a" -path "*apple/$LIB_ARCH/*" | head -1)
if [ -n "$LIB_PATH" ]; then
    cp "$LIB_PATH" "$LIB_DIR/"
    echo "✓ Copied libai.a for $LIB_ARCH"
else
    echo "✗ Failed to find libai.a for $LIB_ARCH"
    exit 1
fi

echo "Downloading header files from source repository..."
mkdir -p "$INCLUDE_DIR"
cd "$INCLUDE_DIR"

# Download headers directly from GitHub
curl -L -o ai.h https://raw.githubusercontent.com/6over3/libai/main/ai.h
echo "✓ Downloaded ai.h"

curl -L -o ai_bridge.h https://raw.githubusercontent.com/6over3/libai/main/ai_bridge.h
echo "✓ Downloaded ai_bridge.h"

# Clean up
cd "$PROJECT_ROOT"
rm -rf "$TEMP_DIR"

echo ""
echo "✓ Download complete!"
echo "  Architecture: $LIB_ARCH"
echo "  Libraries:    $LIB_DIR"
echo "  Headers:      $INCLUDE_DIR"
echo ""
echo "You can now run: pip install -e ."
