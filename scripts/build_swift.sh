#!/bin/bash
set -e

# Build Swift dylib with FoundationModels framework

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SWIFT_SRC="$PROJECT_ROOT/libai/swift/apple_ai.swift"
LIB_DIR="$PROJECT_ROOT/lib"
OUTPUT_DYLIB="$LIB_DIR/libappleai.dylib"

echo "Building Swift dylib with FoundationModels..."

# Create lib directory if it doesn't exist
mkdir -p "$LIB_DIR"

# Check if running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: This script must be run on macOS"
    exit 1
fi

# Check macOS version
OS_VERSION=$(sw_vers -productVersion | cut -d'.' -f1)
if [[ "$OS_VERSION" -lt 26 ]]; then
    echo "Warning: macOS 26.0+ required for Apple Intelligence"
    echo "Current version: $(sw_vers -productVersion)"
    echo "Continuing anyway (library will be built but may not function)"
fi

# Check if Swift source exists
if [[ ! -f "$SWIFT_SRC" ]]; then
    echo "Error: Swift source not found at $SWIFT_SRC"
    exit 1
fi

# Compile Swift to dylib
echo "Compiling $SWIFT_SRC..."
swiftc "$SWIFT_SRC" \
    -O \
    -whole-module-optimization \
    -target arm64-apple-macos26.0 \
    -framework Foundation \
    -framework FoundationModels \
    -emit-library \
    -o "$OUTPUT_DYLIB" \
    -emit-module \
    -emit-module-path "$LIB_DIR/apple_ai.swiftmodule" \
    -Xlinker -install_name \
    -Xlinker @rpath/libappleai.dylib

if [[ $? -eq 0 ]]; then
    echo "✓ Successfully built: $OUTPUT_DYLIB"
    echo "  Size: $(du -h "$OUTPUT_DYLIB" | cut -f1)"
    echo "  Architecture: $(file "$OUTPUT_DYLIB" | cut -d':' -f2)"
else
    echo "✗ Build failed"
    exit 1
fi

# Verify the dylib
echo ""
echo "Verifying dylib..."
otool -L "$OUTPUT_DYLIB" | grep -E "(FoundationModels|Foundation|Swift)" || true

echo ""
echo "✓ Build complete!"
echo "  Output: $OUTPUT_DYLIB"
