"""
Setup script for libai-py Python bindings.

Builds the Cython extension and links against the bundled libai static library.
"""

import os
import sys
import platform
from pathlib import Path
from setuptools import setup, Extension
from Cython.Build import cythonize


# Determine paths
REPO_ROOT = Path(__file__).parent.resolve()
LIB_DIR = REPO_ROOT / "lib"
INCLUDE_DIR = REPO_ROOT / "include"

# Detect architecture
ARCH = platform.machine()
if ARCH not in ["arm64", "x86_64"]:
    print(f"Warning: Unsupported architecture {ARCH}, attempting to use x86_64")
    ARCH = "x86_64"

# Check that library exists
LIBAI_PATH = LIB_DIR / "libai.a"
if not LIBAI_PATH.exists():
    print(f"Error: libai.a not found at {LIBAI_PATH}")
    print("Please run ./scripts/download_libs.sh first")
    sys.exit(1)

# Check that headers exist
AI_HEADER = INCLUDE_DIR / "ai.h"
if not AI_HEADER.exists():
    print(f"Error: ai.h not found at {AI_HEADER}")
    print("Please run ./scripts/download_libs.sh first")
    sys.exit(1)

# Define the Cython extension
extensions = [
    Extension(
        name="libai._libai",
        sources=["libai/_libai.pyx"],
        include_dirs=[str(INCLUDE_DIR)],
        library_dirs=[str(LIB_DIR)],
        libraries=["ai"],
        extra_compile_args=[
            "-O3",  # Optimization
            "-Wall",  # Warnings
        ],
        extra_link_args=[
            # Link required macOS frameworks
            "-framework", "Foundation",
            "-framework", "ApplicationServices",
            "-framework", "CoreFoundation",
            # Weakly link FoundationModels (may not be available until AI is fully enabled)
            "-Wl,-weak_framework,FoundationModels",
            # Add RPATH for Swift runtime libraries
            "-Wl,-rpath,/usr/lib/swift",
            # Allow undefined symbols to be resolved at runtime
            "-Wl,-undefined,dynamic_lookup",
        ],
        language="c",
    )
]

# Cythonize extensions
ext_modules = cythonize(
    extensions,
    compiler_directives={
        "language_level": "3",
        "embedsignature": True,
        "boundscheck": False,
        "wraparound": False,
    },
    annotate=False,  # Set to True to generate HTML annotation files
)

# Run setup
if __name__ == "__main__":
    setup(
        ext_modules=ext_modules,
    )
