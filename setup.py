"""
Setup script for apple-foundation-models-py Python bindings.

Builds the Swift FoundationModels dylib and Cython extension automatically.
"""

import os
import sys
import platform
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize


# Determine paths
REPO_ROOT = Path(__file__).parent.resolve()
LIB_DIR = REPO_ROOT / "lib"
SWIFT_DIR = REPO_ROOT / "applefoundationmodels" / "swift"
SWIFT_SRC = SWIFT_DIR / "foundation_models.swift"
DYLIB_PATH = LIB_DIR / "libfoundation_models.dylib"
SWIFTMODULE_PATH = LIB_DIR / "foundation_models.swiftmodule"

# Detect architecture
ARCH = platform.machine()
if ARCH not in ["arm64", "x86_64"]:
    print(f"Warning: Unsupported architecture {ARCH}, attempting to use x86_64")
    ARCH = "x86_64"


class BuildSwiftThenExt(_build_ext):
    """Custom build_ext that builds Swift dylib before building Cython extension."""

    def run(self):
        """Build Swift dylib, then build Cython extension."""
        self.build_swift_dylib()
        super().run()

    def build_swift_dylib(self):
        """Build the Swift FoundationModels dylib."""
        # Check if we need to rebuild
        needs_rebuild = (
            not DYLIB_PATH.exists() or
            not SWIFT_SRC.exists() or
            SWIFT_SRC.stat().st_mtime > DYLIB_PATH.stat().st_mtime
        )

        if not needs_rebuild:
            print(f"Swift dylib is up to date: {DYLIB_PATH}")
            return

        print("=" * 70)
        print("Building Swift FoundationModels dylib...")
        print("=" * 70)

        # Check if running on macOS
        if platform.system() != "Darwin":
            print("Error: Swift dylib can only be built on macOS")
            sys.exit(1)

        # Check Swift source exists
        if not SWIFT_SRC.exists():
            print(f"Error: Swift source not found at {SWIFT_SRC}")
            sys.exit(1)

        # Check macOS version
        os_version = int(platform.mac_ver()[0].split('.')[0])
        if os_version < 26:
            print(f"Warning: macOS 26.0+ required for Apple Intelligence")
            print(f"Current version: {platform.mac_ver()[0]}")
            print("Continuing anyway (library will be built but may not function)")

        # Create lib directory
        LIB_DIR.mkdir(parents=True, exist_ok=True)

        # Compile Swift to dylib
        cmd = [
            "swiftc", str(SWIFT_SRC),
            "-O",
            "-whole-module-optimization",
            f"-target", f"{ARCH}-apple-macos26.0",
            "-framework", "Foundation",
            "-framework", "FoundationModels",
            "-emit-library",
            "-o", str(DYLIB_PATH),
            "-emit-module",
            "-emit-module-path", str(SWIFTMODULE_PATH),
            "-Xlinker", "-install_name",
            "-Xlinker", f"@rpath/libfoundation_models.dylib",
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)

            print(f"✓ Successfully built: {DYLIB_PATH}")
            print(f"  Size: {DYLIB_PATH.stat().st_size / 1024:.1f} KB")

        except subprocess.CalledProcessError as e:
            print(f"✗ Swift compilation failed")
            print(e.stderr)
            sys.exit(1)
        except FileNotFoundError:
            print("✗ Swift compiler (swiftc) not found")
            print("Please install Xcode Command Line Tools:")
            print("  xcode-select --install")
            sys.exit(1)

# Define the Cython extension
extensions = [
    Extension(
        name="applefoundationmodels._foundationmodels",
        sources=["applefoundationmodels/_foundationmodels.pyx"],
        include_dirs=[str(SWIFT_DIR)],  # Include Swift header directory
        library_dirs=[str(LIB_DIR)],
        libraries=["foundation_models"],  # Link against libfoundation_models.dylib
        extra_compile_args=[
            "-O3",  # Optimization
            "-Wall",  # Warnings
        ],
        extra_link_args=[
            # Set RPATH to find dylib at runtime
            f"-Wl,-rpath,{LIB_DIR}",  # Absolute path for development
            "-Wl,-rpath,@loader_path/../lib",  # Relative path for installed package
            "-Wl,-rpath,@loader_path",  # Also check same directory
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
        cmdclass={
            'build_ext': BuildSwiftThenExt,
        },
    )
