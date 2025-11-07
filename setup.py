"""Setup script for apple-foundation-models-py Python bindings."""

import sys
import platform
import subprocess
import shutil
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as _build_py
from Cython.Build import cythonize

# Paths
REPO_ROOT = Path(__file__).parent.resolve()
PKG_DIR = REPO_ROOT / "applefoundationmodels"
SWIFT_SRC = PKG_DIR / "swift" / "foundation_models.swift"
LIB_DIR = REPO_ROOT / "lib"
PKG_DYLIB = PKG_DIR / "libfoundation_models.dylib"

# Only support Apple Silicon (arm64)
ARCH = "arm64"


def build_swift_dylib():
    """Build the Swift FoundationModels dylib."""
    dylib_path = LIB_DIR / "libfoundation_models.dylib"

    # Skip if up to date
    if dylib_path.exists() and SWIFT_SRC.exists():
        if SWIFT_SRC.stat().st_mtime <= dylib_path.stat().st_mtime:
            print(f"Swift dylib is up to date: {dylib_path}")
            return

    # Validate environment
    if platform.system() != "Darwin":
        sys.exit("Error: Swift dylib can only be built on macOS")
    if not SWIFT_SRC.exists():
        sys.exit(f"Error: Swift source not found at {SWIFT_SRC}")

    print("Building Swift FoundationModels dylib...")
    LIB_DIR.mkdir(parents=True, exist_ok=True)

    # Build dylib
    cmd = [
        "swiftc", str(SWIFT_SRC),
        "-O", "-whole-module-optimization",
        "-target", f"{ARCH}-apple-macos26.0",
        "-framework", "Foundation", "-framework", "FoundationModels",
        "-emit-library", "-o", str(dylib_path),
        "-emit-module", "-emit-module-path", str(LIB_DIR / "foundation_models.swiftmodule"),
        "-Xlinker", "-install_name", "-Xlinker", "@rpath/libfoundation_models.dylib",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Built: {dylib_path} ({dylib_path.stat().st_size / 1024:.1f} KB)")
        shutil.copy2(dylib_path, PKG_DYLIB)
        print(f"✓ Copied to: {PKG_DYLIB}")
    except FileNotFoundError:
        sys.exit("Error: swiftc not found. Install Xcode Command Line Tools: xcode-select --install")
    except subprocess.CalledProcessError as e:
        sys.exit(f"Error: Swift compilation failed\n{e.stderr}")


class BuildPyWithDylib(_build_py):
    """Build Swift dylib before copying package files."""
    def run(self):
        build_swift_dylib()
        super().run()
        if PKG_DYLIB.exists():
            target = Path(self.build_lib) / "applefoundationmodels" / "libfoundation_models.dylib"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PKG_DYLIB, target)


class BuildSwiftThenExt(_build_ext):
    """Build Swift dylib before building Cython extension."""
    def run(self):
        build_swift_dylib()
        super().run()

# Cython extension
ext_modules = cythonize(
    [Extension(
        "applefoundationmodels._foundationmodels",
        sources=["applefoundationmodels/_foundationmodels.pyx"],
        include_dirs=[str(PKG_DIR / "swift")],
        library_dirs=[str(LIB_DIR)],
        libraries=["foundation_models"],
        extra_compile_args=["-O3", "-Wall"],
        extra_link_args=[
            f"-Wl,-rpath,{LIB_DIR}",
            "-Wl,-rpath,@loader_path/../lib",
            "-Wl,-rpath,@loader_path",
        ],
        language="c",
    )],
    compiler_directives={
        "language_level": "3",
        "embedsignature": True,
        "boundscheck": False,
        "wraparound": False,
    },
)

if __name__ == "__main__":
    setup(
        ext_modules=ext_modules,
        cmdclass={"build_py": BuildPyWithDylib, "build_ext": BuildSwiftThenExt},
    )
