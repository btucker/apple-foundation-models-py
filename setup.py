"""Setup script for apple-foundation-models-py Python bindings."""

import sys
import os
import platform
import subprocess
import shutil
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as _build_py
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    WHEEL_AVAILABLE = True
except ImportError:
    WHEEL_AVAILABLE = False
    _bdist_wheel = None

try:
    from Cython.Build import cythonize
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    cythonize = None

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

    # Check for Xcode (required for @Generable macro support)
    xcode_path = Path("/Applications/Xcode.app")
    if xcode_path.exists():
        # Use Xcode's toolchain which includes the FoundationModelsMacros plugin
        swift_compiler = "xcrun"
        sdk_args = [
            "-sdk", "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
        ]
        env = {**os.environ, "DEVELOPER_DIR": "/Applications/Xcode.app/Contents/Developer"}
        print("✓ Using Xcode toolchain (@Generable macro support enabled)")
    else:
        # Fall back to command line tools (no @Generable support)
        swift_compiler = "swiftc"
        sdk_args = []
        env = None
        print("⚠ Using command line tools (@Generable macro not available)")

    # Build dylib
    cmd = [
        swift_compiler,
        "swiftc" if swift_compiler == "xcrun" else None,
        *sdk_args,
        str(SWIFT_SRC),
        "-O", "-whole-module-optimization",
        "-target", f"{ARCH}-apple-macos26.0",
        "-framework", "Foundation", "-framework", "FoundationModels",
        "-emit-library", "-o", str(dylib_path),
        "-emit-module", "-emit-module-path", str(LIB_DIR / "foundation_models.swiftmodule"),
        "-Xlinker", "-install_name", "-Xlinker", "@rpath/libfoundation_models.dylib",
    ]
    cmd = [arg for arg in cmd if arg is not None]  # Remove None entries

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        print(f"✓ Built: {dylib_path} ({dylib_path.stat().st_size / 1024:.1f} KB)")
        shutil.copy2(dylib_path, PKG_DYLIB)
        print(f"✓ Copied to: {PKG_DYLIB}")
    except FileNotFoundError:
        sys.exit("Error: swiftc not found. Install Xcode: https://developer.apple.com/xcode/")
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


class MacOSWheel(_bdist_wheel if WHEEL_AVAILABLE else object):
    """Create a macOS wheel with deployment target that PyPI accepts.

    Uses macosx_11_0 deployment target instead of macosx_26_0 to avoid PyPI
    rejection, while runtime checks in Client.__init__() enforce macOS 26+.
    """
    def get_tag(self):
        # Get the default tags from parent
        python, abi, plat = super().get_tag()
        # Override platform tag to use macosx_11_0 instead of detected macosx_26_0
        # This allows PyPI to accept the wheel while runtime checks enforce macOS 26+
        plat = f"macosx_11_0_{ARCH}"
        return python, abi, plat

# Cython extension - only build on macOS with Cython available
if platform.system() == "Darwin" and CYTHON_AVAILABLE:
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
else:
    ext_modules = []
    print("Skipping Cython extension build (not on macOS or Cython not available)")

# Custom command classes
cmdclass = {
    "build_py": BuildPyWithDylib,
    "build_ext": BuildSwiftThenExt,
}

# Add bdist_wheel command if wheel is available
if WHEEL_AVAILABLE:
    cmdclass["bdist_wheel"] = MacOSWheel

if __name__ == "__main__":
    setup(
        ext_modules=ext_modules,
        cmdclass=cmdclass,
    )
