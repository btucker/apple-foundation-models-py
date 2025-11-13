"""Setup script for apple-foundation-models-py Python bindings."""

import sys
import os
import platform
import subprocess
import shutil
import json
import warnings
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as _build_py

# Suppress warning about swift directory not being a Python package
warnings.filterwarnings("ignore", message=".*applefoundationmodels\\.swift.*")

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
SWIFT_SRC_DIR = PKG_DIR / "swift"
LIB_DIR = REPO_ROOT / "lib"
PKG_DYLIB = PKG_DIR / "libfoundation_models.dylib"
ERROR_CODES_JSON = PKG_DIR / "error_codes.json"
SWIFT_ERROR_CODES_FILE = SWIFT_SRC_DIR / "error_codes.generated.swift"
SWIFT_MODULE_CACHE = REPO_ROOT / ".swift-module-cache"

# Only support Apple Silicon (arm64)
ARCH = "arm64"


def generate_swift_error_code_file():
    """Generate Swift source that embeds error code mappings."""
    if not ERROR_CODES_JSON.exists():
        sys.exit(f"Error: {ERROR_CODES_JSON} not found")

    with ERROR_CODES_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    entries = []
    for entry in data:
        swift_case = entry.get("swift_case")

        # Auto-generate swift_case from name if not provided
        # Pattern: "error" + name without "Error" suffix
        # e.g., InitializationError → errorInitialization
        if not swift_case:
            name = entry["name"]
            if name.endswith("Error"):
                name = name[:-5]  # Remove "Error" suffix
            swift_case = "error" + name

        entries.append((swift_case, int(entry["code"])))

    entries.sort(key=lambda item: item[0])

    lines = [
        "// This file is auto-generated from error_codes.json. Do not edit manually.",
        "// Run setup.py build (or pip install) to regenerate.",
        "",
        "let ERROR_CODE_MAPPINGS: [String: Int32] = [",
    ]
    for name, code in entries:
        lines.append(f'    "{name}": {code},')
    lines.append("]")
    lines.append("")

    content = "\n".join(lines)

    if SWIFT_ERROR_CODES_FILE.exists():
        existing = SWIFT_ERROR_CODES_FILE.read_text(encoding="utf-8")
        if existing == content:
            return

    SWIFT_ERROR_CODES_FILE.write_text(content, encoding="utf-8")
    print(f"✓ Generated Swift error code map: {SWIFT_ERROR_CODES_FILE}")


def build_swift_dylib():
    """Build the Swift FoundationModels dylib."""
    dylib_path = LIB_DIR / "libfoundation_models.dylib"

    generate_swift_error_code_file()

    # Validate environment
    if platform.system() != "Darwin":
        sys.exit("Error: Swift dylib can only be built on macOS")

    # Collect all Swift source files
    swift_sources = sorted(SWIFT_SRC_DIR.glob("*.swift"))
    if not swift_sources:
        sys.exit("Error: No Swift source files found")

    # Skip if up to date - check all Swift sources
    if dylib_path.exists():
        # Get modification time of all Swift sources
        swift_mtimes = [src.stat().st_mtime for src in swift_sources]
        if SWIFT_ERROR_CODES_FILE.exists():
            swift_mtimes.append(SWIFT_ERROR_CODES_FILE.stat().st_mtime)

        latest_src_mtime = max(swift_mtimes) if swift_mtimes else 0

        if latest_src_mtime <= dylib_path.stat().st_mtime:
            print(f"Swift dylib is up to date: {dylib_path}")
            return

    print("Building Swift FoundationModels dylib...")
    LIB_DIR.mkdir(parents=True, exist_ok=True)

    # Check for Xcode (required for @Generable macro support)
    xcode_path = Path("/Applications/Xcode.app")
    if xcode_path.exists():
        # Use Xcode's toolchain which includes the FoundationModelsMacros plugin
        swift_compiler = "xcrun"
        sdk_args = [
            "-sdk",
            "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk",
        ]
        env = {
            **os.environ,
            "DEVELOPER_DIR": "/Applications/Xcode.app/Contents/Developer",
        }
        print("✓ Using Xcode toolchain (@Generable macro support enabled)")
    else:
        # Fall back to command line tools (no @Generable support)
        swift_compiler = "swiftc"
        sdk_args = []
        env = None
        print("⚠ Using command line tools (@Generable macro not available)")

    # Convert Path objects to strings for compiler
    swift_source_paths = sorted(str(path) for path in swift_sources)

    SWIFT_MODULE_CACHE.mkdir(exist_ok=True)

    cmd = [
        swift_compiler,
        "swiftc" if swift_compiler == "xcrun" else None,
        *sdk_args,
        *swift_source_paths,
        "-O",
        "-whole-module-optimization",
        "-target",
        f"{ARCH}-apple-macos26.0",
        "-framework",
        "Foundation",
        "-framework",
        "FoundationModels",
        "-emit-library",
        "-o",
        str(dylib_path),
        "-emit-module",
        "-emit-module-path",
        str(LIB_DIR / "foundation_models.swiftmodule"),
        "-Xlinker",
        "-install_name",
        "-Xlinker",
        "@rpath/libfoundation_models.dylib",
        "-module-cache-path",
        str(SWIFT_MODULE_CACHE),
    ]
    cmd = [arg for arg in cmd if arg is not None]  # Remove None entries

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        print(f"✓ Built: {dylib_path} ({dylib_path.stat().st_size / 1024:.1f} KB)")
        shutil.copy2(dylib_path, PKG_DYLIB)
        print(f"✓ Copied to: {PKG_DYLIB}")
    except FileNotFoundError:
        sys.exit(
            "Error: swiftc not found. Install Xcode: https://developer.apple.com/xcode/"
        )
    except subprocess.CalledProcessError as e:
        sys.exit(f"Error: Swift compilation failed\n{e.stderr}")


class BuildPyWithDylib(_build_py):
    """Build Swift dylib before copying package files."""

    def run(self):
        build_swift_dylib()
        super().run()
        if PKG_DYLIB.exists():
            target = (
                Path(self.build_lib)
                / "applefoundationmodels"
                / "libfoundation_models.dylib"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PKG_DYLIB, target)


class BuildSwiftThenExt(_build_ext):
    """Build Swift dylib before building Cython extension."""

    def run(self):
        build_swift_dylib()
        super().run()


# Cython extension - only build on macOS with Cython available
if platform.system() == "Darwin" and CYTHON_AVAILABLE:
    ext_modules = cythonize(
        [
            Extension(
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
            )
        ],
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

if __name__ == "__main__":
    setup(
        packages=find_packages(
            exclude=["applefoundationmodels.swift", "applefoundationmodels.swift.*"]
        ),
        ext_modules=ext_modules,
        cmdclass=cmdclass,
    )
