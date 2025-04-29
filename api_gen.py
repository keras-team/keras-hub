"""Script to generate keras_hub public API in `keras_hub/api` directory.

Usage:

Run via `./shell/api_gen.sh`.
It generates API and formats user and generated APIs.
"""

import os
import shutil

import namex

PACKAGE = "keras_hub"
BUILD_DIR_NAME = "tmp_build_dir"


def ignore_files(_, filenames):
    return [f for f in filenames if f.endswith("_test.py")]


def copy_source_to_build_directory(root_path):
    # Copy sources (`keras_hub/` directory and setup files) to build dir
    build_dir = os.path.join(root_path, BUILD_DIR_NAME)
    build_package_dir = os.path.join(build_dir, PACKAGE)
    build_src_dir = os.path.join(build_package_dir, "src")
    root_src_dir = os.path.join(root_path, PACKAGE, "src")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_package_dir)
    shutil.copytree(root_src_dir, build_src_dir)
    return build_dir


def export_version_string(api_init_fname):
    with open(api_init_fname) as f:
        contents = f.read()
    with open(api_init_fname, "w") as f:
        contents += (
            "from keras_hub.src.version import __version__ as __version__\n"
        )
        f.write(contents)


def build():
    root_path = os.path.dirname(os.path.abspath(__file__))
    code_api_dir = os.path.join(root_path, PACKAGE, "api")
    # Create temp build dir
    build_dir = copy_source_to_build_directory(root_path)
    build_api_dir = os.path.join(build_dir, PACKAGE)
    build_src_dir = os.path.join(build_api_dir, "src")
    build_api_init_fname = os.path.join(build_api_dir, "__init__.py")
    try:
        os.chdir(build_dir)
        open(build_api_init_fname, "w").close()
        namex.generate_api_files("keras_hub", code_directory="src")
        # Add __version__ to `api/`.
        export_version_string(build_api_init_fname)
        # Copy back the keras/api from build directory
        if os.path.exists(build_src_dir):
            shutil.rmtree(build_src_dir)
        if os.path.exists(code_api_dir):
            shutil.rmtree(code_api_dir)
        shutil.copytree(
            build_api_dir, code_api_dir, ignore=shutil.ignore_patterns("src/")
        )
    finally:
        # Clean up: remove the build directory (no longer needed)
        shutil.rmtree(build_dir)


if __name__ == "__main__":
    build()
