"""Script to generate keras_hub public API in `keras_hub/api` directory.

Usage:

Run via `./shell/api_gen.sh`.
It generates API and formats user and generated APIs.
"""

import os
import shutil

import namex

package = "keras_hub"


def ignore_files(_, filenames):
    return [f for f in filenames if f.endswith("_test.py")]


def copy_source_to_build_directory(root_path):
    # Copy sources (`keras_hub/` directory and setup files) to build dir
    build_dir = os.path.join(root_path, "tmp_build_dir")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.mkdir(build_dir)
    shutil.copytree(
        package, os.path.join(build_dir, package), ignore=ignore_files
    )
    return build_dir


def export_version_string(api_init_fname):
    with open(api_init_fname) as f:
        contents = f.read()
    with open(api_init_fname, "w") as f:
        contents += "from keras_hub.src.version_utils import __version__\n"
        f.write(contents)


def build():
    # Backup the `keras_hub/__init__.py` and restore it on error in api gen.
    root_path = os.path.dirname(os.path.abspath(__file__))
    code_api_dir = os.path.join(root_path, package, "api")
    # Create temp build dir
    build_dir = copy_source_to_build_directory(root_path)
    build_api_dir = os.path.join(build_dir, package, "api")
    build_init_fname = os.path.join(build_dir, package, "__init__.py")
    build_api_init_fname = os.path.join(build_api_dir, "__init__.py")
    try:
        os.chdir(build_dir)
        # Generates `keras_hub/api` directory.
        if os.path.exists(build_api_dir):
            shutil.rmtree(build_api_dir)
        if os.path.exists(build_init_fname):
            os.remove(build_init_fname)
        os.makedirs(build_api_dir)
        namex.generate_api_files(
            "keras_hub", code_directory="src", target_directory="api"
        )
        # Add __version__ to keras package
        export_version_string(build_api_init_fname)
        # Copy back the keras_hub/api and keras_hub/__init__.py from build dir
        if os.path.exists(code_api_dir):
            shutil.rmtree(code_api_dir)
        shutil.copytree(build_api_dir, code_api_dir)
    finally:
        # Clean up: remove the build directory (no longer needed)
        shutil.rmtree(build_dir)


if __name__ == "__main__":
    build()
