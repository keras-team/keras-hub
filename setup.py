# Copyright 2021 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup script."""

import os
import pathlib

from setuptools import find_packages
from setuptools import setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()
if os.path.exists("keras_nlp/version_utils.py"):
    VERSION = get_version("keras_nlp/version_utils.py")
else:
    VERSION = get_version("keras_nlp/src/version_utils.py")

setup(
    name="keras-nlp",
    description=(
        "Industry-strength Natural Language Processing extensions for Keras."
    ),
    long_description=README,
    long_description_content_type="text/markdown",
    version=VERSION,
    url="https://github.com/keras-team/keras-nlp",
    author="Keras team",
    author_email="keras-nlp@google.com",
    license="Apache License 2.0",
    install_requires=[
        "keras-core",
        "absl-py",
        "numpy",
        "packaging",
        "regex",
        "rich",
        "dm-tree",
        "kagglehub",
        # Don't require tensorflow-text on MacOS, there are no binaries for ARM.
        # Also, we rely on tensorflow *transitively* through tensorflow-text.
        # This avoid a slowdown during `pip install keras-nlp` where pip would
        # download many version of both libraries to find compatible versions.
        "tensorflow-text; platform_system != 'Darwin'",
    ],
    extras_require={
        "extras": [
            "rouge-score",
            "sentencepiece",
        ],
    },
    # Supported Python versions
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    packages=find_packages(exclude=("*_test.py",)),
)
