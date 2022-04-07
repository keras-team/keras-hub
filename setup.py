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

import pathlib

from setuptools import find_packages
from setuptools import setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="keras-nlp",
    description=(
        "Industry-strength Natural Language Processing extensions for Keras."
    ),
    long_description=README,
    long_description_content_type="text/markdown",
    version="0.1.1",
    url="https://github.com/keras-team/keras-nlp",
    author="Keras team",
    author_email="keras-nlp@google.com",
    license="Apache License 2.0",
    install_requires=[
        "absl-py",
        "numpy",
        "packaging",
        "tensorflow",
        "tensorflow_text",
    ],
    extras_require={
        "tests": [
            "black",
            "flake8",
            "isort",
            "pytest",
            "pytest-cov",
        ],
        "examples": [
            "datasets",  # For GLUE in BERT example.
            "nltk",
            "wikiextractor",
        ],
    },
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    packages=find_packages(exclude=("*_test.py",)),
)
