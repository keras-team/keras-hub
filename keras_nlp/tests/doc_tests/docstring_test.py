# Copyright 2022 The KerasNLP Authors
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

import doctest
import io
import os
import re
import sys
import unittest

import numpy as np
import pytest
import sentencepiece
import tensorflow as tf
from tensorflow import keras

import keras_nlp
from keras_nlp.tests.doc_tests import docstring_lib
from keras_nlp.tests.doc_tests import fenced_docstring_lib

PACKAGE = "keras_nlp."
DIRECTORY = "keras_nlp"


def find_modules():
    keras_nlp_modules = []
    for name, module in sys.modules.items():
        if name.startswith(PACKAGE):
            keras_nlp_modules.append(module)

    return keras_nlp_modules


def find_files(regex_pattern=None):
    py_files = []
    for root, dirs, files in os.walk(DIRECTORY):
        for file in files:
            if file.endswith(".py"):
                if regex_pattern is not None and regex_pattern.search(file):
                    continue
                py_files.append(os.path.join(root, file))
    return py_files


@pytest.mark.skipif(
    sys.platform == "win32", reason="Numpy prints differently on windows"
)
def test_docstrings():
    keras_nlp_modules = find_modules()
    # As of this writing, it doesn't seem like pytest support load_tests
    # protocol for unittest:
    #     https://docs.pytest.org/en/7.1.x/how-to/unittest.html
    # So we run the unittest.TestSuite manually and report the results back.
    runner = unittest.TextTestRunner()
    suite = unittest.TestSuite()
    for module in keras_nlp_modules:
        # Temporarily stop testing gpt2 & deberta docstrings until we are
        # exporting the symbols.
        if "gpt2" in module.__name__ or "deberta_v3" in module.__name__:
            continue
        suite.addTest(
            doctest.DocTestSuite(
                module,
                test_finder=doctest.DocTestFinder(exclude_empty=False),
                extraglobs={
                    "tf": tf,
                    "np": np,
                    "os": os,
                    "keras": keras,
                    "keras_nlp": keras_nlp,
                },
                checker=docstring_lib.DoctestOutputChecker(),
                optionflags=(
                    doctest.ELLIPSIS
                    | doctest.NORMALIZE_WHITESPACE
                    | doctest.IGNORE_EXCEPTION_DETAIL
                    | doctest.DONT_ACCEPT_BLANKLINE
                ),
            )
        )
        suite.addTest(
            doctest.DocFileSuite(
                module,
                globs={
                    "_print_if_not_none": fenced_docstring_lib._print_if_not_none
                },
                extraglobs={
                    "tf": tf,
                    "np": np,
                    "os": os,
                    "keras": keras,
                    "keras_nlp": keras_nlp,
                },
                parser=fenced_docstring_lib.FencedCellParser(
                    fence_label="python"
                ),
                checker=fenced_docstring_lib.FencedCellOutputChecker(),
                optionflags=(
                    doctest.ELLIPSIS
                    | doctest.NORMALIZE_WHITESPACE
                    | doctest.IGNORE_EXCEPTION_DETAIL
                    | doctest.DONT_ACCEPT_BLANKLINE
                ),
            )
        )
    result = runner.run(suite)
    if not result.wasSuccessful():
        print(result)
    assert result.wasSuccessful()


@pytest.mark.skipif(
    sys.platform == "win32", reason="Numpy prints differently on windows"
)
def test_fenced_docstrings():

    regex_pattern = re.compile(
        r"|".join(
            [
                # Endswith patterns
                "test\\.py$",
                "__init__\\.py$",
                # Unexported symbols
                "deberta_v3",
                "gpt2",
                # Whole string matching
                "^keras_nlp/models/backbone\\.py$",
                "^keras_nlp/models/preprocessor\\.py$",
                "^keras_nlp/models/task\\.py$",
            ]
        )
    )
    keras_nlp_files = find_files(regex_pattern=regex_pattern)
    runner = unittest.TextTestRunner()
    suite = unittest.TestSuite()

    suite.addTest(
        doctest.DocFileSuite(
            *keras_nlp_files,
            module_relative=False,
            parser=fenced_docstring_lib.FencedCellParser(fence_label="python"),
            globs={
                "_print_if_not_none": fenced_docstring_lib._print_if_not_none,
                "tf": tf,
                "np": np,
                "os": os,
                "keras": keras,
                "keras_nlp": keras_nlp,
                "io": io,
                "sentencepiece": sentencepiece,
            },
            checker=fenced_docstring_lib.FencedCellOutputChecker(),
            optionflags=(
                doctest.ELLIPSIS
                | doctest.NORMALIZE_WHITESPACE
                | doctest.IGNORE_EXCEPTION_DETAIL
                | doctest.DONT_ACCEPT_BLANKLINE
            ),
        )
    )

    result = runner.run(suite)
    if not result.wasSuccessful():
        print(result)
    assert result.wasSuccessful()
