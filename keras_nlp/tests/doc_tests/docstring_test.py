# Copyright 2023 The KerasNLP Authors
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
from keras_nlp.tests.doc_tests.fenced_docstring_lib import (
    astor,  # For checking conditional import.
)

PACKAGE = "keras_nlp."


def find_modules():
    keras_nlp_modules = []
    for name, module in sys.modules.items():
        if name.startswith(PACKAGE):
            keras_nlp_modules.append(module)

    return keras_nlp_modules


@pytest.fixture(scope="session")
def docstring_module(pytestconfig):
    return pytestconfig.getoption("docstring_module")


def test_docstrings(docstring_module):
    keras_nlp_modules = find_modules()
    # As of this writing, it doesn't seem like pytest support load_tests
    # protocol for unittest:
    #     https://docs.pytest.org/en/7.1.x/how-to/unittest.html
    # So we run the unittest.TestSuite manually and report the results back.
    runner = unittest.TextTestRunner()
    suite = unittest.TestSuite()
    for module in keras_nlp_modules:
        if docstring_module and docstring_module not in module.__name__:
            continue
        print(f"Adding tests for docstrings in {module.__name__}")
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
    result = runner.run(suite)
    if not result.wasSuccessful():
        print(result)
    assert result.wasSuccessful()


@pytest.mark.extra_large
@pytest.mark.skipif(
    astor is None,
    reason="This test requires `astor`. Please `pip install astor` to run.",
)
def test_fenced_docstrings(docstring_module):
    """Tests fenced code blocks in docstrings.

    This can only be run manually and will take many minutes. Run with:
    `pytest keras_nlp/tests/doc_tests/docstring_test.py --run_extra_large`

    To restrict the docstring you test, you can pass an additional
    --docstring_module flag. For example, to run only "bert" module tests:
    `pytest keras_nlp/tests/doc_tests/docstring_test.py --run_extra_large --docstring_module "models.bert"`
    """
    keras_nlp_modules = find_modules()

    runner = unittest.TextTestRunner()
    suite = unittest.TestSuite()
    for module in keras_nlp_modules:
        if docstring_module and docstring_module not in module.__name__:
            continue
        print(f"Adding tests for fenced docstrings in {module.__name__}")
        suite.addTest(
            doctest.DocTestSuite(
                module,
                test_finder=doctest.DocTestFinder(
                    exclude_empty=False,
                    parser=fenced_docstring_lib.FencedCellParser(
                        fence_label="python"
                    ),
                ),
                globs={
                    "_print_if_not_none": fenced_docstring_lib._print_if_not_none
                },
                extraglobs={
                    "tf": tf,
                    "np": np,
                    "os": os,
                    "keras": keras,
                    "keras_nlp": keras_nlp,
                    "io": io,
                    "sentencepiece": sentencepiece,
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
    result = runner.run(suite)
    if not result.wasSuccessful():
        print(result)
    assert result.wasSuccessful()
