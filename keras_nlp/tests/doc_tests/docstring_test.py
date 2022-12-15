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
import os
import sys
import unittest

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

import keras_nlp
from keras_nlp.tests.doc_tests import docstring_lib

PACKAGE = "keras_nlp."


def find_modules():
    keras_nlp_modules = []
    for name, module in sys.modules.items():
        if name.startswith(PACKAGE):
            keras_nlp_modules.append(module)

    return keras_nlp_modules


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
        # Temporarily stop testing gpt2 docstrings until we are exporing the
        # symbols.
        if "gpt2" in module.__name__:
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
    result = runner.run(suite)
    if not result.wasSuccessful():
        print(result)
    assert result.wasSuccessful()
