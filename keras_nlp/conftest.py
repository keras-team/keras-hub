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

import pytest
import tensorflow as tf

from keras_nlp.backend import config as backend_config
from keras_nlp.backend import keras


def pytest_addoption(parser):
    parser.addoption(
        "--run_large",
        action="store_true",
        default=False,
        help="run large tests",
    )
    parser.addoption(
        "--run_extra_large",
        action="store_true",
        default=False,
        help="run extra_large tests",
    )
    parser.addoption(
        "--docstring_module",
        action="store",
        default="",
        help="restrict docs testing to modules whose name matches this flag",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "large: mark test as being slow or requiring a network",
    )
    config.addinivalue_line(
        "markers",
        "extra_large: mark test as being too large to run continuously",
    )
    config.addinivalue_line(
        "markers",
        "tf_only: mark test as a tf only test",
    )


def pytest_collection_modifyitems(config, items):
    run_extra_large_tests = config.getoption("--run_extra_large")
    # Run large tests for --run_extra_large or --run_large.
    run_large_tests = config.getoption("--run_large") or run_extra_large_tests

    # Messages to annotate skipped tests with.
    skip_large = pytest.mark.skipif(
        not run_large_tests,
        reason="need --run_large option to run",
    )
    skip_extra_large = pytest.mark.skipif(
        not run_extra_large_tests,
        reason="need --run_extra_large option to run",
    )
    skip_tf_only = pytest.mark.skipif(
        not backend_config.backend() == "tensorflow",
        reason="tests only run on tf backend",
    )
    for item in items:
        if "large" in item.keywords:
            item.add_marker(skip_large)
        if "extra_large" in item.keywords:
            item.add_marker(skip_extra_large)
        if "tf_only" in item.keywords:
            item.add_marker(skip_tf_only)


# Disable traceback filtering for quicker debugging of tests failures.
tf.debugging.disable_traceback_filtering()
if backend_config.multi_backend():
    keras.config.disable_traceback_filtering()
