import os

import keras
import pytest

from keras_hub.src.utils.keras_utils import get_backend_package


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
    parser.addoption(
        "--check_gpu",
        action="store_true",
        default=False,
        help="fail if a gpu is not present",
    )


def pytest_configure(config):
    # Skip training in tests on backends that can't compute gradients.
    if not getattr(keras.src.backend, "SUPPORTS_GRADIENT", True):
        keras.Model.fit = lambda *args, **kwargs: pytest.skip(
            "Model.fit() not supported on this backend"
        )
        keras.Model.train_on_batch = lambda *args, **kwargs: pytest.skip(
            "Model.train_on_batch() not supported on this backend"
        )

    # Verify that device has GPU and detected by backend
    if config.getoption("--check_gpu"):
        found_gpu = False
        backend = keras.config.backend()
        if backend == "jax":
            import jax

            try:
                found_gpu = bool(jax.devices("gpu"))
            except RuntimeError:
                found_gpu = False
        elif backend == "tensorflow":
            import tensorflow as tf

            found_gpu = bool(tf.config.list_logical_devices("GPU"))
        elif backend == "torch":
            import torch

            found_gpu = bool(torch.cuda.device_count())
        if not found_gpu:
            pytest.fail(f"No GPUs discovered on the {backend} backend.")

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
    config.addinivalue_line(
        "markers",
        "kaggle_key_required: mark test needing a kaggle key",
    )


def pytest_collection_modifyitems(config, items):
    run_extra_large_tests = config.getoption("--run_extra_large")
    run_large_tests = config.getoption("--run_large")

    # Messages to annotate skipped tests with.
    skip_large = pytest.mark.skipif(
        not run_large_tests,
        reason="need --run_large option to run",
    )
    skip_extra_large = pytest.mark.skipif(
        not run_extra_large_tests,
        reason="need --run_extra_large option to run",
    )
    tf_only = pytest.mark.skipif(
        not keras.config.backend() == "tensorflow",
        reason="tests only run on tf backend",
    )
    found_kaggle_key = all(
        [
            os.environ.get("KAGGLE_USERNAME", None),
            os.environ.get("KAGGLE_KEY", None),
        ]
    )
    kaggle_key_required = pytest.mark.skipif(
        not found_kaggle_key,
        reason="tests only run with a kaggle api key",
    )
    # A pluggable backend can list the tests it supports, in a text file at
    # the root of its repository.
    supported_tests = read_backend_test_list("keras_hub_tests.txt")
    skip_unsupported = pytest.mark.skip(
        reason=f"Not supported by the {keras.config.backend()} backend"
    )
    for item in items:
        if "large" in item.keywords and "extra_large" not in item.keywords:
            item.add_marker(skip_large)
        if "extra_large" in item.keywords:
            item.add_marker(skip_extra_large)
        if "tf_only" in item.keywords:
            item.add_marker(tf_only)
        if "kaggle_key_required" in item.keywords:
            item.add_marker(kaggle_key_required)

        # Skip tests that the pluggable backend doesn't list as passing.
        if supported_tests is not None and not any(
            item.nodeid == path
            or item.nodeid.startswith(path + "/")
            or item.nodeid.startswith(path + "::")
            for path in supported_tests
        ):
            item.add_marker(skip_unsupported)


def read_backend_test_list(filename):
    """Read a test list from the root of a pluggable backend's repository."""
    package = get_backend_package()
    if package is None:
        return None
    path = os.path.join(
        os.path.dirname(os.path.dirname(package.__file__)), filename
    )
    # An installed backend package has no test list.
    if not os.path.exists(path):
        return None
    with open(path) as file:
        return [
            stripped
            for line in file
            if (stripped := line.strip()) and not stripped.startswith("#")
        ]


# Disable traceback filtering for quicker debugging of tests failures.
keras.config.disable_traceback_filtering()
