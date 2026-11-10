import os

import keras
import pytest


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
    # Monkey-patch training methods for OpenVINO backend
    if keras.config.backend() == "openvino":
        keras.Model.fit = lambda *args, **kwargs: pytest.skip(
            "Model.fit() not supported on OpenVINO backend"
        )
        keras.Model.train_on_batch = lambda *args, **kwargs: pytest.skip(
            "Model.train_on_batch() not supported on OpenVINO backend"
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
    openvino_excluded_tests = set()
    if keras.config.backend() == "openvino":
        openvino_excluded_tests = read_test_list(
            os.path.join(
                os.path.dirname(__file__), "openvino_excluded_tests.txt"
            )
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

        if item.nodeid in openvino_excluded_tests:
            item.add_marker(pytest.mark.skip(reason="OpenVINO: Known failure"))


def read_test_list(path):
    with open(path) as f:
        lines = (line.strip() for line in f)
        return {line for line in lines if line and not line.startswith("#")}


# Disable traceback filtering for quicker debugging of tests failures.
keras.config.disable_traceback_filtering()
