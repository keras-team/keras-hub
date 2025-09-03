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
        # Store original methods in case we need to restore them
        if not hasattr(keras.Model, "_original_compile"):
            keras.Model._original_compile = keras.Model.compile
            keras.Model._original_fit = keras.Model.fit
            keras.Model._original_train_on_batch = keras.Model.train_on_batch

        keras.Model.compile = lambda *args, **kwargs: pytest.skip(
            "Model.compile() not supported on OpenVINO backend"
        )
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
    for item in items:
        if "large" in item.keywords:
            item.add_marker(skip_large)
        if "extra_large" in item.keywords:
            item.add_marker(skip_extra_large)
        if "tf_only" in item.keywords:
            item.add_marker(tf_only)
        if "kaggle_key_required" in item.keywords:
            item.add_marker(kaggle_key_required)

        # OpenVINO-specific test skipping
        if keras.config.backend() == "openvino":
            test_name = item.name.split("[")[0]
            test_path = str(item.fspath)

            # OpenVINO supported test paths
            openvino_supported_paths = [
                "keras-hub/integration_tests",
                "keras_hub/src/models/gemma",
                "keras_hub/src/models/gpt2",
                "keras_hub/src/models/mistral",
                "keras_hub/src/samplers/serialization_test.py",
                "keras_hub/src/tests/doc_tests/docstring_test.py",
                "keras_hub/src/tokenizers",
                "keras_hub/src/utils",
            ]

            # Skip specific problematic test methods
            specific_skipping_tests = {
                "test_backbone_basics": "Requires trainable backend",
                "test_score_loss": "Non-implemented roll operation",
                "test_layer_behaviors": "Requires trainable backend",
            }

            if test_name in specific_skipping_tests:
                item.add_marker(
                    pytest.mark.skipif(
                        True,
                        reason="OpenVINO: "
                        f"{specific_skipping_tests[test_name]}",
                    )
                )
                continue

            parts = test_path.replace("\\", "/").split("/")
            try:
                keras_hub_idx = parts.index("keras_hub")
                relative_test_path = "/".join(parts[keras_hub_idx:])
            except ValueError:
                relative_test_path = test_path

            is_whitelisted = any(
                relative_test_path == supported_path
                or relative_test_path.startswith(supported_path + "/")
                for supported_path in openvino_supported_paths
            )

            if not is_whitelisted:
                item.add_marker(
                    pytest.mark.skipif(
                        True, reason="OpenVINO: File/directory not in whitelist"
                    )
                )


# Disable traceback filtering for quicker debugging of tests failures.
keras.config.disable_traceback_filtering()
