import pathlib
from unittest import mock

import pytest
from absl.testing import parameterized

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils import tensor_utils

CONFTEST_PATH = pathlib.Path(__file__).resolve().parents[3] / "conftest.py"


class _Item:
    """The parts of a pytest item that the conftest hook reads."""

    nodeid = "keras_hub/src/tests/requires_tf_marker_test.py::stand_in"

    def __init__(self, *marks):
        self.keywords = dict.fromkeys(marks, True)
        self.skip_markers = []

    def add_marker(self, marker):
        if marker.name == "skipif":
            self.skip_markers.append(
                (marker.args[0], marker.kwargs.get("reason"))
            )


class RequiresTfMarkerTest(TestCase):
    @pytest.fixture(autouse=True)
    def _inject_pytest_config(self, request):
        self._config = request.config

    @parameterized.named_parameters(
        ("with_tf", True),
        ("without_tf", False),
    )
    def test_requires_tf_skips_only_without_tensorflow(self, tf_installed):
        conftest = self._config.pluginmanager.get_plugin(
            str(CONFTEST_PATH)
        ) or self._config.pluginmanager.get_plugin(
            str(self._config.rootpath / "conftest.py")
        )
        self.assertIsNotNone(conftest)
        marked, unmarked = _Item("requires_tf", "large"), _Item("large")
        with mock.patch.object(
            tensor_utils, "tf", object() if tf_installed else None
        ):
            conftest.pytest_collection_modifyitems(
                self._config, [marked, unmarked]
            )
        self.assertIn(
            (not tf_installed, "TensorFlow is not installed."),
            marked.skip_markers,
        )
        self.assertTrue(
            all(
                reason != "TensorFlow is not installed."
                for _, reason in unmarked.skip_markers
            )
        )

    def test_requires_tf_is_registered(self):
        markers = self._config.getini("markers")
        self.assertTrue(any(m.startswith("requires_tf:") for m in markers))
