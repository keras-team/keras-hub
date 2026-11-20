from unittest import mock

import pytest

from keras_hub.src.utils import tensor_utils


class _Item:
    """The parts of a pytest item that the conftest hook reads."""

    nodeid = "keras_hub/src/tests/requires_tf_marker_test.py::stand_in"

    def __init__(self, *marks):
        self.keywords = dict.fromkeys(marks, True)
        self.skip_conditions = []

    def add_marker(self, marker):
        if marker.name == "skipif":
            self.skip_conditions.append(marker.args[0])


@pytest.mark.parametrize("tf_installed", [True, False])
def test_requires_tf_skips_only_without_tensorflow(request, tf_installed):
    conftest = request.config.pluginmanager.get_plugin(
        str(request.config.rootpath / "conftest.py")
    )
    marked, unmarked = _Item("requires_tf"), _Item()
    with mock.patch.object(
        tensor_utils, "tf", object() if tf_installed else None
    ):
        conftest.pytest_collection_modifyitems(
            request.config, [marked, unmarked]
        )
    assert any(marked.skip_conditions) == (not tf_installed)
    assert not any(unmarked.skip_conditions)


def test_requires_tf_is_registered(request):
    markers = request.config.getini("markers")
    assert any(m.startswith("requires_tf:") for m in markers)
