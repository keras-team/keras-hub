from importlib.metadata import PackageNotFoundError
from importlib.metadata import distribution
from importlib.metadata import entry_points

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.vllm.plugin import register_keras_hub


class RegisterKerasHubTest(TestCase):
    def test_returns_without_the_tpu_backend(self):
        # The entry point runs in every vLLM process, including ones with no
        # TPU backend installed. It must be a no-op there, not an error.
        register_keras_hub()

    def test_tpu_registration_declines_when_it_is_not_the_platform(self):
        # The two engines share an architecture string, so the TPU
        # registration has to decline rather than race the GPU one.
        from keras_hub.src.vllm.plugin import _register_tpu_model

        self.assertFalse(_register_tpu_model())

    def test_entry_point_is_exposed(self):
        # vLLM reads entry points from the installed package metadata, so a
        # line in pyproject.toml only counts once it lands there.
        try:
            distribution("keras-hub")
        except PackageNotFoundError:
            self.skipTest("keras-hub is not installed")
        found = {
            e.name: e.value for e in entry_points(group="vllm.general_plugins")
        }
        self.assertEqual(
            found.get("register_keras_hub"),
            "keras_hub.src.vllm.plugin:register_keras_hub",
        )
