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

    def test_backend_decides_when_vllm_cannot_say(self):
        # The tie-breaker exists for GPU boxes with tpu-inference
        # installed. Where vLLM cannot answer it falls back to whether the
        # TPU backend is installed, so neither path ends up unregistered.
        from keras_hub.src.vllm.plugin import serves_on_tpu

        try:
            import tpu_inference  # noqa: F401

            installed = True
        except ImportError:
            installed = False
        self.assertEqual(serves_on_tpu(), installed)

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
