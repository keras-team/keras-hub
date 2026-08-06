from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.vllm.plugin import register_keras_hub


class RegisterKerasHubTest(TestCase):
    def test_returns_without_the_tpu_backend(self):
        # The entry point runs in every vLLM process, including ones with no
        # TPU backend installed. It must be a no-op there, not an error.
        register_keras_hub()

    def test_entry_point_is_declared(self):
        # The registration only ever runs if vLLM can find it.
        import tomllib

        with open("pyproject.toml", "rb") as f:
            config = tomllib.load(f)
        plugins = config["project"]["entry-points"]["vllm.general_plugins"]
        self.assertEqual(
            plugins["register_keras_hub"],
            "keras_hub.src.vllm.plugin:register_keras_hub",
        )
