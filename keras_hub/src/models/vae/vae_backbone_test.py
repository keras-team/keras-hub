import pytest
from keras import ops

from keras_hub.src.models.vae.vae_backbone import VAEBackbone
from keras_hub.src.tests.test_case import TestCase


class VAEBackboneTest(TestCase):
    def setUp(self):
        self.height, self.width = 64, 64
        self.init_kwargs = {
            "encoder_num_filters": [32, 32, 32, 32],
            "encoder_num_blocks": [1, 1, 1, 1],
            "decoder_num_filters": [32, 32, 32, 32],
            "decoder_num_blocks": [1, 1, 1, 1],
            # Use `mode` generate a deterministic output.
            "sampler_method": "mode",
        }
        self.input_data = ops.ones((2, self.height, self.width, 3))

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=VAEBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, self.height, self.width, 3),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=VAEBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=VAEBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            comparison_mode="statistical",
            output_thresholds={"*": {"max": 3e-3, "mean": 3e-4}},
        )
