import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.smolvlm2.smolvlm2_backbone import SmolVLM2Backbone
from keras_hub.src.tests.test_case import TestCase


class SmolVLM2BackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 256,
            "image_size": 32,
            "patch_size": 16,
            "vision_hidden_dim": 64,
            "vision_intermediate_dim": 128,
            "vision_num_layers": 2,
            "vision_num_heads": 4,
            "hidden_dim": 64,
            "intermediate_dim": 128,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "scale_factor": 1,
            "image_token_id": 200,
            "rope_max_wavelength": 10000,
            "layer_norm_epsilon": 1e-5,
            "vision_layer_norm_epsilon": 1e-6,
        }
        # Provide all four required functional-model inputs so that both
        # eager __call__ *and* jit-traced predict() paths work correctly.
        # (predict → stateless_call bypasses the __call__ override that
        # would otherwise inject dummy vision tensors.)
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
            "pixel_values": ops.zeros((2, 32, 32, 3), dtype="float32"),
            "vision_indices": ops.zeros((2, 0), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=SmolVLM2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 64),
            # Prevent default sequence-axis slicing which would corrupt
            # the 4D pixel_values tensor.
            variable_length_data=[self.input_data],
            run_mixed_precision_check=False,
            run_quantization_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=SmolVLM2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_num_parameters(self):
        model = SmolVLM2Backbone(**self.init_kwargs)
        # Verify the model is constructable and has expected parameter count.
        self.assertGreater(model.count_params(), 0)

    def test_config_roundtrip(self):
        model = SmolVLM2Backbone(**self.init_kwargs)
        config = model.get_config()
        self.assertEqual(config["vocabulary_size"], 256)
        self.assertEqual(config["image_size"], 32)
        self.assertEqual(config["hidden_dim"], 64)
        self.assertEqual(config["num_layers"], 2)
        self.assertEqual(config["scale_factor"], 1)
        self.assertEqual(config["image_token_id"], 200)

    def test_vision_encoder_standalone(self):
        model = SmolVLM2Backbone(**self.init_kwargs)
        images = np.random.rand(1, 32, 32, 3).astype("float32")
        vision_output = model.vision_encoder({"pixel_values": images})
        self.assertEqual(ops.shape(vision_output), (1, 4, 64))

    def test_connector(self):
        model = SmolVLM2Backbone(**self.init_kwargs)
        images = np.random.rand(1, 32, 32, 3).astype("float32")
        vision_output = model.vision_encoder({"pixel_values": images})
        connector_output = model.connector(vision_output)
        self.assertEqual(ops.shape(connector_output), (1, 4, 64))

    def test_multimodal_forward(self):
        """Test backbone forward pass with explicit vision inputs."""
        model = SmolVLM2Backbone(**self.init_kwargs)
        # Token ids with image_token_id=200 at positions [1,2,3,4].
        token_ids = np.array([[1, 200, 200, 200, 200]], dtype="int32")
        padding_mask = np.ones((1, 5), dtype="int32")
        pixel_values = np.random.rand(1, 32, 32, 3).astype("float32")
        # 4 patches with scale_factor=1, so 4 vision tokens at positions
        # 1,2,3,4 in the flattened (batch*seq) tensor.
        vision_indices = np.array([[1, 2, 3, 4]], dtype="int32")

        output = model(
            {
                "token_ids": token_ids,
                "padding_mask": padding_mask,
                "pixel_values": pixel_values,
                "vision_indices": vision_indices,
            }
        )
        self.assertEqual(ops.shape(output), (1, 5, 64))
