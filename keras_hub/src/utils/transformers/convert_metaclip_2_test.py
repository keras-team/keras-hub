import numpy as np
import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.metaclip_2.metaclip_2_backbone import (
    MetaCLIP2Backbone,
)
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_huge_preset(self):
        model = MetaCLIP2Backbone.from_preset(
            "hf://facebook/metaclip-2-worldwide-huge-quickgelu",
        )
        # Test with dummy image and token inputs
        images = np.ones((1, 224, 224, 3), dtype="float32")
        token_ids = np.ones((1, 77), dtype="int32")
        outputs = model.predict({"images": images, "token_ids": token_ids})
        # Check output shapes
        self.assertEqual(outputs["vision_logits"].shape, (1, 1))
        self.assertEqual(outputs["text_logits"].shape, (1, 1))

    @pytest.mark.large
    def test_class_detection(self):
        model = Backbone.from_preset(
            "hf://facebook/metaclip-2-worldwide-huge-quickgelu",
            load_weights=False,
        )
        self.assertIsInstance(model, MetaCLIP2Backbone)

    @pytest.mark.large
    def test_backbone_config_conversion(self):
        """Test that config conversion produces correct architecture."""
        from keras_hub.src.utils.transformers import convert_metaclip_2

        # Simulated HuggingFace config for metaclip-2-worldwide-huge-quickgelu
        hf_config = {
            "model_type": "metaclip_2",
            "projection_dim": 1024,
            "vision_config": {
                "hidden_size": 1280,
                "num_hidden_layers": 32,
                "num_attention_heads": 16,
                "intermediate_size": 5120,
                "hidden_act": "quick_gelu",
                "image_size": 224,
                "patch_size": 14,
            },
            "text_config": {
                "vocab_size": 901629,
                "hidden_size": 1024,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "intermediate_size": 4096,
                "hidden_act": "quick_gelu",
                "max_position_embeddings": 77,
            },
        }

        keras_config = convert_metaclip_2.convert_backbone_config(hf_config)

        # Verify vision encoder config
        vision_encoder = keras_config["vision_encoder"]
        self.assertEqual(vision_encoder.hidden_dim, 1280)
        self.assertEqual(vision_encoder.num_layers, 32)
        self.assertEqual(vision_encoder.num_heads, 16)
        self.assertEqual(vision_encoder.intermediate_dim, 5120)
        self.assertEqual(vision_encoder.patch_size, 14)
        self.assertEqual(vision_encoder.image_shape, (224, 224, 3))

        # Verify text encoder config
        text_encoder = keras_config["text_encoder"]
        self.assertEqual(text_encoder.vocabulary_size, 901629)
        self.assertEqual(text_encoder.hidden_dim, 1024)
        self.assertEqual(text_encoder.num_layers, 24)
        self.assertEqual(text_encoder.num_heads, 16)
        self.assertEqual(text_encoder.intermediate_dim, 4096)
        self.assertEqual(text_encoder.max_sequence_length, 77)

        # Verify projection dim
        self.assertEqual(keras_config["projection_dim"], 1024)

    @pytest.mark.large
    def test_backbone_instantiation_from_config(self):
        """Test that backbone can be instantiated from converted config."""
        from keras_hub.src.utils.transformers import convert_metaclip_2

        # Use a smaller config for faster testing
        hf_config = {
            "model_type": "metaclip_2",
            "projection_dim": 256,
            "vision_config": {
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 512,
                "hidden_act": "quick_gelu",
                "image_size": 56,
                "patch_size": 14,
            },
            "text_config": {
                "vocab_size": 1000,
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 512,
                "hidden_act": "quick_gelu",
                "max_position_embeddings": 16,
            },
        }

        keras_config = convert_metaclip_2.convert_backbone_config(hf_config)
        backbone = MetaCLIP2Backbone(**keras_config)

        # Test forward pass
        images = np.ones((2, 56, 56, 3), dtype="float32")
        token_ids = np.ones((2, 16), dtype="int32")
        outputs = backbone({"images": images, "token_ids": token_ids})

        # Check output shapes
        self.assertEqual(outputs["vision_logits"].shape, (2, 2))
        self.assertEqual(outputs["text_logits"].shape, (2, 2))

    @pytest.mark.extra_large
    def test_compare_numerics_with_hf(self):
        transformers = pytest.importorskip("transformers")
        torch = pytest.importorskip("torch")

        keras_model = MetaCLIP2Backbone.from_preset(
            "hf://facebook/metaclip-2-worldwide-huge-quickgelu",
        )
        hf_model = transformers.AutoModel.from_pretrained(
            "facebook/metaclip-2-worldwide-huge-quickgelu",
        )
        hf_model.eval()

        rng = np.random.RandomState(123)
        images = rng.standard_normal((1, 224, 224, 3)).astype("float32")
        token_ids = np.tile(np.arange(77, dtype="int32"), (1, 1))
        keras_outputs = keras_model(
            {"images": images, "token_ids": token_ids},
            training=False,
        )

        torch_images = torch.from_numpy(images).permute(0, 3, 1, 2)
        torch_token_ids = torch.from_numpy(token_ids).to(torch.long)
        with torch.no_grad():
            hf_outputs = hf_model(
                pixel_values=torch_images,
                input_ids=torch_token_ids,
                attention_mask=torch.ones_like(torch_token_ids),
            )

        hf_vision_logits = hf_outputs.logits_per_image.detach().cpu().numpy()
        hf_text_logits = hf_outputs.logits_per_text.detach().cpu().numpy()
        self.assertAllClose(
            keras_outputs["vision_logits"],
            hf_vision_logits,
            atol=1e-4,
            rtol=1e-4,
        )
        self.assertAllClose(
            keras_outputs["text_logits"],
            hf_text_logits,
            atol=1e-4,
            rtol=1e-4,
        )
