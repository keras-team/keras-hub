import numpy as np
import pytest

from keras_hub.src.models.llama3.llama3_tokenizer import Llama3Tokenizer
from keras_hub.src.models.llama3.llama3_vision_backbone import (
    Llama3VisionBackbone,
)
from keras_hub.src.models.llama3.llama3_vision_causal_lm import (
    Llama3VisionCausalLM,
)
from keras_hub.src.models.llama3.llama3_vision_image_converter import (
    Llama3VisionImageConverter,
)
from keras_hub.src.models.llama3.llama3_vision_preprocessor import (
    Llama3VisionPreprocessor,
)
from keras_hub.src.tests.test_case import TestCase


class Llama3VisionCausalLMTest(TestCase):
    def setUp(self):
        self.backbone_kwargs = {
            "vocabulary_size": 100,
            "num_layers": 1,
            "hidden_dim": 16,
            "num_query_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_dim": 32,
            "vision_hidden_dim": 16,
            "vision_num_layers": 1,
            "vision_num_heads": 2,
            "vision_intermediate_dim": 32,
            "vision_patch_size": 4,
            "vision_image_size": 16,
            "cross_attention_layers": [],
        }
        self.backbone = Llama3VisionBackbone(**self.backbone_kwargs)

        self.vocab = ["!", "air", "Ġair", "plane", "Ġat", "port"]
        self.vocab += ["<|begin_of_text|>", "<|end_of_text|>"]
        self.vocab += ["<|start_header_id|>", "<|end_header_id|>"]
        self.vocab += ["<|eot_id|>"]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]

        self.tokenizer = Llama3Tokenizer(
            vocabulary=self.vocab,
            merges=self.merges,
        )
        self.converter = Llama3VisionImageConverter(image_size=(16, 16))
        self.preprocessor = Llama3VisionPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=self.converter,
            sequence_length=10,
        )
        self.model = Llama3VisionCausalLM(
            backbone=self.backbone,
            preprocessor=self.preprocessor,
        )

    def test_causal_lm_call(self):
        """Test forward pass."""
        from keras import ops

        inputs = {
            "text": ["airplane"],
            "images": np.random.randint(0, 255, (1, 32, 32, 3)).astype(
                "float32"
            ),
        }
        preprocessed = self.preprocessor(inputs)
        # Add aspect_ratio_ids since preprocessor doesn't produce it yet
        # Image converter produces multi-tile images (batch, num_tiles, H, W, C)
        # So aspect_ratio_ids/mask must match num_tiles dimension
        batch_size = preprocessed["token_ids"].shape[0]
        # Assuming num_tiles=1 for this test since image size is small
        num_tiles = preprocessed["pixel_values"].shape[1]

        preprocessed["aspect_ratio_ids"] = ops.ones(
            (batch_size, num_tiles), dtype="int32"
        )
        preprocessed["aspect_ratio_mask"] = ops.ones(
            (batch_size, num_tiles), dtype="int32"
        )
        # Convert pixel_values to multi-tile format (batch, num_tiles, H, W, C)
        # Preprocessor outputs (batch, H, W, C), add tile dimension
        pixel_values = ops.expand_dims(preprocessed["pixel_values"], axis=1)
        preprocessed["pixel_values"] = pixel_values
        outputs = self.model(preprocessed)

        text_seq_len = preprocessed["token_ids"].shape[1]
        vocab_size = self.backbone_kwargs["vocabulary_size"]
        self.assertEqual(outputs.shape, (batch_size, text_seq_len, vocab_size))

    def test_generate(self):
        """Test generation (skipped until preprocessor supports it)."""
        self.skipTest(
            "generate_preprocess not yet implemented for vision preprocessor"
        )

    @pytest.mark.large
    def test_serialization(self):
        """Test model serialization."""
        config = self.model.get_config()
        self.assertIn("backbone", config)
        self.assertIn("preprocessor", config)
