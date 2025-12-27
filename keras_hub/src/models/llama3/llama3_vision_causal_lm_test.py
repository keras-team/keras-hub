import numpy as np

from keras_hub.src.models.llama3.llama3_backbone import Llama3BackboneConfig
from keras_hub.src.models.llama3.llama3_tokenizer import Llama3Tokenizer
from keras_hub.src.models.llama3.llama3_vision_backbone import (
    Llama3VisionBackbone,
)
from keras_hub.src.models.llama3.llama3_vision_causal_lm import (
    Llama3VisionCausalLM,
)
from keras_hub.src.models.llama3.llama3_vision_config import Llama3VisionConfig
from keras_hub.src.models.llama3.llama3_vision_config import (
    Llama3VisionEncoderConfig,
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
        # 1. Configs
        vision_config = Llama3VisionEncoderConfig(
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            intermediate_dim=32,
            patch_size=4,
            image_size=16,
        )
        text_config = Llama3BackboneConfig(
            vocabulary_size=100,
            num_layers=1,
            num_heads=2,
            num_query_heads=2,
            num_key_value_heads=2,
            hidden_dim=16,
            intermediate_dim=32,
        )
        self.config = Llama3VisionConfig(
            vision_encoder_config=vision_config, text_config=text_config
        )

        # 2. Backbone
        self.backbone = Llama3VisionBackbone(self.config)

        # 3. Preprocessor Components
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

        # 4. The Model
        self.model = Llama3VisionCausalLM(
            backbone=self.backbone,
            preprocessor=self.preprocessor,
        )

    def test_causal_lm_call(self):
        """Test model with preprocessed inputs."""
        # Preprocess inputs first
        inputs = {
            "text": ["airplane"],
            "images": np.random.randint(0, 255, (1, 32, 32, 3)).astype(
                "float32"
            ),
        }

        preprocessed = self.preprocessor(inputs)

        # Call model with preprocessed inputs
        outputs = self.model(preprocessed)

        # Output shape should be (Batch, Text_Seq_Len, Vocab_Size)
        # With cross-attention, output matches TEXT sequence length
        # (not concatenated)
        # Vision features are injected via cross-attention, not
        # prepended
        batch_size = preprocessed["token_ids"].shape[0]
        text_seq_len = preprocessed["token_ids"].shape[1]
        vocab_size = self.config.text_config.vocabulary_size
        self.assertEqual(outputs.shape, (batch_size, text_seq_len, vocab_size))

    def test_generate(self):
        """Test generation.

        Skip for now as generate_preprocess is not yet implemented.
        """
        # TODO: Implement generate_preprocess and generate_postprocess
        # in Llama3VisionPreprocessor
        self.skipTest(
            "generate_preprocess not yet implemented for vision preprocessor"
        )

    def test_serialization(self):
        """Test get_config."""
        config = self.model.get_config()

        # Check config contains expected keys
        self.assertIn("backbone", config)
        self.assertIn("preprocessor", config)
