import numpy as np

from keras_hub.src.models.mistral.mistral_vision_encoder import (
    Mistral3VisionAttention,
)
from keras_hub.src.models.mistral.mistral_vision_encoder import (
    Mistral3VisionEncoder,
)
from keras_hub.src.models.mistral.mistral_vision_encoder import (
    Mistral3VisionEncoderLayer,
)
from keras_hub.src.models.mistral.mistral_vision_encoder import (
    Mistral3VisionMLP,
)
from keras_hub.src.models.mistral.mistral_vision_encoder import (
    Mistral3VisionRotaryEmbedding,
)
from keras_hub.src.tests.test_case import TestCase


class Mistral3VisionEncoderTest(TestCase):
    def setUp(self):
        self.image_size = 16
        self.patch_size = 4
        self.hidden_dim = 8
        self.num_heads = 2
        self.head_dim = self.hidden_dim // self.num_heads
        self.intermediate_dim = 16
        self.num_layers = 2
        # Deliberately != num_heads to guard against a broadcast bug where
        # the rotary embedding was expanded on the wrong axis.
        self.batch_size = 3
        self.num_patches_per_side = self.image_size // self.patch_size
        self.sequence_length = self.num_patches_per_side**2

    def test_rotary_embedding_output_shape(self):
        rope = Mistral3VisionRotaryEmbedding(
            image_size=self.image_size,
            patch_size=self.patch_size,
            head_dim=self.head_dim,
        )
        position_ids = np.zeros(
            (self.batch_size, self.sequence_length), dtype="int32"
        )
        cos, sin = rope(position_ids)
        expected_shape = (self.batch_size, self.sequence_length, self.head_dim)
        self.assertEqual(cos.shape, expected_shape)
        self.assertEqual(sin.shape, expected_shape)

    def test_attention_output_shape(self):
        attention = Mistral3VisionAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
        )
        inputs = np.random.rand(
            self.batch_size, self.sequence_length, self.hidden_dim
        )
        position_ids = np.zeros(
            (self.batch_size, self.sequence_length), dtype="int32"
        )
        rope = Mistral3VisionRotaryEmbedding(
            image_size=self.image_size,
            patch_size=self.patch_size,
            head_dim=self.head_dim,
        )
        cos, sin = rope(position_ids)
        output = attention(inputs, position_embeddings=(cos, sin))
        self.assertEqual(
            output.shape,
            (self.batch_size, self.sequence_length, self.hidden_dim),
        )

    def test_mlp_output_shape(self):
        mlp = Mistral3VisionMLP(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
        )
        inputs = np.random.rand(
            self.batch_size, self.sequence_length, self.hidden_dim
        )
        output = mlp(inputs)
        self.assertEqual(
            output.shape,
            (self.batch_size, self.sequence_length, self.hidden_dim),
        )

    def test_encoder_layer_output_shape(self):
        layer = Mistral3VisionEncoderLayer(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_heads=self.num_heads,
        )
        inputs = np.random.rand(
            self.batch_size, self.sequence_length, self.hidden_dim
        )
        position_ids = np.zeros(
            (self.batch_size, self.sequence_length), dtype="int32"
        )
        rope = Mistral3VisionRotaryEmbedding(
            image_size=self.image_size,
            patch_size=self.patch_size,
            head_dim=self.head_dim,
        )
        cos, sin = rope(position_ids)
        output = layer(inputs, position_embeddings=(cos, sin))
        self.assertEqual(
            output.shape,
            (self.batch_size, self.sequence_length, self.hidden_dim),
        )

    def test_encoder_output_shape(self):
        encoder = Mistral3VisionEncoder(
            image_size=self.image_size,
            patch_size=self.patch_size,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            intermediate_dim=self.intermediate_dim,
        )
        # HF-style [batch, channels, height, width] input.
        pixel_values = np.random.rand(
            self.batch_size, 3, self.image_size, self.image_size
        )
        output = encoder(pixel_values)
        self.assertEqual(
            output.shape,
            (self.batch_size, self.sequence_length, self.hidden_dim),
        )
