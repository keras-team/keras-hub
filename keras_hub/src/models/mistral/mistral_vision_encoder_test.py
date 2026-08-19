import numpy as np

from keras_hub.src.models.mistral.mistral_vision_encoder import (
    Mistral3ImageFeatureExtractor,
)
from keras_hub.src.models.mistral.mistral_vision_encoder import (
    Mistral3ImageTextEmbeddingMerger,
)
from keras_hub.src.models.mistral.mistral_vision_encoder import (
    Mistral3MultiModalProjector,
)
from keras_hub.src.models.mistral.mistral_vision_encoder import (
    Mistral3PatchMerger,
)
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
from keras_hub.src.models.mistral.mistral_vision_encoder import (
    compute_image_placeholder_indices,
)
from keras_hub.src.models.mistral.mistral_vision_encoder import (
    compute_pixtral_resize_size,
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
        self.text_hidden_dim = 12
        self.spatial_merge_size = 2

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
        # HF-style [num_images, channels, height, width] input. Images are
        # concatenated into a single sequence (block-diagonal attention
        # keeps them from attending to each other), matching Pixtral's
        # multi-image handling, so the output always has batch dim 1.
        pixel_values = np.random.rand(
            self.batch_size, 3, self.image_size, self.image_size
        )
        output = encoder(pixel_values)
        self.assertEqual(
            output.shape,
            (1, self.batch_size * self.sequence_length, self.hidden_dim),
        )

    def test_encoder_output_shape_with_variable_image_sizes(self):
        encoder = Mistral3VisionEncoder(
            image_size=self.image_size,
            patch_size=self.patch_size,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            intermediate_dim=self.intermediate_dim,
        )
        # Two images sharing a common padded canvas but with different real
        # sizes (one cropped in width).
        pixel_values = np.random.rand(2, 3, self.image_size, self.image_size)
        image_sizes = np.array([[16, 16], [16, 8]], dtype="int32")
        output = encoder(pixel_values, image_sizes=image_sizes)
        # 16 valid patches from image 0, 8 valid patches from image 1
        # (cropped to half width) = 24 total tokens.
        self.assertEqual(output.shape, (1, 24, self.hidden_dim))

    def test_patch_merger_output_shape(self):
        merger = Mistral3PatchMerger(
            hidden_dim=self.hidden_dim,
            spatial_merge_size=self.spatial_merge_size,
            patch_size=self.patch_size,
            image_size=self.image_size,
        )
        # Image 0: 4x4 patch grid (16 tokens, 4 merge windows).
        # Image 1: 4x8 patch grid (32 tokens, 8 merge windows).
        image_sizes = np.array([[16, 16], [16, 32]], dtype="int32")
        max_patch_height, max_patch_width = 4, 8
        image_features = np.random.rand(16 + 32, self.hidden_dim)
        merged, valid_count = merger(
            image_features,
            image_sizes=image_sizes,
            max_patch_height=max_patch_height,
            max_patch_width=max_patch_width,
        )
        capacity = 2 * (max_patch_height // 2) * (max_patch_width // 2)
        self.assertEqual(merged.shape, (capacity, self.hidden_dim))
        self.assertEqual(int(valid_count), 12)

    def test_patch_merger_serialization(self):
        merger = Mistral3PatchMerger(
            hidden_dim=self.hidden_dim,
            spatial_merge_size=self.spatial_merge_size,
            patch_size=self.patch_size,
            image_size=self.image_size,
        )
        self.run_serialization_test(merger)

    def test_multimodal_projector_output_shape(self):
        projector = Mistral3MultiModalProjector(
            vision_hidden_dim=self.hidden_dim,
            text_hidden_dim=self.text_hidden_dim,
            spatial_merge_size=self.spatial_merge_size,
            patch_size=self.patch_size,
            image_size=self.image_size,
        )
        # Two images: 4x4 and 4x8 patch grids (16 + 32 = 48 tokens).
        image_sizes = np.array([[16, 16], [16, 32]], dtype="int32")
        image_features = np.random.rand(48, self.hidden_dim)
        output = projector(
            image_features,
            image_sizes=image_sizes,
            max_patch_height=4,
            max_patch_width=8,
        )
        # Padding is sliced off before returning: 4 + 8 = 12 valid windows.
        self.assertEqual(output.shape, (12, self.text_hidden_dim))

    def test_multimodal_projector_serialization(self):
        projector = Mistral3MultiModalProjector(
            vision_hidden_dim=self.hidden_dim,
            text_hidden_dim=self.text_hidden_dim,
            spatial_merge_size=self.spatial_merge_size,
            patch_size=self.patch_size,
            image_size=self.image_size,
        )
        self.run_serialization_test(projector)

    def test_image_text_embedding_merger_scatters_features(self):
        merger = Mistral3ImageTextEmbeddingMerger()
        batch_size, seq_length, hidden_dim = 1, 5, 3
        token_embeddings = np.zeros((batch_size, seq_length, hidden_dim))
        image_features = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        placeholder_indices = np.array([1, 3], dtype="int32")
        output = merger(token_embeddings, image_features, placeholder_indices)
        expected = np.zeros((batch_size, seq_length, hidden_dim))
        expected[0, 1] = 1.0
        expected[0, 3] = 2.0
        self.assertAllClose(output, expected)

    def test_image_text_embedding_merger_flattens_batched_indices(self):
        # A batched `(batch, N)` placeholder_indices tensor (as produced by
        # a `keras.Input`) must be flattened before use, since values are
        # global indices into the flattened `(batch * seq_length,)`
        # sequence either way.
        merger = Mistral3ImageTextEmbeddingMerger()
        token_embeddings = np.zeros((1, 4, 2))
        image_features = np.array([[9.0, 9.0]])
        placeholder_indices_2d = np.array([[2]], dtype="int32")
        output = merger(
            token_embeddings, image_features, placeholder_indices_2d
        )
        expected = np.zeros((1, 4, 2))
        expected[0, 2] = 9.0
        self.assertAllClose(output, expected)

    def test_compute_image_placeholder_indices(self):
        token_ids = np.array([[1, 10, 3, 10], [10, 2, 3, 4]])
        indices = compute_image_placeholder_indices(
            token_ids, image_token_index=10
        )
        # Flat indices into the (batch * seq_length,) sequence.
        self.assertAllEqual(indices, np.array([1, 3, 4]))

    def test_compute_image_placeholder_indices_none_present(self):
        token_ids = np.array([[1, 2, 3]])
        indices = compute_image_placeholder_indices(
            token_ids, image_token_index=10
        )
        self.assertEqual(indices.shape, (0,))

    def test_get_mistral3_image_features_end_to_end_shape(self):
        vision_encoder = Mistral3VisionEncoder(
            image_size=self.image_size,
            patch_size=self.patch_size,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            intermediate_dim=self.intermediate_dim,
        )
        projector = Mistral3MultiModalProjector(
            vision_hidden_dim=self.hidden_dim,
            text_hidden_dim=self.text_hidden_dim,
            spatial_merge_size=self.spatial_merge_size,
            patch_size=self.patch_size,
            image_size=self.image_size,
        )
        # Two images padded to a common (16, 16) canvas; image 1 is
        # cropped to half width.
        pixel_values = np.random.rand(2, 3, self.image_size, self.image_size)
        image_sizes = np.array([[16, 16], [16, 8]], dtype="int32")
        extractor = Mistral3ImageFeatureExtractor(vision_encoder, projector)
        output = extractor(pixel_values, image_sizes)
        # Image 0: 4x4 patches -> 2x2 = 4 merge windows.
        # Image 1: 4x2 patches -> only column 0 qualifies -> 2 merge
        # windows (rows 0 and 2).
        self.assertEqual(output.shape, (6, self.text_hidden_dim))

    def test_get_mistral3_image_features_rejects_unsupported_layer(self):
        vision_encoder = Mistral3VisionEncoder(
            image_size=self.image_size,
            patch_size=self.patch_size,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            intermediate_dim=self.intermediate_dim,
        )
        projector = Mistral3MultiModalProjector(
            vision_hidden_dim=self.hidden_dim,
            text_hidden_dim=self.text_hidden_dim,
            spatial_merge_size=self.spatial_merge_size,
            patch_size=self.patch_size,
            image_size=self.image_size,
        )
        with self.assertRaises(NotImplementedError):
            Mistral3ImageFeatureExtractor(
                vision_encoder,
                projector,
                vision_feature_layer=-2,
            )

    def test_compute_pixtral_resize_size_exact_multiple(self):
        # Already a multiple of `patch_size`, well under `longest_edge`:
        # no scaling, no rounding.
        size = compute_pixtral_resize_size(
            height=16, width=16, longest_edge=32, patch_size=4
        )
        self.assertEqual(size, (16, 16))

    def test_compute_pixtral_resize_size_rounds_up(self):
        # Under `longest_edge`, but not a `patch_size` multiple: each dim
        # rounds up independently, (17 - 1) // 4 + 1 = 5 -> 20.
        size = compute_pixtral_resize_size(
            height=17, width=17, longest_edge=32, patch_size=4
        )
        self.assertEqual(size, (20, 20))

    def test_compute_pixtral_resize_size_clamps_wide_image(self):
        # Wide image: width is the longest edge, so it is scaled down to
        # `longest_edge` and height is scaled by the same ratio, preserving
        # aspect ratio, before rounding up to a `patch_size` multiple.
        # ratio = 40 / 16 = 2.5 -> height = floor(20 / 2.5) = 8,
        # width = floor(40 / 2.5) = 16 (both already patch multiples).
        size = compute_pixtral_resize_size(
            height=20, width=40, longest_edge=16, patch_size=4
        )
        self.assertEqual(size, (8, 16))

    def test_compute_pixtral_resize_size_clamps_tall_image(self):
        # Tall image: height is the longest edge. Same scale factor is
        # applied to both dimensions.
        # ratio = 40 / 16 = 2.5 -> height = floor(40 / 2.5) = 16,
        # width = floor(20 / 2.5) = 8.
        size = compute_pixtral_resize_size(
            height=40, width=20, longest_edge=16, patch_size=4
        )
        self.assertEqual(size, (16, 8))
