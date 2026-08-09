import numpy as np

from keras_hub.src.models.gemma4.gemma4_vision_encoder import (
    Gemma4VisionEncoder,
)
from keras_hub.src.models.gemma4.gemma4_vision_encoder import (
    Gemma4VisionEncoderBlock,
)
from keras_hub.src.models.gemma4.gemma4_vision_encoder import (
    Gemma4VisionPatchEmbedder,
)
from keras_hub.src.tests.test_case import TestCase


class Gemma4VisionEncoderTest(TestCase):
    def test_encoder_block_output_shape(self):
        batch_size = 2
        max_images_per_prompt = 2
        image_size = 16
        hidden_dim = 8
        intermediate_dim = 16

        encoder_block = Gemma4VisionEncoderBlock(
            image_size=image_size,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=2,
            num_heads=2,
            head_dim=4,
            num_key_value_heads=2,
            patch_size=4,
        )

        patch_dim = 3 * 4 * 4
        num_patches = (image_size // 4) ** 2
        dummy_pixel_values = np.random.rand(
            batch_size, max_images_per_prompt, num_patches, patch_dim
        ).astype("float32")
        dummy_pixel_pos = np.ones(
            [batch_size, max_images_per_prompt, num_patches, 2], dtype="int32"
        )
        output = encoder_block(dummy_pixel_values, dummy_pixel_pos)

        # Depending on keras version, call might return a tuple or just the
        # output tensor
        # for layers. If tuple extract the first element.
        if isinstance(output, tuple):
            output = output[0]

        self.assertEqual(
            output.shape,
            (
                batch_size,
                max_images_per_prompt,
                num_patches,
                hidden_dim,
            ),
        )

    def test_vision_patch_embedder_shape(self):
        embedder = Gemma4VisionPatchEmbedder(
            image_size=16,
            patch_size=4,
            hidden_dim=8,
        )
        dummy_pixel_values = np.ones([1, 16, 3 * 4 * 4])
        dummy_pixel_pos = np.ones([1, 16, 2], dtype="int32")
        output = embedder(dummy_pixel_values, dummy_pixel_pos)
        self.assertEqual(output.shape, (1, 16, 8))

    def test_vision_encoder_output_shape(self):
        """End-to-end output shape: (batch*n_images, n_tokens, output_dim)."""
        encoder = Gemma4VisionEncoder(
            image_size=64,
            patch_size=4,
            hidden_dim=8,
            num_layers=2,
            num_heads=2,
            head_dim=4,
            num_key_value_heads=2,
            intermediate_dim=16,
            output_dim=128,
            pool_size=4,
        )
        dummy_pv = np.ones([2, 2, 256, 3 * 4 * 4], dtype="float32")

        # Generate 256 token positions for a 16x16 grid (64//4)
        positions = []
        for y in range(16):
            for x in range(16):
                positions.append([y, x])
        positions = np.array(positions, dtype="int32")
        dummy_pos = np.tile(positions[None, None, ...], (2, 2, 1, 1))

        output = encoder(
            {
                "pixel_values": dummy_pv,
                "pixel_position_ids": dummy_pos,
            }
        )
        self.assertEqual(output.shape, (2, 2, 16, 128))

    def test_num_vision_tokens_attribute(self):
        """Encoder exposes num_vision_tokens_per_image."""
        encoder = Gemma4VisionEncoder(
            image_size=64,
            patch_size=4,
            hidden_dim=8,
            num_layers=2,
            num_heads=2,
            head_dim=4,
            num_key_value_heads=2,
            intermediate_dim=16,
            output_dim=128,
            pool_size=4,
        )
        # (64//4)^2 // 4^2 = 256 // 16 = 16
        self.assertEqual(encoder.num_vision_tokens_per_image, 16)

    def test_vision_encoder_saved_model(self):
        input_data = {
            "pixel_values": np.ones([2, 1, 16, 3 * 4 * 4], dtype="float32"),
            "pixel_position_ids": np.ones([2, 1, 16, 2], dtype="int32"),
        }
        self.run_model_saving_test(
            cls=Gemma4VisionEncoder,
            init_kwargs={
                "image_size": 16,
                "patch_size": 4,
                "hidden_dim": 8,
                "num_layers": 2,
                "num_heads": 2,
                "head_dim": 4,
                "num_key_value_heads": 2,
                "intermediate_dim": 16,
                "output_dim": 8,
                "pool_size": 2,
            },
            input_data=input_data,
        )
