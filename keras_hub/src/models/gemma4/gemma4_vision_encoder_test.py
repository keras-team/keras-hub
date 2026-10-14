import keras
import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.gemma4.gemma4_vision_encoder import (
    Gemma4VisionAveragePooling,
)
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

    def _tiny_encoder(self):
        return Gemma4VisionEncoder(
            image_size=16,
            patch_size=4,
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            head_dim=4,
            num_key_value_heads=2,
            intermediate_dim=16,
            output_dim=8,
            pool_size=2,
        )

    def test_vision_encoder_zero_patches(self):
        # A text-only batch on a multimodal model hands the encoder an empty
        # media axis. It must produce an empty output, not crash or return
        # the unpooled input. The second case keeps an image slot with zero
        # patches, which pins the slice back to a zero-length pooled axis.
        encoder = self._tiny_encoder()
        for max_images in (0, 1):
            out = encoder(
                {
                    "pixel_values": np.zeros(
                        [2, max_images, 0, 3 * 4 * 4], "float32"
                    ),
                    "pixel_position_ids": np.zeros(
                        [2, max_images, 0, 2], "int32"
                    ),
                }
            )
            self.assertEqual(tuple(out.shape), (2, max_images, 0, 8))

    @pytest.mark.tf_only
    def test_vision_encoder_predict_retrace(self):
        # On the TensorFlow backend a second `predict()` with a different
        # media-shape class makes Keras retrace with relaxed dims; a Python
        # branch on the patch count raises `OperatorNotAllowedInGraphError`.
        with_images = {
            "pixel_values": np.ones([2, 1, 16, 3 * 4 * 4], "float32"),
            "pixel_position_ids": np.stack(
                [np.array([[x, y] for y in range(4) for x in range(4)])] * 2
            )[:, None].astype("int32"),
        }
        no_images = {
            "pixel_values": np.zeros([2, 0, 0, 3 * 4 * 4], "float32"),
            "pixel_position_ids": np.zeros([2, 0, 0, 2], "int32"),
        }
        for order in ([no_images, with_images], [with_images, no_images]):
            encoder = self._tiny_encoder()
            outputs = [encoder.predict(x, verbose=0) for x in order]
            shapes = sorted(tuple(o.shape) for o in outputs)
            self.assertEqual(shapes, [(2, 0, 0, 8), (2, 1, 4, 8)])
            # The retraced graph must compute the same values as eager.
            traced = outputs[0 if order[0] is with_images else 1]
            eager = ops.convert_to_numpy(encoder(with_images))
            self.assertAllClose(traced, eager, atol=1e-5)

    @pytest.mark.tf_only
    def test_average_pooling_3d_predict_retrace(self):
        # The pooling layer also accepts a 3-D `(batch, patches, hidden)`
        # input; pin that path under a retrace as well.
        positions = np.stack(
            [np.array([[x, y] for y in range(4) for x in range(4)])] * 2
        ).astype("int32")
        with_patches = [np.ones([2, 16, 8], "float32"), positions]
        no_patches = [
            np.zeros([2, 0, 8], "float32"),
            np.zeros([2, 0, 2], "int32"),
        ]
        for order in ([no_patches, with_patches], [with_patches, no_patches]):
            inputs_in = keras.Input((None, 8))
            positions_in = keras.Input((None, 2), dtype="int32")
            layer = Gemma4VisionAveragePooling(
                image_size=16, patch_size=4, pool_size=2
            )
            model = keras.Model(
                [inputs_in, positions_in], layer(inputs_in, positions_in)
            )
            outputs = [model.predict(x, verbose=0) for x in order]
            shapes = sorted(tuple(o.shape) for o in outputs)
            self.assertEqual(shapes, [(2, 0, 8), (2, 4, 8)])
            traced = outputs[0 if order[0] is with_patches else 1]
            eager = ops.convert_to_numpy(layer(*with_patches))
            self.assertAllClose(traced, eager, atol=1e-5)
