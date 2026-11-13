import numpy as np
from keras import ops

from keras_hub.src.models.unlimited_ocr.unlimited_ocr_vision_encoder import (
    UnlimitedOCRVisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase


class UnlimitedOCRVisionEncoderTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "sam_hidden_dim": 16,
            "sam_num_layers": 2,
            "sam_num_heads": 2,
            "sam_mlp_dim": 32,
            "sam_output_channels": 8,
            "sam_patch_size": 8,
            "sam_image_size": 128,
            "sam_window_size": 4,
            "sam_global_attn_indexes": (1,),
            "sam_downsample_channels": (12, 16),
            "clip_hidden_dim": 16,
            "clip_num_layers": 2,
            "clip_num_heads": 2,
            "clip_intermediate_dim": 32,
            "clip_image_size": 48,
            "clip_patch_size": 16,
        }
        self.encoder = UnlimitedOCRVisionEncoder(**self.init_kwargs)

    def test_vision_encoder_layer(self):
        self.run_layer_test(
            cls=UnlimitedOCRVisionEncoder,
            init_kwargs=self.init_kwargs,
            input_data=np.random.uniform(size=(2, 64, 64, 3)).astype("float32"),
            # A 64 pixel image gives an 8x8 patch grid and a 2x2 feature grid.
            expected_output_shape=(2, 4, 32),
            # Patch embedding (2), position table (1), two SAM blocks (28),
            # neck (6), two strided convolutions (2), CLIP embedding (2),
            # CLIP pre norm (2), and two CLIP layers (32).
            expected_num_trainable_weights=75,
        )

    def test_vision_encoder_resizes_position_tables(self):
        # The same weights must run at the global size and at a tile size.
        # The position tables resize, because both sizes differ from the
        # table sizes.
        encoder = self.encoder
        global_features = encoder(np.zeros((1, 128, 128, 3), "float32"))
        tile_features = encoder(np.zeros((3, 80, 80, 3), "float32"))
        # 128 pixels: 16x16 patches, 8x8 after the first stride, 4x4 after
        # the second. 80 pixels: 10x10 patches, then 5x5, then 3x3.
        self.assertEqual(tuple(global_features.shape), (1, 16, 32))
        self.assertEqual(tuple(tile_features.shape), (3, 9, 32))

    def test_output_concatenates_clip_before_sam(self):
        # The last channels are the SAM features. The first are CLIP.
        encoder = self.encoder
        image = np.random.uniform(size=(1, 64, 64, 3)).astype("float32")
        output = ops.convert_to_numpy(encoder(image))
        x = encoder.sam_patch_embedding(image)
        x = encoder.sam_position_embedding(x)
        for block in encoder.sam_blocks:
            x = block(x)
        x = encoder.sam_neck_norm_1(encoder.sam_neck_conv_1(x))
        x = encoder.sam_neck_norm_2(encoder.sam_neck_conv_2(x))
        x = encoder.sam_net_2(encoder.sam_net_2_padding(x))
        x = encoder.sam_net_3(encoder.sam_net_3_padding(x))
        sam = ops.convert_to_numpy(x)
        self.assertAllClose(
            output[..., 16:], sam.reshape(1, 4, 16), atol=1e-5, rtol=1e-5
        )

    def test_hidden_dims_must_match(self):
        kwargs = dict(self.init_kwargs)
        kwargs["clip_hidden_dim"] = 8
        with self.assertRaisesRegex(ValueError, "clip_hidden_dim"):
            UnlimitedOCRVisionEncoder(**kwargs)
