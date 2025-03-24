from keras import ops
from keras_hub.src.models.video_swin.video_swin_layers import (
    VideoSwinPatchingAndEmbedding,
)
from keras_hub.src.models.video_swin.video_swin_layers import (
    VideoSwinPatchMerging,
)
from keras_hub.src.models.video_swin.video_swin_layers import (
    VideoSwinWindowAttention,
)
from keras_hub.src.tests.test_case import TestCase


class TestVideoSwinPatchingAndEmbedding(TestCase):
    def test_patch_embedding_compute_output_shape(self):
        patch_embedding_model = VideoSwinPatchingAndEmbedding(
            patch_size=(2, 4, 4), embed_dim=96, norm_layer=None
        )
        input_array = ops.ones(shape=(1, 16, 32, 32, 3))
        output_shape = patch_embedding_model(input_array).shape
        expected_output_shape = (1, 8, 8, 8, 96)
        self.assertEqual(output_shape, expected_output_shape)

    def test_patch_embedding_get_config(self):
        patch_embedding_model = VideoSwinPatchingAndEmbedding(
            patch_size=(4, 4, 4), embed_dim=96
        )
        config = patch_embedding_model.get_config()
        assert isinstance(config, dict)
        assert config["patch_size"] == (4, 4, 4)
        assert config["embed_dim"] == 96


class TestVideoSwinWindowAttention(TestCase):

    def setUp(self):
        self.window_attention_model = VideoSwinWindowAttention(
            input_dim=32,
            window_size=(2, 4, 4),
            num_heads=8,
            qkv_bias=True,
            qk_scale=None,
            attn_drop_rate=0.1,
            proj_drop_rate=0.1,
        )

    def test_window_attention_output_shape(self):
        input_shape = (2, 16, 32)
        input_array = ops.ones(input_shape)
        output_shape = self.window_attention_model(input_array).shape
        expected_output_shape = input_shape
        self.assertEqual(output_shape, expected_output_shape)

    def test_window_attention_get_config(self):
        config = self.window_attention_model.get_config()
        # Add assertions based on the specific requirements
        assert isinstance(config, dict)
        assert config["window_size"] == (2, 4, 4)
        assert config["num_heads"] == 8
        assert config["qkv_bias"] is True
        assert config["qk_scale"] is None
        assert config["attn_drop_rate"] == 0.1
        assert config["proj_drop_rate"] == 0.1


class TestVideoSwinPatchMerging(TestCase):
    def setUp(self):
        self.patch_merging = VideoSwinPatchMerging(input_dim=32)

    def test_output_shape(self):
        input_shape = (2, 4, 32, 32, 3)
        input_tensor = ops.ones(input_shape)
        output_shape = self.patch_merging(input_tensor).shape
        expected_shape = (
            input_shape[0],
            input_shape[1],
            input_shape[2] // 2,
            input_shape[3] // 2,
            2 * 32,
        )
        self.assertEqual(output_shape, expected_shape)
