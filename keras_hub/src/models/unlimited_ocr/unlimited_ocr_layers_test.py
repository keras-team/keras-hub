import keras
import numpy as np
from keras import ops

from keras_hub.src.models.unlimited_ocr.unlimited_ocr_layers import (
    UnlimitedOCRCLIPEmbedding,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_layers import (
    UnlimitedOCRImageFeatureExtractor,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_layers import (
    UnlimitedOCRImageTextEmbeddingMerger,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_layers import (
    UnlimitedOCRSamPositionEmbedding,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_layers import (
    UnlimitedOCRSparseMoeBlock,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_layers import (
    _bicubic_antialias_matrix,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_layers import (
    _resize_bicubic,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_vision_encoder import (
    UnlimitedOCRVisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase


class UnlimitedOCRSparseMoeBlockTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "hidden_dim": 16,
            "moe_intermediate_dim": 8,
            "num_shared_experts": 2,
            "num_experts": 4,
            "top_k": 2,
        }

    def test_layer_basics(self):
        self.run_layer_test(
            cls=UnlimitedOCRSparseMoeBlock,
            init_kwargs=self.init_kwargs,
            input_data=np.random.uniform(size=(2, 5, 16)).astype("float32"),
            expected_output_shape=(2, 5, 16),
            # The router (1), the experts (2) and the shared experts (3).
            expected_num_trainable_weights=6,
        )

    def test_moe_block_output(self):
        block = UnlimitedOCRSparseMoeBlock(
            **self.init_kwargs, routed_scaling_factor=1.5
        )
        rng = np.random.RandomState(0)
        x = rng.normal(size=(2, 5, 16)).astype("float32")
        block(x)
        for weight in block.weights:
            weight.assign(rng.normal(size=weight.shape).astype("float32"))
        # Values from `UnlimitedOcrTextMoe` in `transformers` 5.22.2. It ran
        # on the same input and weights. The first four features are kept.
        expected = [
            [
                [11.2993, -13.7247, 24.6924, 98.1177],
                [37.8279, 27.4235, -2.6267, 1.0697],
                [-12.6825, -16.4118, -136.3914, 116.7998],
                [-8.8172, -57.0034, -51.8629, -4.8857],
                [-6.0046, 9.3282, -4.2227, 4.0653],
            ],
            [
                [39.4222, -62.1956, 2.2394, -5.1314],
                [-19.8916, 41.2361, 35.1987, 41.0857],
                [-33.0869, -66.0942, -57.0858, 33.1090],
                [-9.2994, 5.7807, 4.2842, -8.1439],
                [-2.1699, 19.2043, -7.7140, -89.7425],
            ],
        ]
        output = ops.convert_to_numpy(block(x))
        self.assertAllClose(output[:, :, :4], expected, atol=1e-3, rtol=1e-4)


class UnlimitedOCRResizeTest(TestCase):
    def test_bicubic_matrix_is_identity_for_equal_sizes(self):
        self.assertAllClose(_bicubic_antialias_matrix(7, 7), np.eye(7))

    def test_bicubic_matrix_rows_sum_to_one(self):
        for size_in, size_out in ((16, 10), (16, 5), (4, 9), (64, 40)):
            matrix = _bicubic_antialias_matrix(size_in, size_out)
            self.assertEqual(matrix.shape, (size_out, size_in))
            self.assertAllClose(matrix.sum(axis=1), np.ones(size_out))

    def test_bicubic_resize_matches_keras_antialias_resize(self):
        # Keras resizes with the same kernel as PyTorch `antialias=True`.
        x = np.random.normal(size=(1, 16, 16, 3)).astype("float32")
        for size in ((10, 10), (5, 7), (16, 16)):
            expected = ops.image.resize(
                x,
                size,
                interpolation="bicubic",
                antialias=True,
                data_format="channels_last",
            )
            actual = _resize_bicubic(ops.convert_to_tensor(x), size)
            self.assertAllClose(actual, expected, atol=1e-4, rtol=1e-4)


class UnlimitedOCRSamPositionEmbeddingTest(TestCase):
    def test_layer_basics(self):
        self.run_layer_test(
            cls=UnlimitedOCRSamPositionEmbedding,
            init_kwargs={"grid_size": 4, "hidden_dim": 3},
            input_data=np.random.uniform(size=(2, 4, 4, 3)).astype("float32"),
            expected_output_shape=(2, 4, 4, 3),
            # The position table.
            expected_num_trainable_weights=1,
        )

    def test_sam_position_embedding(self):
        layer = UnlimitedOCRSamPositionEmbedding(grid_size=4, hidden_dim=3)
        x = np.random.uniform(size=(2, 4, 4, 3)).astype("float32")
        layer.build(x.shape)
        table = np.random.uniform(size=(1, 4, 4, 3)).astype("float32")
        layer.pos_embed.assign(table)
        # The same grid adds the table without a resize.
        self.assertAllClose(layer(x), x + table, atol=1e-6)
        # Another grid resizes the table to the new grid.
        small = np.zeros((2, 3, 3, 3), "float32")
        output = ops.convert_to_numpy(layer(small))
        self.assertEqual(output.shape, (2, 3, 3, 3))
        # A resized table stays inside the range of the original values.
        self.assertLessEqual(output.max(), table.max() * 1.5)


class UnlimitedOCRCLIPEmbeddingTest(TestCase):
    def test_layer_basics(self):
        self.run_layer_test(
            cls=UnlimitedOCRCLIPEmbedding,
            init_kwargs={"hidden_dim": 4, "grid_size": 3},
            input_data=np.random.uniform(size=(2, 3, 3, 4)).astype("float32"),
            expected_output_shape=(2, 10, 4),
            # The class embedding and the position table.
            expected_num_trainable_weights=2,
        )

    def test_clip_embedding(self):
        layer = UnlimitedOCRCLIPEmbedding(hidden_dim=4, grid_size=3)
        x = np.random.uniform(size=(2, 3, 3, 4)).astype("float32")
        layer.build(x.shape)
        class_embedding = np.random.uniform(size=(4,)).astype("float32")
        positions = np.random.uniform(size=(10, 4)).astype("float32")
        layer.class_embedding.assign(class_embedding)
        layer.position_embedding.assign(positions)
        output = ops.convert_to_numpy(layer(x))
        tokens = np.concatenate(
            [np.broadcast_to(class_embedding, (2, 1, 4)), x.reshape(2, 9, 4)],
            axis=1,
        )
        self.assertAllClose(output, tokens + positions[None], atol=1e-6)
        # A smaller grid resizes only the patch positions.
        small = ops.convert_to_numpy(layer(x[:, :2, :2]))
        self.assertEqual(small.shape, (2, 5, 4))
        self.assertAllClose(
            small[:, 0],
            np.broadcast_to(class_embedding + positions[0], (2, 4)),
            atol=1e-6,
        )


class UnlimitedOCRMergerTest(TestCase):
    def setUp(self):
        self.hidden_dim = 2
        self.image_token_id = 7
        # Sample 0 has an image with 2 tile columns and 1 row, and an image
        # with no tile. Sample 1 has an image with 1 tile column and 2 rows,
        # and one padding image. A tile has 2x2 tokens. A global view has 1
        # token. Sample 0 has 13 + 3 image tokens. Sample 1 has 15.
        rng = np.random.RandomState(1)
        self.global_features = rng.normal(size=(4, 1, 2)).astype("float32")
        self.local_features = rng.normal(size=(4, 4, 2)).astype("float32")
        self.token_embeddings = rng.normal(size=(2, 18, 2)).astype("float32")
        self.image_newline = rng.normal(size=(2,)).astype("float32")
        self.view_separator = rng.normal(size=(2,)).astype("float32")
        token_ids = np.full((2, 18), 3, dtype="int32")
        token_ids[0, 1:17] = self.image_token_id
        token_ids[1, 2:17] = self.image_token_id
        self.input_data = {
            "token_ids": token_ids,
            "token_embeddings": self.token_embeddings,
            "global_features": self.global_features,
            "local_features": self.local_features,
            "patches_grid": np.array(
                [[[2, 1], [0, 0]], [[1, 2], [0, 0]]], dtype="int32"
            ),
        }

    def test_layer_basics(self):
        self.run_layer_test(
            cls=UnlimitedOCRImageTextEmbeddingMerger,
            init_kwargs={
                "hidden_dim": self.hidden_dim,
                "image_token_id": self.image_token_id,
            },
            input_data=self.input_data,
            expected_output_shape=(2, 18, 2),
            # The newline and the view separator.
            expected_num_trainable_weights=2,
            # The feature banks have a batch size other than the text.
            run_training_check=False,
        )

    def test_merger_output(self):
        layer = UnlimitedOCRImageTextEmbeddingMerger(
            hidden_dim=self.hidden_dim, image_token_id=self.image_token_id
        )
        layer(**self.input_data)
        layer.image_newline.assign(self.image_newline)
        layer.view_separator.assign(self.view_separator)
        # Values from `get_image_features` of `UnlimitedOcrModel` in
        # `transformers` 5.22.2. It ran on the same features. The features of
        # each image replace the placeholders of the sample in order.
        expected = [
            [
                [-0.1918, -0.8876],
                [0.3190, -0.2494],
                [1.4621, -2.0601],
                [-0.1724, -0.8779],
                [0.0422, 0.5828],
                [0.8462, -0.8595],
                [-0.3224, -0.3841],
                [1.1338, -1.0999],
                [-1.1006, 1.1447],
                [0.9016, 0.5025],
                [0.8462, -0.8595],
                [1.6243, -0.6118],
                [0.8462, -0.8595],
                [0.3505, -1.3123],
                [-0.5282, -1.0730],
                [0.8462, -0.8595],
                [0.3505, -1.3123],
                [0.3156, -2.0222],
            ],
            [
                [-0.3062, 0.8280],
                [0.2301, 0.7620],
                [0.9009, -0.6837],
                [-0.1229, -0.9358],
                [0.8462, -0.8595],
                [-0.2679, 0.5304],
                [-0.6917, -0.3968],
                [0.8462, -0.8595],
                [-0.6872, -0.8452],
                [-0.6712, -0.0127],
                [0.8462, -0.8595],
                [-1.1173, 0.2344],
                [1.6598, 0.7420],
                [0.8462, -0.8595],
                [0.8654, -2.3015],
                [0.8462, -0.8595],
                [0.3505, -1.3123],
                [-1.3731, 0.3152],
            ],
        ]
        output = ops.convert_to_numpy(layer(**self.input_data))
        self.assertAllClose(output, expected, atol=1e-4)

    def test_merger_without_images(self):
        d = self.hidden_dim
        token_ids = np.full((2, 5), 3, dtype="int32")
        embeddings = np.random.normal(size=(2, 5, d)).astype("float32")
        layer = UnlimitedOCRImageTextEmbeddingMerger(
            hidden_dim=d, image_token_id=self.image_token_id
        )
        output = layer(
            token_ids,
            embeddings,
            np.zeros((0, 4, d), "float32"),
            np.zeros((0, 4, d), "float32"),
            np.zeros((2, 0, 2), "int32"),
        )
        self.assertAllClose(output, embeddings)


class UnlimitedOCRImageFeatureExtractorTest(TestCase):
    def setUp(self):
        vision_encoder = UnlimitedOCRVisionEncoder(
            sam_hidden_dim=16,
            sam_num_layers=2,
            sam_num_heads=2,
            sam_mlp_dim=32,
            sam_output_channels=8,
            sam_patch_size=8,
            sam_image_size=128,
            sam_window_size=4,
            sam_global_attn_indexes=(1,),
            sam_downsample_channels=(12, 16),
            clip_hidden_dim=16,
            clip_num_layers=2,
            clip_num_heads=2,
            clip_intermediate_dim=32,
            clip_image_size=48,
            clip_patch_size=16,
        )
        projector = keras.layers.Dense(12, name="multimodal_projector")
        self.init_kwargs = {
            "vision_encoder": vision_encoder,
            "multimodal_projector": projector,
        }
        self.extractor = UnlimitedOCRImageFeatureExtractor(**self.init_kwargs)

    def test_layer_basics(self):
        self.run_layer_test(
            cls=UnlimitedOCRImageFeatureExtractor,
            init_kwargs=self.init_kwargs,
            input_data={
                "pixel_values": np.zeros((2, 1, 64, 64, 3), "float32"),
                "pixel_values_local": np.zeros((1, 3, 40, 40, 3), "float32"),
            },
            # A 64 pixel view gives 2x2 tokens. A 40 pixel tile gives 2x2.
            expected_output_shape=((2, 4, 12), (3, 4, 12)),
            # The vision encoder (75) and the projector (2).
            expected_num_trainable_weights=77,
            # The inputs and the outputs have different batch sizes.
            run_training_check=False,
        )

    def test_empty_batches_skip_the_encoder(self):
        global_features, local_features = self.extractor(
            np.zeros((2, 0, 128, 128, 3), "float32"),
            np.zeros((2, 0, 80, 80, 3), "float32"),
        )
        self.assertEqual(tuple(global_features.shape), (0, 16, 12))
        self.assertEqual(tuple(local_features.shape), (0, 9, 12))
