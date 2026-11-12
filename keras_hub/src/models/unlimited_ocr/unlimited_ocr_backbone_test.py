import keras
import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.unlimited_ocr.unlimited_ocr_backbone import (
    UnlimitedOCRBackbone,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_vision_encoder import (
    UnlimitedOCRVisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase

IMAGE_TOKEN_ID = 19


class UnlimitedOCRBackboneTest(TestCase):
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
        # A 64 pixel view gives a 2x2 grid of global features. A 40 pixel
        # tile also gives a 2x2 grid of local features.
        self.init_kwargs = {
            "vocabulary_size": 20,
            "num_layers": 2,
            "num_query_heads": 2,
            "hidden_dim": 16,
            "intermediate_dim": 32,
            "moe_intermediate_dim": 8,
            "num_shared_experts": 2,
            "num_experts": 4,
            "top_k": 2,
            "vision_encoder": vision_encoder,
            "image_token_id": IMAGE_TOKEN_ID,
            "global_image_size": 64,
            "local_image_size": 40,
            "sliding_window": 4,
        }
        # Image 0 has two tiles in a row: 2 * (2 * 2 + 1) = 10 local tokens
        # and 7 global tokens. Image 1 has no tile: 7 tokens.
        token_ids = np.full((2, 20), 3, dtype="int32")
        token_ids[0, 1:18] = IMAGE_TOKEN_ID
        token_ids[1, 2:9] = IMAGE_TOKEN_ID
        self.input_data = {
            "token_ids": ops.convert_to_tensor(token_ids),
            "padding_mask": ops.ones((2, 20), dtype="int32"),
            "pixel_values": ops.convert_to_tensor(
                np.random.uniform(size=(2, 1, 64, 64, 3)).astype("float32")
            ),
            "pixel_values_local": ops.convert_to_tensor(
                np.random.uniform(size=(2, 2, 40, 40, 3)).astype("float32")
            ),
            "patches_grid": ops.convert_to_tensor(
                [[[2, 1]], [[0, 0]]], dtype="int32"
            ),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=UnlimitedOCRBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 20, 16),
            variable_length_data=[self.input_data],
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=UnlimitedOCRBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_dtype_reaches_the_vision_encoder(self):
        # The vision encoder has the default dtype. The backbone dtype must
        # apply to the encoder and to every sublayer, as `from_preset` does.
        model = UnlimitedOCRBackbone(**self.init_kwargs, dtype="bfloat16")
        encoder = model.vision_encoder
        self.assertEqual(encoder.dtype_policy.name, "bfloat16")
        for sublayer in encoder._flatten_layers():
            # `Sequential` makes an input layer, which keeps float32.
            if isinstance(sublayer, keras.layers.InputLayer):
                continue
            self.assertEqual(sublayer.compute_dtype, "bfloat16", sublayer.name)
            self.assertEqual(sublayer.variable_dtype, "bfloat16", sublayer.name)
        output = model(self.input_data)
        self.assertEqual(ops.dtype(output), "bfloat16")

    def test_built_vision_encoder_with_another_dtype_raises(self):
        init_kwargs = dict(self.init_kwargs)
        init_kwargs["vision_encoder"](np.zeros((1, 64, 64, 3), dtype="float32"))
        with self.assertRaisesRegex(ValueError, "dtype"):
            UnlimitedOCRBackbone(**init_kwargs, dtype="bfloat16")

    def test_quantize(self):
        # `run_quantization_test` never calls `Model.quantize`. This test
        # calls it on a built backbone.
        model = UnlimitedOCRBackbone(**self.init_kwargs)
        model(self.input_data)
        model.quantize("int8")
        dense = model.transformer_layers[0].mlp._feedforward_gate_dense
        self.assertEqual(dense.quantization_mode, "int8")
        output = model(self.input_data)
        self.assertEqual(tuple(output.shape), (2, 20, 16))
        self.assertTrue(np.isfinite(ops.convert_to_numpy(output)).all())
        # The config and the weights restore the quantized backbone.
        revived = UnlimitedOCRBackbone.from_config(model.get_config())
        revived.set_weights(model.get_weights())
        self.assertAllClose(
            revived(self.input_data), model(self.input_data), atol=1e-5
        )

    def test_first_layer_is_dense_and_others_are_sparse(self):
        model = UnlimitedOCRBackbone(**self.init_kwargs)
        self.assertFalse(model.transformer_layers[0].sparse)
        self.assertTrue(model.transformer_layers[1].sparse)

    def test_image_features_reach_the_placeholders(self):
        model = UnlimitedOCRBackbone(**self.init_kwargs)
        original = ops.convert_to_numpy(model(self.input_data))
        changed_data = dict(self.input_data)
        changed_data["pixel_values"] = ops.convert_to_tensor(
            np.random.uniform(size=(2, 1, 64, 64, 3)).astype("float32")
        )
        changed = ops.convert_to_numpy(model(changed_data))
        # The row of image 0 starts with a token before the image. The
        # attention is causal, so that token stays the same.
        self.assertAllClose(original[0, 0], changed[0, 0], atol=1e-5)
        difference = np.abs(original[0, 17] - changed[0, 17]).max()
        self.assertGreater(float(difference), 0)
        # The text positions of image 1 before the image stay the same.
        self.assertAllClose(original[1, :2], changed[1, :2], atol=1e-5)

    def test_text_only_input(self):
        model = UnlimitedOCRBackbone(**self.init_kwargs)
        token_ids = np.full((2, 5), 3, dtype="int32")
        output = model(
            {
                "token_ids": token_ids,
                "padding_mask": np.ones((2, 5), dtype="int32"),
                "pixel_values": np.zeros((2, 0, 64, 64, 3), dtype="float32"),
                "pixel_values_local": np.zeros(
                    (2, 0, 40, 40, 3), dtype="float32"
                ),
                "patches_grid": np.zeros((2, 0, 2), dtype="int32"),
            }
        )
        self.assertEqual(tuple(output.shape), (2, 5, 16))

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=UnlimitedOCRBackbone,
            preset="unlimited_ocr",
            input_data={
                "token_ids": ops.ones((1, 4), dtype="int32"),
                "padding_mask": ops.ones((1, 4), dtype="int32"),
                "pixel_values": ops.zeros((1, 0, 1024, 1024, 3)),
                "pixel_values_local": ops.zeros((1, 0, 640, 640, 3)),
                "patches_grid": ops.zeros((1, 0, 2), dtype="int32"),
            },
            expected_output_shape=(1, 4, 1280),
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in UnlimitedOCRBackbone.presets:
            self.run_preset_test(
                cls=UnlimitedOCRBackbone,
                preset=preset,
                input_data={
                    "token_ids": ops.ones((1, 4), dtype="int32"),
                    "padding_mask": ops.ones((1, 4), dtype="int32"),
                    "pixel_values": ops.zeros((1, 0, 1024, 1024, 3)),
                    "pixel_values_local": ops.zeros((1, 0, 640, 640, 3)),
                    "patches_grid": ops.zeros((1, 0, 2), dtype="int32"),
                },
            )
