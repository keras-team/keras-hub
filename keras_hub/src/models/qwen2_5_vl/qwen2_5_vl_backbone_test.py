import keras
import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_backbone import (
    Qwen2_5VLBackbone,
)
from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_vision_encoder import (
    Qwen2_5VLVisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen2_5VLBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 1000,
            "num_layers": 2,
            "num_query_heads": 8,
            "num_key_value_heads": 4,
            "hidden_dim": 128,
            "intermediate_dim": 256,
        }
        self.input_data = {
            "token_ids": np.array([[1, 2, 3, 4, 5]]),
            "padding_mask": np.array([[1, 1, 1, 1, 1]]),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=Qwen2_5VLBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(1, 5, 128),
        )

    def test_num_parameters(self):
        model = Qwen2_5VLBackbone(**self.init_kwargs)
        # Pinned so architecture changes have to be intentional.
        self.assertEqual(model.count_params(), 552064)

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen2_5VLBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )


class Qwen2_5VLMultimodalBackboneTest(TestCase):
    """Tests for the backbone with a vision encoder attached."""

    def setUp(self):
        self.vision_encoder = Qwen2_5VLVisionEncoder(
            patch_size=2,
            temporal_patch_size=2,
            in_channels=3,
            embed_dim=64,
            out_dim=128,
            depth=2,
            num_heads=4,
            intermediate_dim=256,
            spatial_merge_size=2,
            window_size=8,
            fullatt_block_indexes=[1],
        )
        self.init_kwargs = {
            "vocabulary_size": 1000,
            "num_layers": 2,
            "num_query_heads": 8,
            "num_key_value_heads": 4,
            "hidden_dim": 128,
            "intermediate_dim": 256,
            # head_dim = 128/8 = 16 → sections must sum to 8.
            "mrope_section": [2, 3, 3],
            "vision_encoder": self.vision_encoder,
        }
        self.input_data = {
            "token_ids": np.array([[1, 2, 3, 4, 5]]),
            "padding_mask": np.array([[1, 1, 1, 1, 1]]),
        }

    def _vision_inputs(self, seq_len=8):
        """6 merged vision tokens from grid [1, 4, 6] at positions 1-6."""
        grid_thw = np.array([[[1, 4, 6]]], dtype="int32")
        pixel_values = np.random.rand(1, 24, 2, 2, 2, 3).astype("float32")
        vision_indices = np.array([list(range(1, 7))], dtype="int32")
        # Sequential positions (equivalent to 1-D RoPE).
        position_ids = np.broadcast_to(
            np.arange(seq_len)[None, None, :], (1, 3, seq_len)
        ).astype("int32")
        token_ids = np.full((1, seq_len), 5, dtype="int32")
        token_ids[0, 0] = 1
        token_ids[0, 1:7] = 999
        return {
            "token_ids": token_ids,
            "padding_mask": np.ones((1, seq_len), dtype="int32"),
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw,
            "vision_indices": vision_indices,
            "position_ids": position_ids,
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=Qwen2_5VLBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self._vision_inputs(),
            expected_output_shape=(1, 8, 128),
            variable_length_data=[
                self._vision_inputs(seq_len) for seq_len in (9, 10)
            ],
            run_mixed_precision_check=False,
            run_quantization_check=False,
        )

    def test_mixed_bfloat16_forward(self):
        """Vision interleave must work under a mixed-bfloat16 policy."""
        previous = keras.mixed_precision.global_policy()
        keras.mixed_precision.set_global_policy("mixed_bfloat16")
        try:
            backbone = Qwen2_5VLBackbone(**self.init_kwargs)
            output = backbone(self._vision_inputs())
            self.assertDTypeEqual(output, "bfloat16")
        finally:
            keras.mixed_precision.set_global_policy(previous)

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen2_5VLBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self._vision_inputs(),
        )

    def test_multimodal_backbone_builds(self):
        model = Qwen2_5VLBackbone(**self.init_kwargs)
        self.assertGreater(model.count_params(), 0)
        self.assertIsNotNone(model.vision_encoder)
        self.assertTrue(hasattr(model, "interleave_embeddings"))

    def test_multimodal_backbone_forward(self):
        """Vision embeddings must reach the sequence at the vision
        positions."""
        backbone = Qwen2_5VLBackbone(**self.init_kwargs)
        outputs = []
        for value in (0.0, 1.0):
            inputs = self._vision_inputs()
            inputs["pixel_values"] = np.full_like(inputs["pixel_values"], value)
            outputs.append(ops.convert_to_numpy(backbone(inputs)))
        # without interleaving, both forwards would be bit-identical at
        # the vision positions.
        vision_positions = np.arange(1, 7)
        vision_diff = np.abs(
            outputs[0][0, vision_positions] - outputs[1][0, vision_positions]
        ).max()
        self.assertGreater(vision_diff, 1e-3)

    def test_backbone_text_only_injection(self):
        """Multimodal backbone accepts text-only inputs via __call__."""
        backbone = Qwen2_5VLBackbone(**self.init_kwargs)
        output = backbone(self.input_data)
        self.assertEqual(tuple(output.shape), (1, 5, 128))

    def test_backbone_mrope_positions(self):
        """M-RoPE position ids alter outputs vs. plain sequential RoPE."""
        backbone = Qwen2_5VLBackbone(**self.init_kwargs)
        inputs = self._vision_inputs()
        out_flat = backbone(inputs)

        # Vision tokens follow the merged 2x3 grid at t=0: height/width
        # channels index the (r, c) coords, text continues at 3.
        pos_mrope = np.array(
            [
                [
                    [0, 0, 0, 0, 0, 0, 0, 3],
                    [0, 0, 0, 0, 1, 1, 1, 3],
                    [0, 0, 1, 2, 0, 1, 2, 3],
                ]
            ],
            dtype="int32",
        )
        out_mrope = backbone({**inputs, "position_ids": pos_mrope})
        self.assertNotAllClose(out_flat, out_mrope)

    def test_backbone_mrope_sequential_equals_1d(self):
        """Sequential M-RoPE positions reproduce the plain 1-D RoPE path.

        A multimodal backbone fed text-only inputs gets sequential
        position ids injected, so its M-RoPE recomposition must match a
        text-only backbone (``mrope_section=None``) with shared weights —
        the branch that actually executes ``rotary_embedding_layer``.
        """
        text_inputs = {
            "token_ids": np.array([[1, 2, 3, 4, 5]]),
            "padding_mask": np.ones((1, 5), dtype="int32"),
        }
        seq_len = 5
        seq_ids = np.broadcast_to(
            np.arange(seq_len)[None, None, :], (1, 3, seq_len)
        ).astype("int32")
        multimodal = Qwen2_5VLBackbone(**self.init_kwargs)
        out_mrope = multimodal({**text_inputs, "position_ids": seq_ids})

        text_only_kwargs = {
            k: v
            for k, v in self.init_kwargs.items()
            if k not in ("vision_encoder", "mrope_section")
        }
        text_only = Qwen2_5VLBackbone(**text_only_kwargs, mrope_section=None)
        text_only(text_inputs)  # build before weight sharing
        text_only.token_embedding.set_weights(
            multimodal.token_embedding.get_weights()
        )
        for text_layer, mm_layer in zip(
            text_only.transformer_layers, multimodal.transformer_layers
        ):
            text_layer.set_weights(mm_layer.get_weights())
        text_only.layer_norm.set_weights(multimodal.layer_norm.get_weights())

        out_1d = text_only(text_inputs)
        self.assertAllClose(out_mrope, out_1d, atol=1e-5)
