import keras
import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.qwen3_vl.qwen3_vl_backbone import Qwen3VLBackbone
from keras_hub.src.models.qwen3_vl.qwen3_vl_vision_encoder import (
    Qwen3VLVisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.keras_utils import running_on_gpu


def _atol():
    # GPUs run float32 matmuls in TF32, and batched, single-sample and
    # still-frame passes take different kernels. A100 runs reach 2.5e-3.
    return 5e-3 if running_on_gpu() else 1e-5


TEXT_KWARGS = {
    "vocabulary_size": 10,
    "num_layers": 4,
    "num_query_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "hidden_dim": 16,
    "intermediate_dim": 32,
    "mrope_section": [2, 1, 1],
}


class Qwen3VLBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = dict(TEXT_KWARGS)
        self.input_data = {
            "token_ids": np.arange(10, dtype="int32").reshape(2, 5),
            "padding_mask": np.ones((2, 5), dtype="int32"),
            "position_ids": np.tile(
                np.arange(5, dtype="int32")[:, None], (2, 1, 3)
            ),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=Qwen3VLBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 16),
        )

    def test_position_ids_default_to_1d(self):
        model = Qwen3VLBackbone(**self.init_kwargs)
        explicit = model(self.input_data)
        default_inputs = {
            k: v for k, v in self.input_data.items() if k != "position_ids"
        }
        self.assertAllClose(model(default_inputs), explicit)

    def test_position_ids_change_output(self):
        model = Qwen3VLBackbone(**self.init_kwargs)
        # Default weights are small enough to make attention near uniform.
        rng = np.random.RandomState(0)
        model.set_weights([rng.randn(*w.shape) for w in model.get_weights()])
        shifted = dict(self.input_data)
        shifted["position_ids"] = self.input_data["position_ids"].copy()
        shifted["position_ids"][:, :, 1] *= 2
        base = ops.convert_to_numpy(model(self.input_data))
        out = ops.convert_to_numpy(model(shifted))
        self.assertGreater(np.abs(base - out).max(), 1e-6)

    def test_num_parameters(self):
        model = Qwen3VLBackbone(**self.init_kwargs)
        self.assertEqual(model.count_params(), 12816)

    def test_tie_word_embeddings(self):
        untied = Qwen3VLBackbone(**self.init_kwargs)
        tied = Qwen3VLBackbone(**self.init_kwargs, tie_word_embeddings=True)
        embedding = tied.token_embedding
        # No separate output projection.
        self.assertEqual(
            untied.count_params() - tied.count_params(),
            embedding.input_dim * embedding.output_dim,
        )
        hidden = tied(self.input_data)
        logits = embedding(hidden, reverse=True)
        expected = ops.matmul(hidden, ops.transpose(embedding.embeddings))
        self.assertAllClose(logits, expected, atol=1e-5)
        restored = Qwen3VLBackbone.from_config(tied.get_config())
        self.assertTrue(restored.tie_word_embeddings)
        self.assertEqual(restored.count_params(), tied.count_params())

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen3VLBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )


class Qwen3VLMultimodalBackboneTest(TestCase):
    def setUp(self):
        self.vision_encoder = Qwen3VLVisionEncoder(
            depth=3,
            hidden_size=16,
            num_heads=2,
            intermediate_size=32,
            out_hidden_size=16,
            patch_size=4,
            temporal_patch_size=2,
            spatial_merge_size=2,
            num_position_embeddings=64,
            deepstack_visual_indexes=(0, 1),
        )
        self.init_kwargs = dict(TEXT_KWARGS, vision_encoder=self.vision_encoder)
        self.input_data = self._inputs()

    def _inputs(self, seq_len=10):
        # One 4x4 patch grid merges into 4 vision tokens at positions 2..5.
        return {
            "token_ids": np.arange(seq_len, dtype="int32")[None, :],
            "padding_mask": np.ones((1, seq_len), dtype="int32"),
            "position_ids": np.tile(
                np.arange(seq_len, dtype="int32")[:, None], (1, 1, 3)
            ),
            "pixel_values": np.random.RandomState(0)
            .randn(1, 16, 2, 4, 4, 3)
            .astype("float32"),
            "image_grid_thw": np.array([[[1, 4, 4]]], dtype="int32"),
            "vision_indices": np.array([[2, 3, 4, 5]], dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=Qwen3VLBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(1, 10, 16),
            variable_length_data=[self._inputs(n) for n in (6, 8)],
            # The vision encoder is built outside the policy under test.
            # `test_mixed_bfloat16_forward` covers mixed precision.
            run_mixed_precision_check=False,
            run_quantization_check=False,
        )

    def test_functional_symbolic_text_inputs(self):
        backbone = Qwen3VLBackbone(**self.init_kwargs)
        data = {
            "token_ids": np.arange(10, dtype="int32").reshape(2, 5),
            "padding_mask": np.ones((2, 5), dtype="int32"),
        }
        # Unknown and known sequence lengths, with an unknown batch size.
        for seq_len in (None, 5):
            inputs = {
                "token_ids": keras.Input((seq_len,), dtype="int32"),
                "padding_mask": keras.Input((seq_len,), dtype="int32"),
            }
            model = keras.Model(inputs, backbone(inputs))
            self.assertEqual(model.output.shape, (None, seq_len, 16))
            self.assertAllClose(model(data), backbone(data))

    def test_mixed_bfloat16_forward(self):
        previous = keras.mixed_precision.global_policy()
        keras.mixed_precision.set_global_policy("mixed_bfloat16")
        try:
            # The vision tower runs in float32 by default, whatever the global
            # policy, or in the dtype it is given.
            for vision_dtype, vision_compute_dtype in [
                (None, "float32"),
                ("mixed_bfloat16", "bfloat16"),
            ]:
                config = self.vision_encoder.get_config()
                config["dtype"] = vision_dtype
                vision_encoder = Qwen3VLVisionEncoder.from_config(config)
                for layer in (
                    vision_encoder,
                    vision_encoder.get_layer("blocks_0"),
                ):
                    self.assertEqual(layer.compute_dtype, vision_compute_dtype)
                backbone = Qwen3VLBackbone(
                    **dict(self.init_kwargs, vision_encoder=vision_encoder)
                )
                self.assertEqual(backbone.compute_dtype, "bfloat16")
                output = backbone(self.input_data)
                self.assertDTypeEqual(output, "bfloat16")
                output = ops.convert_to_numpy(ops.cast(output, "float32"))
                self.assertTrue(np.all(np.isfinite(output)))
        finally:
            keras.mixed_precision.set_global_policy(previous)

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen3VLBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_multimodal_forward_shape(self):
        model = Qwen3VLBackbone(**self.init_kwargs)
        self.assertEqual(
            ops.shape(model(self.input_data)),
            (1, 10, 16),
        )

    def test_text_only_call_on_multimodal_backbone(self):
        model = Qwen3VLBackbone(**self.init_kwargs)
        text_inputs = {
            k: self.input_data[k] for k in ("token_ids", "padding_mask")
        }
        self.assertEqual(ops.shape(model(text_inputs)), (1, 10, 16))

    def test_partial_vision_inputs_raise(self):
        model = Qwen3VLBackbone(**self.init_kwargs)
        inputs = dict(self.input_data)
        del inputs["vision_indices"]
        with self.assertRaisesRegex(ValueError, "vision_indices"):
            model(inputs)

    def test_text_only_predict_on_multimodal_backbone(self):
        model = Qwen3VLBackbone(**self.init_kwargs)
        text_inputs = {
            k: self.input_data[k] for k in ("token_ids", "padding_mask")
        }
        self.assertAllClose(
            model.predict(text_inputs, verbose=0),
            model(text_inputs),
            atol=_atol(),
        )

    def test_batch_matches_single_samples(self):
        # Vision indices are positions within each sample, so a batch gives
        # the same result as each of its samples on its own.
        model = Qwen3VLBackbone(**self.init_kwargs)
        other = dict(self.input_data)
        other["pixel_values"] = self.input_data["pixel_values"] + 1.0
        other["vision_indices"] = np.array([[3, 4, 5, 6]], dtype="int32")
        batch = {
            k: np.concatenate([self.input_data[k], other[k]])
            for k in self.input_data
        }
        output = ops.convert_to_numpy(model(batch))
        self.assertAllClose(output[:1], model(self.input_data), atol=_atol())
        self.assertAllClose(output[1:], model(other), atol=_atol())

    def test_vision_inputs_change_output(self):
        model = Qwen3VLBackbone(**self.init_kwargs)
        other = dict(self.input_data)
        other["pixel_values"] = self.input_data["pixel_values"] + 1.0
        base = ops.convert_to_numpy(model(self.input_data))
        out = ops.convert_to_numpy(model(other))
        self.assertGreater(np.abs(base - out).max(), 1e-6)

    def test_single_frame_pixels_match_repeated_frames(self):
        model = Qwen3VLBackbone(**self.init_kwargs)
        still = dict(self.input_data)
        still["pixel_values"] = self.input_data["pixel_values"][:, :, :1]
        repeated = dict(self.input_data)
        repeated["pixel_values"] = np.repeat(still["pixel_values"], 2, axis=2)
        # A still frame sums the patch kernel over time before the matmul.
        self.assertAllClose(model(still), model(repeated), atol=_atol())

    def test_deepstack_depends_on_vision_indices(self):
        model = Qwen3VLBackbone(**self.init_kwargs)
        moved = dict(self.input_data)
        moved["vision_indices"] = np.array([[3, 4, 5, 6]], dtype="int32")
        base = ops.convert_to_numpy(model(self.input_data))
        out = ops.convert_to_numpy(model(moved))
        self.assertGreater(np.abs(base - out).max(), 1e-6)

    def test_num_parameters(self):
        model = Qwen3VLBackbone(**self.init_kwargs)
        self.assertEqual(model.vision_encoder.count_params(), 25136)
        self.assertEqual(model.count_params(), 37952)

    def test_mismatched_vision_width(self):
        with self.assertRaises(ValueError):
            Qwen3VLBackbone(**dict(self.init_kwargs, hidden_dim=32))

    def test_config_roundtrip(self):
        model = Qwen3VLBackbone(**self.init_kwargs)
        clone = Qwen3VLBackbone.from_config(model.get_config())
        self.assertEqual(
            clone.vision_encoder.deepstack_visual_indexes,
            self.vision_encoder.deepstack_visual_indexes,
        )
        self.assertEqual(clone.mrope_section, [2, 1, 1])
