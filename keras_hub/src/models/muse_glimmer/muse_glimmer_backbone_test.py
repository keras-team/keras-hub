import numpy as np
import pytest
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.muse_glimmer.muse_glimmer_backbone import (
    MuseGlimmerBackbone,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_layers import (
    MuseGlimmerCenteredRMSNorm,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_layers import (
    MuseGlimmerRMSNorm,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_vision_encoder import (
    _QUERY_BLOCK_SIZE,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_vision_encoder import (
    MuseGlimmerVisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.keras_utils import running_on_gpu


class MuseGlimmerBackboneTest(TestCase, parameterized.TestCase):
    def setUp(self):
        # === Text-only Backbone ===
        self.text_init_kwargs = {
            "vocabulary_size": 20,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "hidden_dim": 32,
            "intermediate_dim": 32,
            "head_dim": 8,
            "sliding_window_size": 4,
            "layer_types": ["sliding_attention", "full_attention"],
        }
        self.text_backbone_input_data = {
            "token_ids": np.ones((2, 5), dtype="int32"),
            "padding_mask": np.ones((2, 5), dtype="int32"),
        }

        # === Vision + Text Backbone ===
        vision_encoder = MuseGlimmerVisionEncoder(
            num_layers=2,
            hidden_size=8,
            num_heads=2,
            intermediate_size=16,
            patch_size=2,
            patch_temporal=2,
            merge_size=2,
            pos_emb_height=4,
            pos_emb_width=4,
            layer_types=["window_attention", "full_attention"],
        )
        self.init_kwargs = dict(
            self.text_init_kwargs,
            vision_encoder=vision_encoder,
            projector_hidden_dim=16,
        )
        grid_thw = np.array([[[1, 4, 4]]], dtype="int32")  # (1, 1, 3)
        total_patches = 1 * 4 * 4
        patch_dim = 2 * 3 * 2 * 2  # patch_temporal * 3 * patch_size**2
        pixel_values = np.random.randn(1, total_patches, patch_dim).astype(
            "float32"
        )
        self.input_data = {
            "token_ids": np.ones((1, 10), dtype="int32"),
            "padding_mask": np.ones((1, 10), dtype="int32"),
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw,
            "vision_indices": np.array([[2, 3, 4, 5]], dtype="int32"),
        }

        # === Assistant/drafter opt-in flags ===
        self.assistant_base_kwargs = {
            "vocabulary_size": 20,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "hidden_dim": 16,
            "intermediate_dim": 32,
            "head_dim": 4,
            "sliding_window_size": 4,
            "layer_types": ["sliding_attention", "sliding_attention"],
        }

    @parameterized.named_parameters(
        ("text_and_vision", "text_and_vision"), ("text_only", "text_only")
    )
    def test_backbone_basics(self, backbone_type):
        if backbone_type == "text_and_vision":
            init_kwargs = self.init_kwargs
            input_data = self.input_data
            # `pixel_values`/`image_grid_thw` have a fixed correspondence,
            # so the default axis-1 truncation for variable-length checks
            # (built for plain token sequences) would corrupt them.
            variable_length_data = [input_data]
        else:
            init_kwargs = self.text_init_kwargs
            input_data = self.text_backbone_input_data
            variable_length_data = None

        batch_size, seq_len = input_data["token_ids"].shape
        self.run_backbone_test(
            cls=MuseGlimmerBackbone,
            init_kwargs=init_kwargs,
            input_data=input_data,
            expected_output_shape=(batch_size, seq_len, 32),
            variable_length_data=variable_length_data,
        )

    @parameterized.named_parameters(
        ("text_and_vision", "text_and_vision"), ("text_only", "text_only")
    )
    def test_saved_model(self, backbone_type):
        if backbone_type == "text_and_vision":
            init_kwargs = self.init_kwargs
            input_data = self.input_data
        else:
            init_kwargs = self.text_init_kwargs
            input_data = self.text_backbone_input_data

        self.run_model_saving_test(
            cls=MuseGlimmerBackbone,
            init_kwargs=init_kwargs,
            input_data=input_data,
        )

    def test_text_only_call_on_multimodal_backbone(self):
        model = MuseGlimmerBackbone(**self.init_kwargs)
        output = model(self.text_backbone_input_data)
        self.assertEqual(ops.shape(output), (2, 5, 32))

    def test_vision_batch_of_two(self):
        model = MuseGlimmerBackbone(**self.init_kwargs)
        pixel_values = np.random.randn(2, 16, 24).astype("float32")
        batch = {
            "token_ids": np.array(
                [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] * 2, dtype="int32"
            )
            + np.array([[0], [3]], dtype="int32"),
            "padding_mask": np.ones((2, 10), dtype="int32"),
            "pixel_values": pixel_values,
            "image_grid_thw": np.array([[[1, 4, 4]]] * 2, dtype="int32"),
            # Each row has its own placeholder positions.
            "vision_indices": np.array(
                [[2, 3, 4, 5], [3, 4, 5, 6]], dtype="int32"
            ),
        }
        output = model(batch)
        self.assertEqual(ops.shape(output), (2, 10, 32))
        # Each row must match the same prompt run alone. TF32 matmuls on GPU
        # differ by up to 6e-3 between the batched and single runs.
        tol = 2e-2 if running_on_gpu() else 1e-4
        for i in range(2):
            single = {key: value[i : i + 1] for key, value in batch.items()}
            self.assertAllClose(
                ops.convert_to_numpy(output)[i],
                ops.convert_to_numpy(model(single))[0],
                atol=tol,
                rtol=tol,
            )

    def test_fit_with_images_over_one_query_block(self):
        # The patches fill more than one query block of the traced masked
        # attention. Training must differentiate through the blocks.
        model = MuseGlimmerBackbone(**self.init_kwargs)
        grid = 2 * int(np.ceil(np.sqrt(_QUERY_BLOCK_SIZE + 1) / 2))
        num_patches = grid * grid
        self.assertGreater(num_patches, _QUERY_BLOCK_SIZE)
        num_tokens = num_patches // 4
        seq_len = num_tokens + 4
        x = {
            "token_ids": np.ones((1, seq_len), dtype="int32"),
            "padding_mask": np.ones((1, seq_len), dtype="int32"),
            "pixel_values": np.random.randn(1, num_patches, 24).astype(
                "float32"
            ),
            "image_grid_thw": np.array([[[1, grid, grid]]], dtype="int32"),
            "vision_indices": np.arange(2, 2 + num_tokens, dtype="int32")[None],
        }
        y = np.zeros((1, seq_len, 32), dtype="float32")
        model.compile(optimizer="sgd", loss="mse")
        history = model.fit(x, y, batch_size=1, epochs=1, verbose=0)
        self.assertTrue(np.isfinite(history.history["loss"][0]))

    def test_assistant_backbone_basics(self):
        # All five opt-in flags combined, mirroring
        # `MuseGlimmerAssistantCausalLM`'s real constructor arguments.
        init_kwargs = dict(
            self.assistant_base_kwargs,
            use_bidirectional_attention=True,
            context_projection_layer_ids=[0, 1],
            use_external_embeddings=True,
            enable_qk_scale_and_gate=False,
            use_sandwich_norm=False,
        )
        input_data = {
            "noise_embeds": np.random.randn(2, 5, 16).astype("float32"),
            "padding_mask": np.ones((2, 5), dtype="int32"),
            "context_hidden_states": np.random.randn(2, 5, 32).astype(
                "float32"
            ),
        }
        self.run_backbone_test(
            cls=MuseGlimmerBackbone,
            init_kwargs=init_kwargs,
            input_data=input_data,
            expected_output_shape=(2, 5, 16),
        )

    def test_use_bidirectional_attention_flag(self):
        init_kwargs = dict(
            self.assistant_base_kwargs, use_bidirectional_attention=True
        )
        model = MuseGlimmerBackbone(**init_kwargs)
        input_data = {
            "token_ids": np.ones((2, 5), dtype="int32"),
            "padding_mask": np.ones((2, 5), dtype="int32"),
        }
        output = model(input_data)
        self.assertEqual(ops.shape(output), (2, 5, 16))

    def test_use_external_embeddings_flag(self):
        init_kwargs = dict(
            self.assistant_base_kwargs, use_external_embeddings=True
        )
        model = MuseGlimmerBackbone(**init_kwargs)
        self.assertIsNone(model.token_embedding)
        input_data = {
            "noise_embeds": np.random.randn(2, 5, 16).astype("float32"),
            "padding_mask": np.ones((2, 5), dtype="int32"),
        }
        output = model(input_data)
        self.assertEqual(ops.shape(output), (2, 5, 16))

    def test_context_projection_layer_ids_flag(self):
        # Context injection is only ever used together with bidirectional
        # attention in the real assistant checkpoint (see
        # `muse_glimmer_decoder.py`'s docstring) — the causal mask branch
        # does not account for the extra context key/value length, since
        # that combination never occurs in practice.
        init_kwargs = dict(
            self.assistant_base_kwargs,
            context_projection_layer_ids=[0, 1],
            use_bidirectional_attention=True,
        )
        model = MuseGlimmerBackbone(**init_kwargs)
        input_data = {
            "token_ids": np.ones((2, 5), dtype="int32"),
            "padding_mask": np.ones((2, 5), dtype="int32"),
            "context_hidden_states": np.random.randn(2, 5, 32).astype(
                "float32"
            ),
        }
        output = model(input_data)
        self.assertEqual(ops.shape(output), (2, 5, 16))

    def test_context_projection_requires_bidirectional_attention(self):
        with self.assertRaisesRegex(ValueError, "use_bidirectional_attention"):
            MuseGlimmerBackbone(
                **self.assistant_base_kwargs,
                context_projection_layer_ids=[0, 1],
            )

    def test_vision_encoder_requires_projector_hidden_dim(self):
        init_kwargs = dict(self.init_kwargs, projector_hidden_dim=None)
        with self.assertRaisesRegex(ValueError, "projector_hidden_dim"):
            MuseGlimmerBackbone(**init_kwargs)

    def test_invalid_layer_types_raise(self):
        for layer_types in (
            ["full", "sliding_attention"],
            ["sliding_attention"],
            ["sliding_attention"] * 3,
        ):
            with self.assertRaisesRegex(ValueError, "layer_types"):
                MuseGlimmerBackbone(
                    **dict(self.text_init_kwargs, layer_types=layer_types)
                )

    def test_query_heads_must_be_multiple_of_key_value_heads(self):
        init_kwargs = dict(self.text_init_kwargs, num_key_value_heads=3)
        with self.assertRaisesRegex(ValueError, "num_key_value_heads"):
            MuseGlimmerBackbone(**init_kwargs)

    def test_assistant_full_attention_layer_uses_rope(self):
        assistant = MuseGlimmerBackbone(
            **dict(
                self.assistant_base_kwargs,
                layer_types=["sliding_attention", "full_attention"],
                use_bidirectional_attention=True,
                context_projection_layer_ids=[0, 1],
                use_external_embeddings=True,
            )
        )
        target = MuseGlimmerBackbone(
            **dict(
                self.text_init_kwargs,
                layer_types=["sliding_attention", "full_attention"],
            )
        )
        self.assertTrue(
            assistant.transformer_layers[1]._self_attention_layer.use_rope
        )
        self.assertFalse(
            target.transformer_layers[1]._self_attention_layer.use_rope
        )

    @parameterized.named_parameters(
        ("text_and_vision", "text_and_vision"), ("text_only", "text_only")
    )
    def test_quantize(self, backbone_type):
        # `run_quantization_test` never calls `Model.quantize`, which
        # raises on any unbuilt layer.
        if backbone_type == "text_and_vision":
            backbone = MuseGlimmerBackbone(**self.init_kwargs)
            backbone(self.input_data)
        else:
            backbone = MuseGlimmerBackbone(**self.text_init_kwargs)
        backbone.quantize("int8")

    def test_quantized_config_round_trip(self):
        backbone = MuseGlimmerBackbone(**self.init_kwargs)
        backbone(self.input_data)
        backbone.quantize("int8")
        revived = MuseGlimmerBackbone.from_config(backbone.get_config())
        revived.set_weights(backbone.get_weights())
        self.assertAllClose(
            revived(self.input_data), backbone(self.input_data), atol=1e-5
        )

    def test_dtype_reaches_vision_encoder_from_config(self):
        config = MuseGlimmerBackbone(**self.init_kwargs).get_config()
        config["dtype"] = "bfloat16"
        revived = MuseGlimmerBackbone.from_config(config)
        self.assertEqual(revived.dtype_policy.name, "bfloat16")
        self.assertEqual(revived.vision_encoder.dtype_policy.name, "bfloat16")

    def test_enable_qk_scale_and_gate_false_flag(self):
        init_kwargs = dict(
            self.assistant_base_kwargs, enable_qk_scale_and_gate=False
        )
        model = MuseGlimmerBackbone(**init_kwargs)
        input_data = {
            "token_ids": np.ones((2, 5), dtype="int32"),
            "padding_mask": np.ones((2, 5), dtype="int32"),
        }
        output = model(input_data)
        self.assertEqual(ops.shape(output), (2, 5, 16))
        for layer in model.transformer_layers:
            self.assertFalse(
                hasattr(layer._self_attention_layer, "_gate_dense")
            )

    def test_use_sandwich_norm_false_flag(self):
        init_kwargs = dict(self.assistant_base_kwargs, use_sandwich_norm=False)
        model = MuseGlimmerBackbone(**init_kwargs)
        input_data = {
            "token_ids": np.ones((2, 5), dtype="int32"),
            "padding_mask": np.ones((2, 5), dtype="int32"),
        }
        output = model(input_data)
        self.assertEqual(ops.shape(output), (2, 5, 16))
        for layer in model.transformer_layers:
            self.assertFalse(hasattr(layer, "_post_attention_layernorm"))
            self.assertFalse(hasattr(layer, "_post_feedforward_layernorm"))

    def test_qk_norm_with_scale_flag(self):
        input_data = {
            "token_ids": np.ones((2, 5), dtype="int32"),
            "padding_mask": np.ones((2, 5), dtype="int32"),
        }
        scaled = MuseGlimmerBackbone(
            **dict(self.assistant_base_kwargs, qk_norm_with_scale=True)
        )
        self.assertEqual(ops.shape(scaled(input_data)), (2, 5, 16))
        for layer in scaled.transformer_layers:
            attention = layer._self_attention_layer
            self.assertFalse(hasattr(attention, "_qk_norm"))
            self.assertTrue(attention._query_norm.with_scale)
            self.assertTrue(attention._key_norm.with_scale)
            self.assertTrue(hasattr(attention._query_norm, "scale"))
            self.assertTrue(hasattr(attention._key_norm, "scale"))
        unscaled = MuseGlimmerBackbone(**self.assistant_base_kwargs)
        for layer in unscaled.transformer_layers:
            attention = layer._self_attention_layer
            self.assertFalse(hasattr(attention, "_query_norm"))
            self.assertFalse(attention._qk_norm.with_scale)
            self.assertFalse(hasattr(attention._qk_norm, "scale"))

    def test_use_centered_norm_false_flag(self):
        init_kwargs = dict(self.assistant_base_kwargs, use_centered_norm=False)
        model = MuseGlimmerBackbone(**init_kwargs)
        input_data = {
            "token_ids": np.ones((2, 5), dtype="int32"),
            "padding_mask": np.ones((2, 5), dtype="int32"),
        }
        self.assertEqual(ops.shape(model(input_data)), (2, 5, 16))
        for layer in model.transformer_layers:
            for norm in (
                layer._input_layernorm,
                layer._pre_feedforward_layernorm,
            ):
                self.assertIsInstance(norm, MuseGlimmerRMSNorm)
                self.assertNotIsInstance(norm, MuseGlimmerCenteredRMSNorm)

    def test_assistant_checkpoint_flags(self):
        # The flags match `convert_muse_glimmer_assistant.py`.
        init_kwargs = dict(
            self.assistant_base_kwargs,
            use_bidirectional_attention=True,
            context_projection_layer_ids=[0, 1],
            use_external_embeddings=True,
            enable_qk_scale_and_gate=False,
            use_sandwich_norm=False,
            qk_norm_with_scale=True,
            use_centered_norm=False,
        )
        input_data = {
            "noise_embeds": np.random.randn(2, 5, 16).astype("float32"),
            "padding_mask": np.ones((2, 5), dtype="int32"),
            "context_hidden_states": np.random.randn(2, 5, 32).astype(
                "float32"
            ),
        }
        self.run_backbone_test(
            cls=MuseGlimmerBackbone,
            init_kwargs=init_kwargs,
            input_data=input_data,
            expected_output_shape=(2, 5, 16),
        )
        config = MuseGlimmerBackbone(**init_kwargs).get_config()
        self.assertTrue(config["qk_norm_with_scale"])
        self.assertFalse(config["use_centered_norm"])
        restored = MuseGlimmerBackbone.from_config(config)
        self.assertTrue(restored.qk_norm_with_scale)
        self.assertFalse(restored.use_centered_norm)

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        # Sizes match the assistant `config.json`: hidden_size 6656 and
        # five context layers.
        assistant_input_data = {
            "noise_embeds": np.random.randn(1, 4, 6656).astype("float32"),
            "padding_mask": np.ones((1, 4), dtype="int32"),
            "context_hidden_states": np.random.randn(1, 4, 5 * 6656).astype(
                "float32"
            ),
        }
        for preset in MuseGlimmerBackbone.presets:
            if preset.endswith("_assistant"):
                input_data = assistant_input_data
            else:
                input_data = self.text_backbone_input_data
            self.run_preset_test(
                cls=MuseGlimmerBackbone,
                preset=preset,
                input_data=input_data,
            )
