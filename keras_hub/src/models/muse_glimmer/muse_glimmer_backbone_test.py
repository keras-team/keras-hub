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
    MuseGlimmerVisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase


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
        # Each row must match the same prompt run alone.
        for i in range(2):
            single = {key: value[i : i + 1] for key, value in batch.items()}
            self.assertAllClose(
                ops.convert_to_numpy(output)[i],
                ops.convert_to_numpy(model(single))[0],
                atol=1e-4,
                rtol=1e-4,
            )

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
