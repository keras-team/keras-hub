import numpy as np
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.muse_glimmer.muse_glimmer_backbone import (
    MuseGlimmerBackbone,
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
            # The vision encoder's position-embedding interpolation isn't
            # dtype-policy-aware, so it doesn't follow the mixed-precision
            # policy applied to the backbone.
            run_mixed_precision_check=(backbone_type != "text_and_vision"),
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

    def test_num_parameters(self):
        model = MuseGlimmerBackbone(**self.text_init_kwargs)
        self.assertGreater(model.count_params(), 0)

    def test_multimodal_backbone_builds(self):
        model = MuseGlimmerBackbone(**self.init_kwargs)
        self.assertGreater(model.count_params(), 0)
        self.assertIsNotNone(model.vision_encoder)
        self.assertTrue(hasattr(model, "interleave_embeddings"))

    def test_text_only_call_on_multimodal_backbone(self):
        model = MuseGlimmerBackbone(**self.init_kwargs)
        output = model(self.text_backbone_input_data)
        self.assertEqual(ops.shape(output), (2, 5, 32))

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

    def test_assistant_configuration_forward(self):
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
        model = MuseGlimmerBackbone(**init_kwargs)
        input_data = {
            "noise_embeds": np.random.randn(2, 5, 16).astype("float32"),
            "padding_mask": np.ones((2, 5), dtype="int32"),
            "context_hidden_states": np.random.randn(2, 5, 32).astype(
                "float32"
            ),
        }
        output = model(input_data)
        self.assertEqual(ops.shape(output), (2, 5, 16))

    def test_assistant_config_get_config_round_trip(self):
        init_kwargs = dict(
            self.assistant_base_kwargs,
            use_bidirectional_attention=True,
            context_projection_layer_ids=[0, 1],
            use_external_embeddings=True,
            enable_qk_scale_and_gate=False,
            use_sandwich_norm=False,
        )
        model = MuseGlimmerBackbone(**init_kwargs)
        restored = MuseGlimmerBackbone.from_config(model.get_config())
        self.assertTrue(restored.use_bidirectional_attention)
        self.assertEqual(restored.context_projection_layer_ids, [0, 1])
        self.assertTrue(restored.use_external_embeddings)
        self.assertFalse(restored.enable_qk_scale_and_gate)
        self.assertFalse(restored.use_sandwich_norm)
