import os

import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.muse_glimmer.muse_glimmer_assistant_causal_lm import (
    MuseGlimmerAssistantCausalLM,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_backbone import (
    MuseGlimmerBackbone,
)
from keras_hub.src.tests.test_case import TestCase


class MuseGlimmerAssistantCausalLMTest(TestCase):
    def setUp(self):
        self.backbone = MuseGlimmerBackbone(
            vocabulary_size=20,
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=16,
            intermediate_dim=32,
            head_dim=4,
            sliding_window_size=4,
            layer_types=["sliding_attention", "sliding_attention"],
            use_bidirectional_attention=True,
            context_projection_layer_ids=[0, 1],
            use_external_embeddings=True,
            enable_qk_scale_and_gate=False,
            use_sandwich_norm=False,
        )
        self.model = MuseGlimmerAssistantCausalLM(
            backbone=self.backbone,
            block_size=5,
        )

    def test_call_with_cache(self):
        batch_size = 2
        block_size = 5
        max_length = 4
        noise_embeds = np.random.randn(batch_size, block_size, 16).astype(
            "float32"
        )
        context_hidden_states = np.random.randn(batch_size, 1, 32).astype(
            "float32"
        )
        padding_mask = np.ones((batch_size, block_size), dtype="int32")
        # (batch, num_layers, 2, max_length, num_key_value_heads, head_dim)
        cache = ops.zeros((batch_size, 2, 2, max_length, 2, 4))

        output, next_cache = self.model.call_with_cache(
            noise_embeds=noise_embeds,
            context_hidden_states=context_hidden_states,
            cache=cache,
            cache_update_index=0,
            padding_mask=padding_mask,
        )
        self.assertEqual(ops.shape(output), (batch_size, block_size, 16))
        self.assertEqual(
            ops.shape(next_cache), (batch_size, 2, 2, max_length, 2, 4)
        )

    def test_call_with_cache_default_padding_mask(self):
        batch_size = 1
        block_size = 5
        max_length = 4
        noise_embeds = np.random.randn(batch_size, block_size, 16).astype(
            "float32"
        )
        context_hidden_states = np.random.randn(batch_size, 1, 32).astype(
            "float32"
        )
        cache = ops.zeros((batch_size, 2, 2, max_length, 2, 4))

        output, next_cache = self.model.call_with_cache(
            noise_embeds=noise_embeds,
            context_hidden_states=context_hidden_states,
            cache=cache,
            cache_update_index=0,
        )
        self.assertEqual(ops.shape(output), (batch_size, block_size, 16))
        self.assertEqual(
            ops.shape(next_cache), (batch_size, 2, 2, max_length, 2, 4)
        )

    def test_call_with_cache_persists_context_across_cycles(self):
        batch_size = 1
        block_size = 5
        max_length = 4
        # (batch, num_layers, 2, max_length, num_key_value_heads, head_dim)
        cache = ops.zeros((batch_size, 2, 2, max_length, 2, 4))

        noise_embeds_0 = np.random.randn(batch_size, block_size, 16).astype(
            "float32"
        )
        context_0 = np.random.randn(batch_size, 1, 32).astype("float32")
        _, cache = self.model.call_with_cache(
            noise_embeds=noise_embeds_0,
            context_hidden_states=context_0,
            cache=cache,
            cache_update_index=0,
        )
        key_cache_0 = ops.convert_to_numpy(cache[:, 0, 0, 0, ...])

        # A second cycle writes a new context position further along;
        # the first cycle's cached slot must be unchanged.
        noise_embeds_1 = np.random.randn(batch_size, block_size, 16).astype(
            "float32"
        )
        context_1 = np.random.randn(batch_size, 1, 32).astype("float32")
        _, cache = self.model.call_with_cache(
            noise_embeds=noise_embeds_1,
            context_hidden_states=context_1,
            cache=cache,
            cache_update_index=1,
        )
        key_cache_0_after = ops.convert_to_numpy(cache[:, 0, 0, 0, ...])
        self.assertAllClose(key_cache_0, key_cache_0_after)

    def test_call_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.model(
                {
                    "noise_embeds": np.zeros((1, 5, 16), dtype="float32"),
                    "padding_mask": np.ones((1, 5), dtype="int32"),
                    "context_hidden_states": np.zeros(
                        (1, 3, 32), dtype="float32"
                    ),
                }
            )

    def test_generate_step_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.model.generate_step(
                {
                    "noise_embeds": np.zeros((1, 5, 16), dtype="float32"),
                    "padding_mask": np.ones((1, 5), dtype="int32"),
                }
            )

    def test_get_config(self):
        config = self.model.get_config()
        self.assertEqual(config["block_size"], 5)

    @pytest.mark.large
    def test_model_saving(self):
        import keras

        path = os.path.join(self.get_temp_dir(), "model.keras")
        self.model.save(path)
        loaded_model = keras.saving.load_model(path)
        self.assertIsInstance(loaded_model, MuseGlimmerAssistantCausalLM)

        batch_size = 1
        block_size = 5
        max_length = 4
        noise_embeds = np.zeros((batch_size, block_size, 16), dtype="float32")
        context_hidden_states = np.zeros((batch_size, 1, 32), dtype="float32")
        cache = ops.zeros((batch_size, 2, 2, max_length, 2, 4))
        output_orig, _ = self.model.call_with_cache(
            noise_embeds=noise_embeds,
            context_hidden_states=context_hidden_states,
            cache=cache,
            cache_update_index=0,
        )
        output_loaded, _ = loaded_model.call_with_cache(
            noise_embeds=noise_embeds,
            context_hidden_states=context_hidden_states,
            cache=cache,
            cache_update_index=0,
        )
        self.assertAllClose(output_orig, output_loaded)
