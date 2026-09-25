import os

import numpy as np
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.gemma4.gemma4_assistant_causal_lm import (
    Gemma4AssistantCausalLM,
)
from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone
from keras_hub.src.models.gemma4.gemma4_causal_lm import Gemma4CausalLM
from keras_hub.src.tests.test_case import TestCase


class Gemma4AssistantTest(TestCase, parameterized.TestCase):
    def _create_assistant(self):
        backbone = Gemma4Backbone(
            vocabulary_size=256,
            num_layers=4,
            num_query_heads=4,
            num_key_value_heads=1,
            hidden_dim=8,
            intermediate_dim=16,
            head_dim=4,
            global_head_dim=8,
            image_size=16,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
        )
        # backbone_hidden_size=16 matches the target hidden_dim used in tests.
        return Gemma4AssistantCausalLM(
            preprocessor=None,
            backbone=backbone,
            backbone_hidden_size=16,
            num_centroids=4,
            centroid_intermediate_top_k=2,
            use_ordered_embeddings=True,
        )

    def _make_inputs(self, token_ids_raw):
        batch_size = token_ids_raw.shape[0]
        seq_len = token_ids_raw.shape[1]
        max_length = 20
        token_ids = np.zeros((batch_size, max_length), dtype="int32")
        token_ids[:, :seq_len] = token_ids_raw
        padding_mask = np.zeros((batch_size, max_length), dtype="bool")
        padding_mask[:, :seq_len] = True
        return {
            "token_ids": ops.convert_to_tensor(token_ids),
            "padding_mask": ops.convert_to_tensor(padding_mask),
        }

    def setUp(self):
        self.model = self._create_assistant()
        self.backbone = self.model.backbone

    def test_call_with_cache(self):
        batch_size = 2
        target_num_layers = 6
        max_head_dim = 8  # max(head_dim=4, global_head_dim=8)
        target_kv_heads = 1
        cache_seq = 5

        target_cache = np.zeros(
            (
                batch_size,
                target_num_layers,
                2,
                cache_seq,
                target_kv_heads,
                max_head_dim,
            ),
            dtype="float32",
        )
        target_cache = ops.convert_to_tensor(target_cache)

        last_token_embedding = ops.convert_to_tensor(
            np.random.randn(batch_size, 1, 16).astype("float32")
        )
        last_hidden_state = ops.convert_to_tensor(
            np.random.randn(batch_size, 1, 16).astype("float32")
        )

        logits, next_hidden = self.model.call_with_cache(
            last_token_embedding=last_token_embedding,
            last_hidden_state=last_hidden_state,
            target_cache=target_cache,
            cache_update_index=cache_seq - 1,
        )

        self.assertEqual(ops.shape(logits), (batch_size, 1, 256))
        self.assertEqual(ops.shape(next_hidden), (batch_size, 1, 16))

    def _create_target_model(self):
        target_backbone = Gemma4Backbone(
            vocabulary_size=256,
            num_layers=6,
            num_query_heads=4,
            num_key_value_heads=1,
            hidden_dim=16,
            intermediate_dim=32,
            head_dim=8,
            image_size=16,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
        )
        return Gemma4CausalLM(
            preprocessor=None,
            backbone=target_backbone,
        )

    def test_speculative_generate(self):
        target_model = self._create_target_model()

        batch_size = 1
        seq_len = 5
        token_ids_raw = np.random.randint(0, 100, (batch_size, seq_len))
        inputs = self._make_inputs(token_ids_raw)

        output = target_model.generate(
            inputs,
            assistant_model=self.model,
            stop_token_ids=None,
        )
        self.assertIsNotNone(output)

    def test_assistant_weights_not_tracked(self):
        target_model = self._create_target_model()

        batch_size = 1
        seq_len = 5
        token_ids_raw = np.random.randint(0, 100, (batch_size, seq_len))
        inputs = self._make_inputs(token_ids_raw)

        initial_weights_len = len(target_model.weights)
        initial_trainable_len = len(target_model.trainable_weights)
        initial_count_params = target_model.count_params()

        target_model.generate(
            inputs,
            assistant_model=self.model,
            stop_token_ids=None,
        )

        self.assertEqual(len(target_model.weights), initial_weights_len)
        self.assertEqual(
            len(target_model.trainable_weights), initial_trainable_len
        )
        self.assertEqual(target_model.count_params(), initial_count_params)

        target_weight_ids = {id(w) for w in target_model.weights}
        for w in self.model.weights:
            self.assertNotIn(id(w), target_weight_ids)

    def test_generate_keeps_user_attached_assistant(self):
        target_model = self._create_target_model()
        assistant = self._create_assistant()

        # User deliberately attaches the assistant beforehand.
        target_model.user_draft = assistant

        batch_size = 1
        seq_len = 5
        token_ids_raw = np.random.randint(0, 100, (batch_size, seq_len))
        inputs = self._make_inputs(token_ids_raw)

        initial_weights_len = len(target_model.weights)

        target_model.generate(
            inputs,
            assistant_model=assistant,
            stop_token_ids=None,
        )

        self.assertEqual(len(target_model.weights), initial_weights_len)
        target_weight_ids = {id(w) for w in target_model.weights}
        for w in assistant.weights:
            self.assertIn(id(w), target_weight_ids)

    def test_new_assistant_rebuilds_speculative_graph(self):
        target_model = self._create_target_model()

        token_ids_raw = np.array([[10, 20, 30, 40, 50]], dtype="int32")
        inputs = self._make_inputs(token_ids_raw)

        assistant1 = self._create_assistant()
        assistant2 = self._create_assistant()

        target_model.generate(
            inputs,
            assistant_model=assistant1,
            stop_token_ids=None,
        )

        # Set up a spy on assistant2 to verify its Python method is invoked.
        # Without the cache-key fix, TF (tf.function reuse) and JAX (jit cache
        # hit) reuse the old compiled graph and never re-trace, so the spy
        # count stays 0 (red on TF/JAX). Torch is eager and will pass either
        # way.
        spy_call_count = [0]
        original_call_with_cache = assistant2.call_with_cache

        def spy_call(*args, **kwargs):
            spy_call_count[0] += 1
            return original_call_with_cache(*args, **kwargs)

        object.__setattr__(assistant2, "call_with_cache", spy_call)

        target_model.generate(
            inputs,
            assistant_model=assistant2,
            stop_token_ids=None,
        )

        self.assertGreater(spy_call_count[0], 0)

    def test_model_saving(self):
        import keras

        path = os.path.join(self.get_temp_dir(), "model.keras")
        self.model.save(path)
        loaded_model = keras.saving.load_model(path)

        self.assertIsInstance(loaded_model, Gemma4AssistantCausalLM)

        batch_size = 2
        target_num_layers = 6
        max_head_dim = 8
        target_kv_heads = 1
        cache_seq = 5
        target_cache = ops.zeros(
            (
                batch_size,
                target_num_layers,
                2,
                cache_seq,
                target_kv_heads,
                max_head_dim,
            )
        )
        last_token_embedding = ops.zeros((batch_size, 1, 16))
        last_hidden_state = ops.zeros((batch_size, 1, 16))

        logits_orig, h_orig = self.model.call_with_cache(
            last_token_embedding=last_token_embedding,
            last_hidden_state=last_hidden_state,
            target_cache=target_cache,
            cache_update_index=cache_seq - 1,
        )
        logits_loaded, h_loaded = loaded_model.call_with_cache(
            last_token_embedding=last_token_embedding,
            last_hidden_state=last_hidden_state,
            target_cache=target_cache,
            cache_update_index=cache_seq - 1,
        )
        self.assertAllClose(logits_orig, logits_loaded)
        self.assertAllClose(h_orig, h_loaded)
