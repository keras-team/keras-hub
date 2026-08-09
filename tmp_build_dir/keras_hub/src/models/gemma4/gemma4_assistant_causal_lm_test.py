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
    def setUp(self):
        self.backbone = Gemma4Backbone(
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
        self.model = Gemma4AssistantCausalLM(
            preprocessor=None,
            backbone=self.backbone,
            backbone_hidden_size=16,
            num_centroids=4,
            centroid_intermediate_top_k=2,
            use_ordered_embeddings=True,
        )

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

    def test_speculative_generate(self):
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
        target_model = Gemma4CausalLM(
            preprocessor=None,
            backbone=target_backbone,
        )

        batch_size = 1
        max_length = 20
        seq_len = 5
        token_ids_raw = np.random.randint(0, 100, (batch_size, seq_len))
        token_ids = np.zeros((batch_size, max_length), dtype="int32")
        token_ids[:, :seq_len] = token_ids_raw
        padding_mask = np.zeros((batch_size, max_length), dtype="bool")
        padding_mask[:, :seq_len] = True
        token_ids = ops.convert_to_tensor(token_ids)
        padding_mask = ops.convert_to_tensor(padding_mask)

        output = target_model.generate(
            {
                "token_ids": token_ids,
                "padding_mask": padding_mask,
            },
            assistant_model=self.model,
            stop_token_ids=None,
        )
        self.assertIsNotNone(output)

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
