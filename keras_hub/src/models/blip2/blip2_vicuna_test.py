import keras
import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.blip2.blip2_vicuna import BLIP2Vicuna
from keras_hub.src.tests.test_case import TestCase


class BLIP2VicunaTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 100,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 4,
            "hidden_dim": 32,
            "intermediate_dim": 64,
            "num_query_tokens": 2,
            "qformer_hidden_dim": 32,
            "language_projection": None,
        }
        self.input_data = {
            "qformer_features": np.random.uniform(size=(2, 2, 32)).astype(
                "float32"
            ),
            "token_ids": np.ones((2, 5), dtype="int32"),
            "padding_mask": np.ones((2, 5), dtype="int32"),
        }

    def test_call(self):
        lm = BLIP2Vicuna(**self.init_kwargs)
        output = lm(self.input_data)
        # 2 visual query tokens prepended to 5 text tokens.
        self.assertEqual(output.shape, (2, 7, 32))

    def test_no_learned_positions_flag(self):
        lm = BLIP2Vicuna(**self.init_kwargs)
        self.assertFalse(lm.uses_learned_positions)
        # Cache geometry exposed for BLIP2CausalLM (no grouped-query attention).
        self.assertEqual(lm.num_heads, 4)
        self.assertEqual(lm.head_dim, 8)

    def test_token_embedding_access(self):
        lm = BLIP2Vicuna(**self.init_kwargs)
        token_ids = np.array([[1, 2, 3]])
        embeddings = lm.token_embedding(token_ids)
        self.assertEqual(embeddings.shape, (1, 3, 32))

    def test_call_with_cache(self):
        lm = BLIP2Vicuna(**self.init_kwargs)
        seq_len = 7
        x = ops.convert_to_tensor(
            np.random.uniform(size=(2, seq_len, 32)).astype("float32")
        )
        padding_mask = ops.ones((2, seq_len), dtype="int32")
        cache = ops.zeros((2, lm.num_layers, 2, seq_len, lm.num_heads, 8))
        hidden, new_cache = lm.call_with_cache(x, padding_mask, cache, 0)
        self.assertEqual(hidden.shape, (2, seq_len, 32))
        self.assertEqual(
            new_cache.shape, (2, lm.num_layers, 2, seq_len, lm.num_heads, 8)
        )

    def test_serialization(self):
        lm = BLIP2Vicuna(**self.init_kwargs)
        new_lm = BLIP2Vicuna.from_config(lm.get_config())
        self.assertEqual(new_lm.get_config(), lm.get_config())

    @pytest.mark.large
    def test_saved_model(self):
        lm = BLIP2Vicuna(**self.init_kwargs)
        model_path = self.get_temp_dir() + "/model.keras"
        lm.save(model_path)
        reloaded_model = keras.models.load_model(model_path)
        np.testing.assert_allclose(
            ops.convert_to_numpy(lm(self.input_data)),
            ops.convert_to_numpy(reloaded_model(self.input_data)),
            atol=1e-5,
        )
