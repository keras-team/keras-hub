"""Tests for BLIP-2 custom OPT."""

import keras
import numpy as np
import pytest

from keras_hub.src.models.blip2.blip2_custom_opt import Blip2CustomOPT
from keras_hub.src.tests.test_case import TestCase


class Blip2CustomOPTTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 100,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 64,
            "intermediate_dim": 128,
            "max_sequence_length": 15,
            "qformer_hidden_dim": 64,
            "num_query_tokens": 2,
        }
        self.input_data = {
            "qformer_features": np.random.uniform(size=(2, 2, 64)).astype(
                "float32"
            ),
            "token_ids": np.ones((2, 5), dtype="int32"),
            "padding_mask": np.ones((2, 5), dtype="bool"),
        }

    def test_backbone_call(self):
        backbone = Blip2CustomOPT(**self.init_kwargs)
        output = backbone(self.input_data)
        # 2 visual tokens + 5 text tokens = 7 total tokens
        self.assertEqual(output.shape, (2, 7, 64))

    def test_token_embedding_access(self):
        backbone = Blip2CustomOPT(**self.init_kwargs)
        token_ids = np.array([[1, 2, 3]])
        embeddings = backbone.token_embedding(token_ids)
        self.assertEqual(embeddings.shape, (1, 3, 64))

    def test_serialization(self):
        backbone = Blip2CustomOPT(**self.init_kwargs)
        new_backbone = Blip2CustomOPT.from_config(backbone.get_config())
        self.assertEqual(new_backbone.get_config(), backbone.get_config())

    def test_position_embeddings(self):
        backbone = Blip2CustomOPT(**self.init_kwargs)
        # Shift qformer features significantly
        data1 = self.input_data
        data2 = {
            "qformer_features": data1["qformer_features"] + 10.0,
            "token_ids": data1["token_ids"],
            "padding_mask": data1["padding_mask"],
        }
        out1 = backbone(data1)
        out2 = backbone(data2)
        # Position embeddings for token_ids should be same, but
        # since it's a transformer, changes in qformer_features
        # will affect all subsequent outputs.
        self.assertNotAllClose(out1, out2)

    @pytest.mark.large
    def test_saved_model(self):
        from keras import ops

        backbone = Blip2CustomOPT(**self.init_kwargs)
        model_path = self.get_temp_dir() + "/model.keras"
        backbone.save(model_path)
        reloaded_model = keras.models.load_model(model_path)

        # Verify output using backend-agnostic conversion
        np.testing.assert_allclose(
            ops.convert_to_numpy(backbone(self.input_data)),
            ops.convert_to_numpy(reloaded_model(self.input_data)),
            atol=1e-5,
        )
