import numpy as np
import pytest

from keras_hub.src.models.blip2.blip2_custom_opt import BLIP2CustomOPT
from keras_hub.src.tests.test_case import TestCase


class BLIP2CustomOPTTest(TestCase):
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
            "dropout": 0.0,
            "language_projection": None,
        }
        self.input_data = {
            "qformer_features": np.random.uniform(size=(2, 2, 64)).astype(
                "float32"
            ),
            "token_ids": np.ones((2, 5), dtype="int32"),
            "padding_mask": np.ones((2, 5), dtype="bool"),
        }

    def test_backbone_call(self):
        backbone = BLIP2CustomOPT(**self.init_kwargs)
        output = backbone(self.input_data)
        self.assertEqual(output.shape, (2, 7, 64))

    def test_token_embedding_access(self):
        backbone = BLIP2CustomOPT(**self.init_kwargs)
        token_ids = np.array([[1, 2, 3]])
        embeddings = backbone.token_embedding(token_ids)
        self.assertEqual(embeddings.shape, (1, 3, 64))

    def test_serialization(self):
        backbone = BLIP2CustomOPT(**self.init_kwargs)
        self.run_serialization_test(backbone)

    def test_position_embeddings(self):
        backbone = BLIP2CustomOPT(**self.init_kwargs)
        data1 = self.input_data
        data2 = {
            "qformer_features": data1["qformer_features"] + 10.0,
            "token_ids": data1["token_ids"],
            "padding_mask": data1["padding_mask"],
        }
        out1 = backbone(data1)
        out2 = backbone(data2)
        self.assertNotAllClose(out1, out2)

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=BLIP2CustomOPT,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            atol=1e-5,
            rtol=1e-5,
        )
