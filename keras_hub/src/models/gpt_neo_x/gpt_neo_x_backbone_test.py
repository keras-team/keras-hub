import keras
import pytest
from keras import ops

from keras_hub.src.models.gpt_neo_x.gpt_neo_x_backbone import GPTNeoXBackbone
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.keras_utils import gelu_approximate


class GPTNeoXBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 2,
            "intermediate_dim": 4,
            "max_sequence_length": 5,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=GPTNeoXBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 2),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=GPTNeoXBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_tie_word_embeddings_false(self):
        kwargs = dict(self.init_kwargs, tie_word_embeddings=False)
        backbone = GPTNeoXBackbone(**kwargs)
        self.assertFalse(backbone.token_embedding.tie_weights)
        self.assertEqual(
            backbone.token_embedding.reverse_embeddings.shape,
            (kwargs["hidden_dim"], kwargs["vocabulary_size"]),
        )
        self.assertFalse(backbone.get_config()["tie_word_embeddings"])

        restored = GPTNeoXBackbone.from_config(backbone.get_config())
        self.assertFalse(restored.token_embedding.tie_weights)

    def test_activation(self):
        kwargs = dict(self.init_kwargs, activation="gelu")
        backbone = GPTNeoXBackbone(**kwargs)
        for layer in backbone.transformer_layers:
            self.assertEqual(layer.activation, keras.activations.gelu)
        self.assertEqual(backbone.get_config()["activation"], "gelu")

        default_backbone = GPTNeoXBackbone(**self.init_kwargs)
        for layer in default_backbone.transformer_layers:
            self.assertEqual(layer.activation, gelu_approximate)
        default_serialized = keras.activations.serialize(gelu_approximate)
        self.assertEqual(
            default_backbone.get_config()["activation"], default_serialized
        )
