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

    def test_presets(self):
        expected_presets = [
            "pythia_70m_en",
            "pythia_160m_en",
            "pythia_410m_en",
            "pythia_1.4b_en",
            "pythia_2.8b_en",
            "pythia_6.9b_en",
            "pythia_12b_en",
            "gpt_neox_20b_en",
        ]
        for preset in expected_presets:
            self.assertIn(preset, GPTNeoXBackbone.presets)
            self.assertTrue(
                GPTNeoXBackbone.presets[preset]["kaggle_handle"].startswith(
                    f"kaggle://keras/gpt_neox/keras/{preset}/"
                )
            )

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=GPTNeoXBackbone,
            preset="pythia_70m_en",
            input_data={
                "token_ids": ops.array([[1, 2, 3, 4]], dtype="int32"),
                "padding_mask": ops.array([[1, 1, 1, 1]], dtype="int32"),
            },
            expected_output_shape=(1, 4, 512),
            expected_partial_output=None,
        )
