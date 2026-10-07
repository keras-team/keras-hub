import os

import numpy as np

from keras_hub.src.models.gemma.gemma_backbone import GemmaBackbone
from keras_hub.src.tests.test_case import TestCase


class GemmaDoraTest(TestCase):
    def setUp(self):
        self._init_kwargs = {
            "vocabulary_size": 50,
            "num_layers": 2,
            "num_query_heads": 2,
            "num_key_value_heads": 2,
            "hidden_dim": 32,
            "intermediate_dim": 16,
            "head_dim": 16,
            "layer_norm_epsilon": 1e-6,
        }

    def test_dora_fine_tuning(self):
        # Set up backbone and preprocessor.
        backbone = GemmaBackbone(**self._init_kwargs)
        backbone.enable_dora(4)
        # 4 layers, 3 weights per layer
        self.assertLen(backbone.trainable_weights, 4 * 3)
        self.assertLen(backbone.non_trainable_weights, 20)
        input_data = {
            "token_ids": np.ones((2, 5), dtype="int32"),
            "padding_mask": np.ones((2, 5), dtype="int32"),
        }
        targets = np.random.normal(size=(2, 5, self._init_kwargs["hidden_dim"]))

        # Test fine-tuning
        backbone.compile(optimizer="sgd", loss="mse")
        backbone.fit(input_data, targets, epochs=1)

        # Test saving and reloading.
        temp_filepath = os.path.join(
            self.get_temp_dir(), "dora_model.weights.h5"
        )
        backbone.save_weights(temp_filepath)
        new_backbone = GemmaBackbone(**self._init_kwargs)
        new_backbone.load_weights(temp_filepath)
        ref_out = backbone(input_data)
        new_out = new_backbone(input_data)
        self.assertAllClose(ref_out, new_out)

    def test_dora_fine_tuning_target_names(self):
        # Set up backbone and preprocessor.
        backbone = GemmaBackbone(**self._init_kwargs)
        backbone.enable_dora(4, target_layer_names=["query"])
        # 2 layers, 3 weights per layer
        self.assertLen(backbone.trainable_weights, 2 * 3)
        self.assertLen(backbone.non_trainable_weights, 20)
        input_data = {
            "token_ids": np.ones((2, 5), dtype="int32"),
            "padding_mask": np.ones((2, 5), dtype="int32"),
        }
        targets = np.random.normal(size=(2, 5, self._init_kwargs["hidden_dim"]))

        # Test fine-tuning
        backbone.compile(optimizer="sgd", loss="mse")
        backbone.fit(input_data, targets, epochs=1)

        # Test saving and reloading.
        temp_filepath = os.path.join(
            self.get_temp_dir(), "dora_model.weights.h5"
        )
        backbone.save_weights(temp_filepath)
        new_backbone = GemmaBackbone(**self._init_kwargs)
        new_backbone.load_weights(temp_filepath)
        ref_out = backbone(input_data)
        new_out = new_backbone(input_data)
        self.assertAllClose(ref_out, new_out)

    def test_dora_saving_and_reloading(self):
        backbone = GemmaBackbone(**self._init_kwargs)
        initial_model_filepath = os.path.join(
            self.get_temp_dir(), "base.weights.h5"
        )
        backbone.save_weights(initial_model_filepath)

        backbone.enable_dora(4)
        input_data = {
            "token_ids": np.ones((2, 5), dtype="int32"),
            "padding_mask": np.ones((2, 5), dtype="int32"),
        }
        targets = np.random.normal(size=(2, 5, self._init_kwargs["hidden_dim"]))
        backbone.compile(optimizer="sgd", loss="mse")
        backbone.fit(input_data, targets, epochs=1)

        dora_filepath = os.path.join(self.get_temp_dir(), "dora_model.dora.h5")
        backbone.save_dora_weights(dora_filepath)

        # New backbone with same initial weights
        new_backbone = GemmaBackbone(**self._init_kwargs)
        new_backbone.load_weights(initial_model_filepath)
        new_backbone.enable_dora(4)
        new_backbone.load_dora_weights(dora_filepath)

        ref_out = backbone(input_data)
        new_out = new_backbone(input_data)
        self.assertAllClose(ref_out, new_out)

        # Test exceptions
        backbone = GemmaBackbone(**self._init_kwargs)
        with self.assertRaisesRegex(ValueError, "no dora-enabled layers"):
            backbone.save_dora_weights(dora_filepath)
        backbone.enable_dora(5)
        with self.assertRaisesRegex(ValueError, "ranks must match"):
            backbone.load_dora_weights(dora_filepath)
        with self.assertRaisesRegex(ValueError, "filename must end in"):
            backbone.save_dora_weights("bad_filepath")
