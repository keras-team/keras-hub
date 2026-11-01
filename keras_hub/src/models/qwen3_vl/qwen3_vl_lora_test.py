import os

import numpy as np

from keras_hub.src.models.qwen3_vl.qwen3_vl_backbone import Qwen3_VLBackbone
from keras_hub.src.tests.test_case import TestCase


class Qwen3_VLLoraTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 64,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "hidden_dim": 16,
            "intermediate_dim": 32,
            "mrope_section": [2, 1, 1],
        }
        self.input_data = {
            "token_ids": np.ones((2, 5), dtype="int32"),
            "padding_mask": np.ones((2, 5), dtype="int32"),
            "position_ids": np.tile(
                np.arange(5, dtype="int32")[None, :, None], (2, 1, 3)
            ),
        }
        self.targets = np.random.normal(size=(2, 5, 16))

    def test_lora_fine_tuning(self):
        backbone = Qwen3_VLBackbone(**self.init_kwargs)
        backbone.enable_lora(4)

        # Ensure that only LoRA weights are trainable.
        self.assertGreater(len(backbone.trainable_weights), 0)
        self.assertGreater(len(backbone.non_trainable_weights), 0)

        backbone.compile(optimizer="sgd", loss="mse")
        backbone.fit(self.input_data, self.targets, epochs=1)

        # Test saving and reloading.
        temp_filepath = os.path.join(
            self.get_temp_dir(), "lora_model.weights.h5"
        )
        backbone.save_weights(temp_filepath)
        new_backbone = Qwen3_VLBackbone(**self.init_kwargs)
        new_backbone.load_weights(temp_filepath)
        self.assertAllClose(
            backbone(self.input_data), new_backbone(self.input_data)
        )

    def test_lora_saving_and_reloading(self):
        backbone = Qwen3_VLBackbone(**self.init_kwargs)
        initial_model_filepath = os.path.join(
            self.get_temp_dir(), "base.weights.h5"
        )
        backbone.save_weights(initial_model_filepath)

        backbone.enable_lora(4)
        backbone.compile(optimizer="sgd", loss="mse")
        backbone.fit(self.input_data, self.targets, epochs=1)

        lora_filepath = os.path.join(self.get_temp_dir(), "lora_model.lora.h5")
        backbone.save_lora_weights(lora_filepath)

        # New backbone with same initial weights.
        new_backbone = Qwen3_VLBackbone(**self.init_kwargs)
        new_backbone.load_weights(initial_model_filepath)
        new_backbone.enable_lora(4)
        new_backbone.load_lora_weights(lora_filepath)
        self.assertAllClose(
            backbone(self.input_data), new_backbone(self.input_data)
        )

        # Test exceptions.
        backbone = Qwen3_VLBackbone(**self.init_kwargs)
        with self.assertRaisesRegex(ValueError, "no lora-enabled layers"):
            backbone.save_lora_weights(lora_filepath)
        backbone.enable_lora(5)
        with self.assertRaisesRegex(ValueError, "ranks must match"):
            backbone.load_lora_weights(lora_filepath)
        with self.assertRaisesRegex(ValueError, "filename must end in"):
            backbone.save_lora_weights("bad_filepath")
