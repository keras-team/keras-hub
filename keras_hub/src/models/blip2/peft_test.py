"""Tests for PEFT on BLIP-2."""

import os

import numpy as np

from keras_hub.src.models.blip2.blip2_backbone import Blip2Backbone
from keras_hub.src.models.blip2.blip2_custom_opt import Blip2CustomOPT
from keras_hub.src.models.blip2.blip2_qformer import Blip2QFormer
from keras_hub.src.models.blip2.blip2_vision_encoder import Blip2VisionEncoder
from keras_hub.src.tests.test_case import TestCase


class Blip2LoraTest(TestCase):
    def setUp(self):
        self.image_size = 32
        self.hidden_dim = 4
        self.vision_encoder_kwargs = {
            "image_size": self.image_size,
            "patch_size": 4,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 8,
            "intermediate_dim": 16,
            "use_patch_bias": True,
            "use_class_token": True,
            "use_mha_bias": True,
            "use_mlp_bias": True,
            "dropout_rate": 0.0,
            "layer_norm_epsilon": 1e-6,
        }
        self.qformer_kwargs = {
            "num_query_tokens": 2,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": self.hidden_dim,
            "intermediate_dim": 8,
            "vision_dim": 8,
            "cross_attention_frequency": 1,
        }
        self.language_model_kwargs = {
            "vocabulary_size": 14,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": self.hidden_dim,
            "intermediate_dim": 8,
            "num_query_tokens": 2,
            "qformer_hidden_dim": self.hidden_dim,
            "max_sequence_length": 10,
        }
        self.input_data = {
            "images": np.ones(
                (2, self.image_size, self.image_size, 3), dtype="float32"
            ),
            "token_ids": np.ones((2, 5), dtype="int32"),
            "padding_mask": np.ones((2, 5), dtype="bool"),
        }
        self.targets = np.random.normal(size=(2, 7, self.hidden_dim))

    def _get_init_kwargs(self):
        return {
            "vision_encoder": Blip2VisionEncoder(**self.vision_encoder_kwargs),
            "qformer": Blip2QFormer(**self.qformer_kwargs),
            "language_model": Blip2CustomOPT(**self.language_model_kwargs),
        }

    def test_lora_fine_tuning(self):
        backbone = Blip2Backbone(**self._get_init_kwargs())
        backbone.language_model.enable_lora(rank=4)

        backbone.compile(optimizer="sgd", loss="mse")
        backbone.fit(self.input_data, self.targets, epochs=1)

        path = os.path.join(self.get_temp_dir(), "lora_model.weights.h5")
        backbone.save_weights(path)

        new_backbone = Blip2Backbone(**self._get_init_kwargs())
        new_backbone.language_model.enable_lora(rank=4)
        new_backbone.load_weights(path)

        self.assertAllClose(
            backbone(self.input_data), new_backbone(self.input_data)
        )

    def test_lora_saving_and_reloading(self):
        backbone = Blip2Backbone(**self._get_init_kwargs())
        base_path = os.path.join(self.get_temp_dir(), "base.weights.h5")
        backbone.save_weights(base_path)

        backbone.language_model.enable_lora(rank=4)
        backbone.compile(optimizer="sgd", loss="mse")
        backbone.fit(self.input_data, self.targets, epochs=1)

        lora_path = os.path.join(self.get_temp_dir(), "lora.lora.h5")
        backbone.language_model.save_lora_weights(lora_path)

        new_backbone = Blip2Backbone(**self._get_init_kwargs())
        new_backbone.load_weights(base_path)
        new_backbone.language_model.load_lora_weights(lora_path)

        self.assertAllClose(
            backbone(self.input_data), new_backbone(self.input_data)
        )

        language_model = Blip2CustomOPT(**self.language_model_kwargs)
        with self.assertRaisesRegex(
            ValueError, "There are no lora-enabled layers"
        ):
            language_model.save_lora_weights(lora_path)
        language_model.enable_lora(rank=8)
        with self.assertRaisesRegex(ValueError, "ranks must match"):
            language_model.load_lora_weights(lora_path)
        with self.assertRaisesRegex(ValueError, "filename must end in"):
            language_model.save_lora_weights("bad_path")
