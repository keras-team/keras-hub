import os

import numpy as np

from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
from keras_hub.src.tests.test_case import TestCase


class Gemma3LoraTest(TestCase):
    def setUp(self):
        self.text_init_kwargs = {
            # vocabulary
            "vocabulary_size": 256,
            # image
            "image_size": 16,
            # model
            "num_layers": 6,
            "num_query_heads": 2,
            "num_key_value_heads": 1,
            "hidden_dim": 8,
            "intermediate_dim": 16,
            "head_dim": 4,
            # other model args
            "query_head_dim_normalize": True,
            "use_query_key_norm": True,
            "use_post_ffw_norm": True,
            "use_post_attention_norm": True,
            "final_logit_soft_cap": None,
            "attention_logit_soft_cap": None,
            "use_sliding_window_attention": True,
            "sliding_window_size": 1024,
            "vision_encoder": None,
        }

    def test_text_lora_fine_tuning(self):
        # Set up backbone and preprocessor.
        backbone = Gemma3Backbone(**self.text_init_kwargs)
        backbone.enable_lora(4)

        self.assertLen(backbone.trainable_weights, 24)
        self.assertLen(backbone.non_trainable_weights, 80)
        input_data = {
            "token_ids": np.ones((2, 5), dtype="int32"),
            "padding_mask": np.ones((2, 5), dtype="int32"),
        }
        targets = np.random.normal(
            size=(2, 5, self.text_init_kwargs["hidden_dim"])
        )

        # Test fine-tuning
        backbone.compile(optimizer="sgd", loss="mse")
        backbone.fit(input_data, targets, epochs=1)

        # Test saving and reloading.
        temp_filepath = os.path.join(
            self.get_temp_dir(), "lora_model.weights.h5"
        )
        backbone.save_weights(temp_filepath)
        new_backbone = Gemma3Backbone(**self.text_init_kwargs)
        new_backbone.load_weights(temp_filepath)
        ref_out = backbone(input_data)
        new_out = new_backbone(input_data)
        self.assertAllClose(ref_out, new_out)

    def test_text_lora_saving_and_reloading(self):
        backbone = Gemma3Backbone(**self.text_init_kwargs)
        initial_model_filepath = os.path.join(
            self.get_temp_dir(), "base.weights.h5"
        )
        backbone.save_weights(initial_model_filepath)

        backbone.enable_lora(4)
        input_data = {
            "token_ids": np.ones((2, 5), dtype="int32"),
            "padding_mask": np.ones((2, 5), dtype="int32"),
        }
        targets = np.random.normal(
            size=(2, 5, self.text_init_kwargs["hidden_dim"])
        )
        backbone.compile(optimizer="sgd", loss="mse")
        backbone.fit(input_data, targets, epochs=1)

        lora_filepath = os.path.join(self.get_temp_dir(), "lora_model.lora.h5")
        backbone.save_lora_weights(lora_filepath)

        # New backbone with same initial weights
        new_backbone = Gemma3Backbone(**self.text_init_kwargs)
        new_backbone.load_weights(initial_model_filepath)
        new_backbone.enable_lora(4)
        new_backbone.load_lora_weights(lora_filepath)

        ref_out = backbone(input_data)
        new_out = new_backbone(input_data)
        self.assertAllClose(ref_out, new_out)

        # Test exceptions
        backbone = Gemma3Backbone(**self.text_init_kwargs)
        with self.assertRaisesRegex(ValueError, "no lora-enabled layers"):
            backbone.save_lora_weights(lora_filepath)
        backbone.enable_lora(5)
        with self.assertRaisesRegex(ValueError, "ranks must match"):
            backbone.load_lora_weights(lora_filepath)
        with self.assertRaisesRegex(ValueError, "filename must end in"):
            backbone.save_lora_weights("bad_filepath")
