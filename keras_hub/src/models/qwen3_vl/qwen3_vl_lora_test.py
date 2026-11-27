import os

import numpy as np

from keras_hub.src.models.qwen3_vl.qwen3_vl_backbone import Qwen3VLBackbone
from keras_hub.src.models.qwen3_vl.qwen3_vl_vision_encoder import (
    Qwen3VLVisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3VLLoraTest(TestCase):
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
        backbone = Qwen3VLBackbone(**self.init_kwargs)
        backbone.enable_lora(4)

        # Only the query and value LoRA kernels of the 2 layers train.
        self.assertLen(backbone.trainable_weights, 8)
        self.assertLen(backbone.non_trainable_weights, 25)

        backbone.compile(optimizer="sgd", loss="mse")
        backbone.fit(self.input_data, self.targets, epochs=1)

        # Test saving and reloading.
        temp_filepath = os.path.join(
            self.get_temp_dir(), "lora_model.weights.h5"
        )
        backbone.save_weights(temp_filepath)
        new_backbone = Qwen3VLBackbone(**self.init_kwargs)
        new_backbone.load_weights(temp_filepath)
        self.assertAllClose(
            backbone(self.input_data), new_backbone(self.input_data)
        )

    def test_target_layer_names(self):
        backbone = Qwen3VLBackbone(**self.init_kwargs)
        backbone.enable_lora(4, target_layer_names=["query"])
        self.assertLen(backbone.trainable_weights, 4)
        for weight in backbone.trainable_weights:
            self.assertIn("/query/lora_kernel", weight.path)

    def test_multimodal_lora(self):
        vision_encoder = Qwen3VLVisionEncoder(
            depth=2,
            hidden_size=16,
            num_heads=2,
            intermediate_size=32,
            out_hidden_size=16,
            patch_size=4,
            temporal_patch_size=2,
            spatial_merge_size=2,
            num_position_embeddings=64,
            deepstack_visual_indexes=(0,),
        )
        backbone = Qwen3VLBackbone(
            **self.init_kwargs, vision_encoder=vision_encoder
        )
        targets = ["query", "value", "qkv"]
        backbone.enable_lora(4, target_layer_names=targets)
        paths = [w.path for w in backbone.trainable_weights]
        # The text attention and the vision `qkv` projections train. Keras
        # `Dense` LoRA keeps the bias trainable, so only the kernels freeze.
        self.assertTrue(any("/qkv/lora_kernel" in p for p in paths))
        self.assertFalse(any(p.endswith("/kernel") for p in paths))
        # 4 vision patches merge into 1 token at position 2.
        inputs = dict(self.input_data)
        inputs["pixel_values"] = (
            np.random.RandomState(0).rand(2, 4, 2, 4, 4, 3).astype("float32")
        )
        inputs["image_grid_thw"] = np.array([[[1, 2, 2]]] * 2, dtype="int32")
        inputs["vision_indices"] = np.array([[2], [2]], dtype="int32")
        backbone.compile(optimizer="sgd", loss="mse")
        backbone.fit(inputs, self.targets, epochs=1, verbose=0)

        # `.lora.h5` files hold only the LoRA kernels, not the trained
        # `qkv` bias, so reload through the full weights.
        filepath = os.path.join(self.get_temp_dir(), "vl_lora.weights.h5")
        backbone.save_weights(filepath)
        new_backbone = Qwen3VLBackbone.from_config(backbone.get_config())
        new_backbone.enable_lora(4, target_layer_names=targets)
        new_backbone.load_weights(filepath)
        self.assertAllClose(backbone(inputs), new_backbone(inputs))

    def test_lora_saving_and_reloading(self):
        backbone = Qwen3VLBackbone(**self.init_kwargs)
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
        new_backbone = Qwen3VLBackbone(**self.init_kwargs)
        new_backbone.load_weights(initial_model_filepath)
        new_backbone.enable_lora(4)
        new_backbone.load_lora_weights(lora_filepath)
        self.assertAllClose(
            backbone(self.input_data), new_backbone(self.input_data)
        )

        # Test exceptions.
        backbone = Qwen3VLBackbone(**self.init_kwargs)
        with self.assertRaisesRegex(ValueError, "no lora-enabled layers"):
            backbone.save_lora_weights(lora_filepath)
        backbone.enable_lora(5)
        with self.assertRaisesRegex(ValueError, "ranks must match"):
            backbone.load_lora_weights(lora_filepath)
        with self.assertRaisesRegex(ValueError, "filename must end in"):
            backbone.save_lora_weights("bad_filepath")
