import os

import keras
import numpy as np

from keras_hub.src.models.gemma.gemma_backbone import GemmaBackbone
from keras_hub.src.tests.test_case import TestCase


class GemmaDoRATest(TestCase):
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
        self.input_data = {
            "token_ids": np.ones((2, 5), dtype="int32"),
            "padding_mask": np.ones((2, 5), dtype="int32"),
        }

    def test_dora_fine_tuning(self):
        backbone = GemmaBackbone(**self._init_kwargs)
        backbone.enable_dora(4)
        # 4 target layers, 3 trainable DoRA weights per layer
        # (lora_kernel_a, lora_kernel_b, dora_magnitude).
        self.assertLen(backbone.trainable_weights, 4 * 3)
        self.assertLen(backbone.non_trainable_weights, 20)

        targets = np.random.normal(size=(2, 5, self._init_kwargs["hidden_dim"]))
        backbone.compile(optimizer="sgd", loss="mse")
        backbone.fit(self.input_data, targets, epochs=1)
        # Training should move the magnitudes (direct gradient path), unlike
        # plain LoRA where the zero-initialized `B` keeps the update tiny.
        for layer in backbone._flatten_layers(include_self=False):
            if getattr(layer, "dora_enabled", False):
                mag = keras.ops.convert_to_numpy(layer.dora_magnitude)
                kernel = keras.ops.convert_to_numpy(layer._kernel)
                reduce_axes = tuple(range(kernel.ndim - 1))
                initial = np.sqrt((kernel**2).sum(axis=reduce_axes))
                self.assertGreater(float(np.abs(mag - initial).max()), 0.0)

    def test_dora_fine_tuning_target_names(self):
        backbone = GemmaBackbone(**self._init_kwargs)
        backbone.enable_dora(4, target_layer_names=["query"])
        self.assertLen(backbone.trainable_weights, 2 * 3)
        self.assertLen(backbone.non_trainable_weights, 20)

        targets = np.random.normal(size=(2, 5, self._init_kwargs["hidden_dim"]))
        backbone.compile(optimizer="sgd", loss="mse")
        backbone.fit(self.input_data, targets, epochs=1)

    def test_dora_matches_base_model_at_initialization(self):
        # DoRA initializes B to zeros (so the direction update is zero) and
        # the magnitude to column norms of the pretrained kernel, so the
        # effective kernel is exactly the original kernel. The model output
        # should therefore be unchanged immediately after `enable_dora`.
        backbone = GemmaBackbone(**self._init_kwargs)
        before = keras.ops.convert_to_numpy(backbone(self.input_data))
        backbone.enable_dora(4)
        after = keras.ops.convert_to_numpy(backbone(self.input_data))
        self.assertAllClose(before, after, atol=1e-4)

    def test_dora_saving_and_reloading(self):
        backbone = GemmaBackbone(**self._init_kwargs)
        initial_weights_path = os.path.join(
            self.get_temp_dir(), "base.weights.h5"
        )
        backbone.save_weights(initial_weights_path)

        backbone.enable_dora(4)
        targets = np.random.normal(size=(2, 5, self._init_kwargs["hidden_dim"]))
        backbone.compile(optimizer="sgd", loss="mse")
        backbone.fit(self.input_data, targets, epochs=1)

        dora_filepath = os.path.join(self.get_temp_dir(), "dora.dora.h5")
        backbone.save_dora_weights(dora_filepath)

        new_backbone = GemmaBackbone(**self._init_kwargs)
        new_backbone.load_weights(initial_weights_path)
        new_backbone.enable_dora(4)
        new_backbone.load_dora_weights(dora_filepath)

        ref_out = backbone(self.input_data)
        new_out = new_backbone(self.input_data)
        self.assertAllClose(ref_out, new_out)

        # Error paths.
        backbone = GemmaBackbone(**self._init_kwargs)
        with self.assertRaisesRegex(ValueError, "no dora-enabled layers"):
            backbone.save_dora_weights(dora_filepath)
        backbone.enable_dora(5)
        with self.assertRaisesRegex(ValueError, "ranks must match"):
            backbone.load_dora_weights(dora_filepath)
        with self.assertRaisesRegex(ValueError, "filename must end in"):
            backbone.save_dora_weights("bad_filepath")
