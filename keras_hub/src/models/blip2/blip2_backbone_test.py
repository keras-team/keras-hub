"""Tests for BLIP-2 backbone."""

import numpy as np
import pytest
import keras

from keras_hub.src.models.blip2.blip2_backbone import Blip2Backbone
from keras_hub.src.models.blip2.blip2_custom_opt import Blip2CustomOPT
from keras_hub.src.models.blip2.blip2_qformer import Blip2QFormer
from keras_hub.src.models.blip2.blip2_vision_encoder import Blip2VisionEncoder
from keras_hub.src.tests.test_case import TestCase


class Blip2BackboneTest(TestCase):
    def setUp(self):
        self.batch_size = 2
        self.image_size = 32
        self.seq_length = 5
        self.num_query_tokens = 2
        self.hidden_dim = 4

        vision_encoder = Blip2VisionEncoder(
            image_size=self.image_size,
            patch_size=4,
            num_layers=2,
            num_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            use_patch_bias=True,
            use_class_token=True,
            use_mha_bias=True,
            use_mlp_bias=True,
            dropout_rate=0.0,
            layer_norm_epsilon=1e-6,
            initializer_range=0.02,
            dtype="float32",
        )
        qformer = Blip2QFormer(
            num_query_tokens=self.num_query_tokens,
            num_layers=2,
            num_heads=2,
            hidden_dim=self.hidden_dim,
            intermediate_dim=8,
            vision_dim=8,
            cross_attention_frequency=1,
            dropout=0.0,
            layer_norm_epsilon=1e-6,
            dtype="float32",
        )
        language_model = Blip2CustomOPT(
            vocabulary_size=14,
            num_layers=2,
            num_heads=2,
            hidden_dim=self.hidden_dim,
            intermediate_dim=8,
            num_query_tokens=self.num_query_tokens,
            qformer_hidden_dim=self.hidden_dim,
            max_sequence_length=10,
            dropout=0.0,
            language_projection=None,
            dtype="float32",
        )

        self.init_kwargs = {
            "vision_encoder": vision_encoder,
            "qformer": qformer,
            "language_model": language_model,
        }
        self.input_data = {
            "images": np.ones(
                (self.batch_size, self.image_size, self.image_size, 3),
                dtype="float32",
            ),
            "token_ids": np.ones(
                (self.batch_size, self.seq_length), dtype="int32"
            ),
            "padding_mask": np.ones(
                (self.batch_size, self.seq_length), dtype="bool"
            ),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=Blip2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(
                self.batch_size,
                self.num_query_tokens + self.seq_length,
                self.hidden_dim,
            ),
            variable_length_data=[self.input_data],
            run_quantization_check=False,
            run_mixed_precision_check=False,
        )

    def test_architecture_characteristics(self):
        backbone = Blip2Backbone(**self.init_kwargs)
        self.assertEqual(backbone.num_query_tokens, self.num_query_tokens)
        self.assertEqual(backbone.qformer.hidden_dim, self.hidden_dim)
        self.assertEqual(backbone.language_model.hidden_dim, self.hidden_dim)

    def test_saved_model(self):
        # Mirror Gemma3 pattern — no atol/rtol overrides.
        # dtype="float32" is pinned on all sub-models in setUp so the
        # save/load round-trip is numerically stable on CPU backends.
        self.run_model_saving_test(
            cls=Blip2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=Blip2Backbone,
            preset="blip2_base",
            input_data=self.input_data,
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Blip2Backbone.presets:
            self.run_preset_test(
                cls=Blip2Backbone,
                preset=preset,
                input_data=self.input_data,
            )