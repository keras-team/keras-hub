"""Tests for BLIP-2 Causal LM model."""

from unittest.mock import patch

import numpy as np
import pytest

from keras_hub.src.models.blip2.blip2_backbone import Blip2Backbone
from keras_hub.src.models.blip2.blip2_causal_lm import Blip2CausalLM
from keras_hub.src.models.blip2.blip2_causal_lm_preprocessor import (
    Blip2CausalLMPreprocessor,
)
from keras_hub.src.models.blip2.blip2_custom_opt import Blip2CustomOPT
from keras_hub.src.models.blip2.blip2_image_converter import Blip2ImageConverter
from keras_hub.src.models.blip2.blip2_qformer import Blip2QFormer
from keras_hub.src.models.blip2.blip2_tokenizer import Blip2Tokenizer
from keras_hub.src.models.blip2.blip2_vision_encoder import Blip2VisionEncoder
from keras_hub.src.tests.test_case import TestCase


class Blip2CausalLMTest(TestCase):
    def setUp(self):
        vocab = {
            "<s>": 0,
            "<pad>": 1,
            "</s>": 2,
            "\u010a": 3,
            "Ġ": 4,
            "t": 5,
            "h": 6,
            "e": 7,
        }
        merges = ["Ġ t", "h e"]
        tokenizer = Blip2Tokenizer(vocabulary=vocab, merges=merges)

        # === Text model ===
        self.text_preprocessor = Blip2CausalLMPreprocessor(
            tokenizer=tokenizer,
            image_converter=None,
            sequence_length=10,
        )
        language_model = Blip2CustomOPT(
            vocabulary_size=8,
            num_layers=2,
            num_heads=2,
            hidden_dim=4,
            intermediate_dim=8,
            num_query_tokens=0,
            qformer_hidden_dim=4,
            max_sequence_length=20,
        )
        text_backbone = Blip2Backbone(
            vision_encoder=None,
            qformer=None,
            language_model=language_model,
        )
        self.text_init_kwargs = {
            "preprocessor": self.text_preprocessor,
            "backbone": text_backbone,
        }
        self.text_train_data = ({"text": ["the", "the"]},)
        self.text_input_data = self.text_preprocessor(*self.text_train_data)[0]

        # === Vision + Text model ===
        self.preprocessor = Blip2CausalLMPreprocessor(
            tokenizer=tokenizer,
            image_converter=Blip2ImageConverter(image_size=(32, 32)),
            sequence_length=10,
        )
        vision_encoder = Blip2VisionEncoder(
            image_size=32,
            patch_size=8,
            num_layers=2,
            num_heads=2,
            hidden_dim=16,
            intermediate_dim=32,
            use_patch_bias=True,
            use_class_token=True,
            use_mha_bias=True,
            use_mlp_bias=True,
            dropout_rate=0.0,
            layer_norm_epsilon=1e-5,
        )
        qformer = Blip2QFormer(
            num_query_tokens=4,
            num_layers=2,
            num_heads=2,
            hidden_dim=4,
            intermediate_dim=8,
            vision_dim=16,
        )
        language_model_v = Blip2CustomOPT(
            vocabulary_size=8,
            num_layers=2,
            num_heads=2,
            hidden_dim=4,
            intermediate_dim=8,
            num_query_tokens=4,
            qformer_hidden_dim=4,
            max_sequence_length=20,
        )
        backbone = Blip2Backbone(
            vision_encoder=vision_encoder,
            qformer=qformer,
            language_model=language_model_v,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": backbone,
        }
        self.train_data = (
            {
                "images": np.ones((2, 32, 32, 3), dtype="float32"),
                "text": ["the", "the"],
            },
        )
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=Blip2CausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 10, 8),
        )

    def test_text_causal_lm_basics(self):
        self.run_task_test(
            cls=Blip2CausalLM,
            init_kwargs=self.text_init_kwargs,
            train_data=self.text_train_data,
            expected_output_shape=(2, 10, 8),
        )

    def test_generate_compilation(self):
        causal_lm = Blip2CausalLM(**self.init_kwargs)
        # Check generate with default compile
        causal_lm.generate(self.input_data)
        # Check generate with custom sampler
        causal_lm.compile(sampler="top_k")
        causal_lm.generate(self.input_data)

    def test_early_stopping(self):
        causal_lm = Blip2CausalLM(**self.init_kwargs)

        def wrapper(token_ids, cache, cache_update_index, **kwargs):
            batch_size = token_ids.shape[0]
            logits = np.zeros((batch_size, 1, 8))
            logits[:, :, 2] = 1.0  # EOS
            return logits, token_ids, cache

        with patch.object(causal_lm, "call_with_cache", wraps=wrapper):
            prompt = {"text": ["t", "t"], "images": self.input_data["images"]}
            output = causal_lm.generate(prompt)
            self.assertEqual(["t", "t"], output)

    def test_text_early_stopping(self):
        causal_lm = Blip2CausalLM(**self.text_init_kwargs)

        def wrapper(token_ids, cache, cache_update_index, **kwargs):
            batch_size = token_ids.shape[0]
            logits = np.zeros((batch_size, 1, 8))
            logits[:, :, 2] = 1.0  # EOS
            return logits, token_ids, cache

        with patch.object(causal_lm, "call_with_cache", wraps=wrapper):
            prompt = ["t", "t"]
            output = causal_lm.generate(prompt)
            self.assertEqual(prompt, output)

    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Blip2CausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_text_saved_model(self):
        self.run_model_saving_test(
            cls=Blip2CausalLM,
            init_kwargs=self.text_init_kwargs,
            input_data=self.text_input_data,
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Blip2CausalLM.presets:
            self.run_preset_test(
                cls=Blip2CausalLM,
                preset=preset,
                input_data=self.input_data,
            )
