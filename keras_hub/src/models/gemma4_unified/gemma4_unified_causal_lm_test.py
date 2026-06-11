import copy
import os
from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.gemma4.gemma4_audio_encoder import Gemma4AudioEncoder
from keras_hub.src.models.gemma4_unified.gemma4_unified_audio_converter import (
    Gemma4UnifiedAudioConverter,
)
from keras_hub.src.models.gemma4_unified.gemma4_unified_backbone import (
    Gemma4UnifiedBackbone,
)
from keras_hub.src.models.gemma4_unified.gemma4_unified_causal_lm import (
    Gemma4UnifiedCausalLM,
)
from keras_hub.src.models.gemma4_unified.gemma4_unified_causal_lm_preprocessor import (  # noqa: E501
    Gemma4UnifiedCausalLMPreprocessor,
)
from keras_hub.src.models.gemma4_unified.gemma4_unified_image_converter import (
    Gemma4UnifiedImageConverter,
)
from keras_hub.src.models.gemma4_unified.gemma4_unified_vision_embedder import (
    Gemma4UnifiedVisionEmbedder,
)
from keras_hub.src.tests.mocks.mock_gemma4_tokenizer import MockGemma4Tokenizer
from keras_hub.src.tests.test_case import TestCase


class Gemma4UnifiedCausalLMTest(TestCase, parameterized.TestCase):
    def setUp(self):
        self.tokenizer = MockGemma4Tokenizer()

        # === Text model ===
        self.text_preprocessor = Gemma4UnifiedCausalLMPreprocessor(
            image_converter=None,
            tokenizer=self.tokenizer,
            sequence_length=20,
            max_images_per_prompt=0,
            num_vision_tokens_per_image=0,
        )

        text_backbone_init_kwargs = {
            "vocabulary_size": (
                self.text_preprocessor.tokenizer.vocabulary_size()
            ),
            "image_size": None,
            "num_layers": 6,
            "num_query_heads": 2,
            "num_key_value_heads": 1,
            "hidden_dim": 8,
            "intermediate_dim": 16,
            "head_dim": 4,
            "use_sliding_window_attention": True,
            "sliding_window_size": 16,
            "attention_logit_soft_cap": None,
            "final_logit_soft_cap": None,
            "vision_encoder": None,
        }

        self.text_backbone = Gemma4UnifiedBackbone(**text_backbone_init_kwargs)
        self.text_init_kwargs = {
            "preprocessor": self.text_preprocessor,
            "backbone": self.text_backbone,
        }
        self.text_train_data = (
            {
                "prompts": ["the quick brown fox", "the quick brown fox"],
                "responses": ["the earth is round", "the earth is round"],
            },
        )
        self.text_input_data = self.text_preprocessor(*self.text_train_data)[0]

        # === Multimodal model (Vision + Audio + Text) ===
        self.image_converter = Gemma4UnifiedImageConverter(
            image_size=(16, 16),
            patch_size=4,
        )
        self.mock_audio_converter = Gemma4UnifiedAudioConverter(
            num_mels=8,
            num_fft_bins=8,
            frame_length=8,
            max_audio_length=1,
            stride=2,
            sampling_rate=100,
        )
        self.preprocessor = Gemma4UnifiedCausalLMPreprocessor(
            image_converter=self.image_converter,
            audio_converter=self.mock_audio_converter,
            tokenizer=self.tokenizer,
            sequence_length=20,
            max_images_per_prompt=2,
            num_vision_tokens_per_image=4,
            audio_input_feat_size=8,
        )

        vision_embedder = Gemma4UnifiedVisionEmbedder(
            hidden_dim=8,
            model_patch_size=4,
            mm_posemb_size=4,
            num_soft_tokens=4,
        )
        mock_audio_encoder = Gemma4AudioEncoder(
            input_feat_size=8,
            hidden_size=8,
            num_heads=2,
            num_layers=2,
            output_dim=8,
        )
        backbone_init_kwargs = copy.deepcopy(text_backbone_init_kwargs)
        backbone_init_kwargs["vision_encoder"] = vision_embedder
        backbone_init_kwargs["audio_encoder"] = mock_audio_encoder
        self.backbone = Gemma4UnifiedBackbone(**backbone_init_kwargs)
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }

        self.train_data = (
            {
                "prompts": tf.constant(
                    [
                        "the quick brown fox <|image> <|audio|>",
                        "the quick brown fox <|audio|>",
                    ]
                ),
                "responses": tf.constant(
                    ["the earth is round", "the earth is round"]
                ),
                "pixel_values": tf.constant(
                    np.ones([2, 2, 16, 3 * 4 * 4], dtype="float32")
                ),
                "pixel_position_ids": tf.constant(
                    np.ones([2, 2, 16, 2], dtype="int32")
                ),
                "audio_mel": tf.constant(
                    np.ones((2, 1, 50, 8), dtype="float32")
                ),
                "audio_mel_mask": tf.constant(
                    np.ones((2, 1, 50), dtype="int32")
                ),
            },
        )
        self.input_data = self.preprocessor(*self.train_data)[0]

    @parameterized.named_parameters(
        ("text_and_vision", "text_and_vision"), ("text_only", "text_only")
    )
    def test_causal_lm_basics(self, modality_type):
        if modality_type == "text_and_vision":
            init_kwargs = self.init_kwargs
            train_data = self.train_data
        else:
            init_kwargs = self.text_init_kwargs
            train_data = self.text_train_data

        self.run_task_test(
            cls=Gemma4UnifiedCausalLM,
            init_kwargs=init_kwargs,
            train_data=train_data,
            expected_output_shape=(2, 20, self.tokenizer.vocabulary_size()),
        )

    def test_text_early_stopping(self):
        causal_lm = Gemma4UnifiedCausalLM(**self.text_init_kwargs)
        call_with_cache = causal_lm.call_with_cache

        def wrapper(*args, **kwargs):
            """Modify output logits to always favour end_token_id."""
            logits, hidden_states, cache = call_with_cache(*args, **kwargs)
            index = self.text_preprocessor.tokenizer.end_token_id
            update = ops.ones_like(logits)[:, :, index] * 1.0e9
            update = ops.expand_dims(update, axis=-1)
            logits = ops.slice_update(logits, (0, 0, index), update)
            return logits, hidden_states, cache

        with patch.object(causal_lm, "call_with_cache", wraps=wrapper):
            prompt = ["the quick brown fox", "the quick"]
            output = causal_lm.generate(prompt)
            self.assertEqual(prompt, output)

    def test_text_generate_compilation(self):
        causal_lm = Gemma4UnifiedCausalLM(**self.text_init_kwargs)
        # Assert we do not recompile with successive calls.
        causal_lm.generate("the quick brown fox")
        first_fn = causal_lm.generate_function
        causal_lm.generate("the quick brown fox")
        second_fn = causal_lm.generate_function
        self.assertEqual(first_fn, second_fn)
        # Assert we do recompile after compile is called.
        causal_lm.compile(sampler="greedy")
        self.assertIsNone(causal_lm.generate_function)

    @parameterized.named_parameters(
        ("text_and_vision", "text_and_vision"), ("text_only", "text_only")
    )
    def test_saved_model(self, modality_type):
        if modality_type == "text_and_vision":
            init_kwargs = self.init_kwargs
            input_data = self.input_data
        else:
            init_kwargs = self.text_init_kwargs
            input_data = self.text_input_data

        model = Gemma4UnifiedCausalLM(**init_kwargs)
        model_output = model(input_data)

        path = os.path.join(self.get_temp_dir(), "model.weights.h5")
        model.save_weights(path)

        restored_model = Gemma4UnifiedCausalLM(**init_kwargs)
        _ = restored_model(input_data)
        restored_model.load_weights(path)

        self.assertEqual(len(model.weights), len(restored_model.weights))
        weights = model.get_weights()
        restored_weights = restored_model.get_weights()
        for w1, w2 in zip(weights, restored_weights):
            self.assertAllClose(w1, w2, atol=1e-5, rtol=1e-5)

        restored_output = restored_model(input_data)
        self.assertAllClose(model_output, restored_output, atol=1e-5, rtol=1e-5)

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Gemma4UnifiedCausalLM.presets:
            self.run_preset_test(
                cls=Gemma4UnifiedCausalLM,
                preset=preset,
                input_data=self.text_input_data,
            )
