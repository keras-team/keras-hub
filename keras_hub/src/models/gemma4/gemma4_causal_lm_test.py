import copy
from unittest.mock import patch

import keras
import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone
from keras_hub.src.models.gemma4.gemma4_causal_lm import Gemma4CausalLM
from keras_hub.src.models.gemma4.gemma4_causal_lm_preprocessor import (
    Gemma4CausalLMPreprocessor,
)
from keras_hub.src.models.gemma4.gemma4_image_converter import (
    Gemma4ImageConverter,
)
from keras_hub.src.models.gemma4.gemma4_vision_encoder import (
    Gemma4VisionEncoder,
)
from keras_hub.src.tests.mocks.mock_gemma4_tokenizer import MockGemma4Tokenizer
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.keras_utils import fused_attention_op_available
from keras_hub.src.utils.keras_utils import gpu_supports_fused_attention_op
from keras_hub.src.utils.keras_utils import running_on_gpu


class Gemma4CausalLMTest(TestCase, parameterized.TestCase):
    def setUp(self):
        self.tokenizer = MockGemma4Tokenizer()

        # === Text model ===
        self.text_preprocessor = Gemma4CausalLMPreprocessor(
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
            "image_size": 16,
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

        self.text_backbone = Gemma4Backbone(**text_backbone_init_kwargs)
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

        # === Vision + Text model ===
        self.image_converter = Gemma4ImageConverter(
            image_size=(16, 16),
            patch_size=4,
        )
        self.preprocessor = Gemma4CausalLMPreprocessor(
            image_converter=self.image_converter,
            tokenizer=self.tokenizer,
            sequence_length=20,
            max_images_per_prompt=2,
            num_vision_tokens_per_image=4,
        )

        vision_encoder = Gemma4VisionEncoder(
            image_size=16,
            patch_size=4,
            pool_size=2,
            num_layers=2,
            num_heads=2,
            head_dim=4,
            num_key_value_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            output_dim=8,
        )
        backbone_init_kwargs = copy.deepcopy(text_backbone_init_kwargs)
        backbone_init_kwargs["vision_encoder"] = vision_encoder
        self.backbone = Gemma4Backbone(**backbone_init_kwargs)
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }

        self.train_data = (
            {
                "prompts": tf.constant(
                    [
                        "the quick brown fox <|image>",
                        "the quick brown fox",
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
            cls=Gemma4CausalLM,
            init_kwargs=init_kwargs,
            train_data=train_data,
            expected_output_shape=(2, 20, self.tokenizer.vocabulary_size()),
        )

    def test_text_flash_attention_call(self):
        if (
            keras.config.backend() != "jax"
            or not fused_attention_op_available()
            or not gpu_supports_fused_attention_op()
        ):
            self.skipTest("`flash_attention` testing requires the JAX backend.")

        with patch(
            "keras.src.backend.nn.dot_product_attention",
            wraps=keras.src.backend.nn.dot_product_attention,
        ) as mock_func:
            causal_lm = Gemma4CausalLM(**self.text_init_kwargs)
            causal_lm.generate("the quick brown fox")
            if running_on_gpu():
                mock_func.assert_called()
            else:
                mock_func.assert_not_called()

    def test_text_early_stopping(self):
        causal_lm = Gemma4CausalLM(**self.text_init_kwargs)
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

    def test_text_multitoken_stopping(self):
        causal_lm = Gemma4CausalLM(**self.text_init_kwargs)
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
            output = causal_lm.generate(prompt, stop_token_ids=(3,))
            self.assertEqual(prompt, output)

    def test_text_generate_compilation(self):
        causal_lm = Gemma4CausalLM(**self.text_init_kwargs)
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

        self.run_model_saving_test(
            cls=Gemma4CausalLM,
            init_kwargs=init_kwargs,
            input_data=input_data,
        )

    def test_litert_export(self):
        """Test LiteRT export for Gemma4CausalLM with small text model."""
        model = Gemma4CausalLM(**self.text_init_kwargs)

        input_data = self.text_input_data.copy()
        if "padding_mask" in input_data:
            input_data["padding_mask"] = ops.cast(
                input_data["padding_mask"], "int32"
            )

        expected_output_shape = (
            2,
            20,
            self.text_preprocessor.tokenizer.vocabulary_size(),
        )

        self.run_litert_export_test(
            model=model,
            input_data=input_data,
            expected_output_shape=expected_output_shape,
            comparison_mode="statistical",
            output_thresholds={"*": {"max": 1e-2, "mean": 1e-4}},
        )

    @pytest.mark.large
    def test_litert_export_multimodal(self):
        """Test LiteRT export for multimodal Gemma4CausalLM."""
        model = Gemma4CausalLM(**self.init_kwargs)

        input_data = self.input_data.copy()
        if "padding_mask" in input_data:
            input_data["padding_mask"] = ops.cast(
                input_data["padding_mask"], "int32"
            )

        expected_output_shape = (
            2,
            20,
            self.preprocessor.tokenizer.vocabulary_size(),
        )

        self.run_litert_export_test(
            model=model,
            input_data=input_data,
            expected_output_shape=expected_output_shape,
            comparison_mode="statistical",
            output_thresholds={"*": {"max": 1e-2, "mean": 1e-4}},
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Gemma4CausalLM.presets:
            self.run_preset_test(
                cls=Gemma4CausalLM,
                preset=preset,
                input_data=self.text_input_data,
            )
