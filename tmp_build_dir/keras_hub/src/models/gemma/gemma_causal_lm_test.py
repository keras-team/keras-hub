import os
from unittest.mock import patch

import keras
import pytest
from keras import ops

from keras_hub.src.models.gemma.gemma_backbone import GemmaBackbone
from keras_hub.src.models.gemma.gemma_causal_lm import GemmaCausalLM
from keras_hub.src.models.gemma.gemma_causal_lm_preprocessor import (
    GemmaCausalLMPreprocessor,
)
from keras_hub.src.models.gemma.gemma_tokenizer import GemmaTokenizer
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.keras_utils import fused_attention_op_available
from keras_hub.src.utils.keras_utils import gpu_supports_fused_attention_op
from keras_hub.src.utils.keras_utils import running_on_gpu


class GemmaCausalLMTest(TestCase):
    def setUp(self):
        self.tokenizer = GemmaTokenizer(
            proto=os.path.join(
                self.get_test_data_dir(), "gemma_test_vocab.spm"
            ),
        )
        self.preprocessor = GemmaCausalLMPreprocessor(
            self.tokenizer,
            sequence_length=8,
        )
        # Test Gemma 2 like config, as it's the more complicated case.
        self.backbone = GemmaBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            head_dim=2,
            sliding_window_size=3,
            use_sliding_window_attention=True,
            attention_logit_soft_cap=50,
            final_logit_soft_cap=30,
            query_head_dim_normalize=False,
            use_post_ffw_norm=True,
            use_post_attention_norm=True,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = (["the quick brown fox", "the quick brown fox"],)
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=GemmaCausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 8, 11),
        )

    def test_cache_correctness(self):
        token_ids = self.input_data["token_ids"]
        padding_mask = ops.ones_like(self.input_data["padding_mask"])
        causal_lm = GemmaCausalLM(**self.init_kwargs)
        full_logits = causal_lm(
            {"token_ids": token_ids, "padding_mask": padding_mask}
        )
        token_ids = self.input_data["token_ids"]
        _, cache = causal_lm._build_cache(token_ids)
        cache = ops.zeros_like(cache)
        cached_logits = []
        for i in range(self.preprocessor.sequence_length):
            sliced = token_ids[:, i][:, None]
            logits, _, cache = causal_lm.call_with_cache(sliced, cache, i)
            cached_logits.append(logits)
        cached_logits = ops.concatenate(cached_logits, 1)
        self.assertAllClose(full_logits, cached_logits, atol=0.002)

    def test_generate(self):
        causal_lm = GemmaCausalLM(**self.init_kwargs)
        # String input.
        prompt = "the quick brown fox"
        output = causal_lm.generate("the quick brown fox")
        self.assertTrue(prompt in output)
        # Int tensor input.
        prompt_ids = self.preprocessor.generate_preprocess([prompt])
        causal_lm.preprocessor = None
        outputs = causal_lm.generate(prompt_ids, stop_token_ids=None)
        # Assert prompt is in output in token id space.
        self.assertAllEqual(
            outputs["token_ids"][:, :4],
            prompt_ids["token_ids"][:, :4],
        )
        self.assertAllEqual(
            outputs["padding_mask"][:, :4],
            prompt_ids["padding_mask"][:, :4],
        )

    def test_flash_attention_call(self):
        if (
            keras.config.backend() != "jax"
            or not fused_attention_op_available()
            or not gpu_supports_fused_attention_op()
        ):
            self.skipTest("`flash_attention` testing requires the Jax backend.")

        with patch("keras.src.backend.nn.dot_product_attention") as mock_func:
            causal_lm = GemmaCausalLM(**self.init_kwargs)
            causal_lm.generate("the quick brown fox")
            if running_on_gpu():
                mock_func.assert_called()
            else:
                mock_func.assert_not_called()

    def test_generate_with_bfloat16(self):
        original_floatx = keras.config.floatx()
        keras.config.set_floatx("float16")
        try:
            causal_lm = GemmaCausalLM(**self.init_kwargs)
            # String input.
            prompt = "the quick brown fox"
            output = causal_lm.generate("the quick brown fox")
            self.assertTrue(prompt in output)
            # Int tensor input.
            prompt_ids = self.preprocessor.generate_preprocess([prompt])
            causal_lm.preprocessor = None
            outputs = causal_lm.generate(prompt_ids, stop_token_ids=None)
            # Assert prompt is in output in token id space.
            self.assertAllEqual(
                outputs["token_ids"][:, :4],
                prompt_ids["token_ids"][:, :4],
            )
            self.assertAllEqual(
                outputs["padding_mask"][:, :4],
                prompt_ids["padding_mask"][:, :4],
            )
        finally:
            # Restore floatx to the original value to prevent impact on other
            # tests even if there is an exception.
            keras.config.set_floatx(original_floatx)

    def test_early_stopping(self):
        causal_lm = GemmaCausalLM(**self.init_kwargs)
        call_with_cache = causal_lm.call_with_cache

        def wrapper(*args, **kwargs):
            """Modify output logits to always favor end_token_id"""
            logits, hidden_states, cache = call_with_cache(*args, **kwargs)
            index = self.preprocessor.tokenizer.end_token_id
            update = ops.ones_like(logits)[:, :, index] * 1.0e9
            update = ops.expand_dims(update, axis=-1)
            logits = ops.slice_update(logits, (0, 0, index), update)
            return logits, hidden_states, cache

        with patch.object(causal_lm, "call_with_cache", wraps=wrapper):
            prompt = ["the quick brown fox", "the quick"]
            output = causal_lm.generate(prompt)
            # We should immediately abort and output the prompt.
            self.assertEqual(prompt, output)

    def test_multitoken_stopping(self):
        causal_lm = GemmaCausalLM(**self.init_kwargs)
        call_with_cache = causal_lm.call_with_cache

        def wrapper(*args, **kwargs):
            """Modify output logits to always favor end_token_id"""
            logits, hidden_states, cache = call_with_cache(*args, **kwargs)
            index = self.preprocessor.tokenizer.end_token_id
            update = ops.ones_like(logits)[:, :, index] * 1.0e9
            update = ops.expand_dims(update, axis=-1)
            logits = ops.slice_update(logits, (0, 0, index), update)
            return logits, hidden_states, cache

        with patch.object(causal_lm, "call_with_cache", wraps=wrapper):
            prompt = ["the quick brown fox", "the quick"]

            output = causal_lm.generate(prompt, stop_token_ids=(3,))
            # We should immediately abort and output the prompt.
            self.assertEqual(prompt, output)

    def test_generate_compilation(self):
        causal_lm = GemmaCausalLM(**self.init_kwargs)
        # Assert we do not recompile with successive calls.
        causal_lm.generate("the quick brown fox")
        first_fn = causal_lm.generate_function
        causal_lm.generate("the quick brown fox")
        second_fn = causal_lm.generate_function
        self.assertEqual(first_fn, second_fn)
        # Assert we do recompile after compile is called.
        causal_lm.compile(sampler="greedy")
        self.assertIsNone(causal_lm.generate_function)

    @pytest.mark.kaggle_key_required
    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=GemmaCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        """Test LiteRT export for GemmaCausalLM with small test model."""
        model = GemmaCausalLM(**self.init_kwargs)

        # Convert boolean padding_mask to int32 for LiteRT compatibility
        input_data = self.input_data.copy()
        if "padding_mask" in input_data:
            input_data["padding_mask"] = ops.cast(
                input_data["padding_mask"], "int32"
            )

        expected_output_shape = (
            2,
            8,
            self.preprocessor.tokenizer.vocabulary_size(),
        )

        self.run_litert_export_test(
            model=model,
            input_data=input_data,
            expected_output_shape=expected_output_shape,
            comparison_mode="statistical",
            output_thresholds={"*": {"max": 1e-3, "mean": 1e-5}},
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in GemmaCausalLM.presets:
            self.run_preset_test(
                cls=GemmaCausalLM,
                preset=preset,
                input_data=self.input_data,
            )

    def test_score_logits(self):
        # Setup prompts, models, and associated expected shapes.
        prompts = ["the quick brown fox", "the quick brown fox"]
        causal_lm = GemmaCausalLM(**self.init_kwargs)
        expected_score_shape = (2, 8, 11)

        # Preprocess prompts to get tokenized representations and padding masks.
        preprocessed_prompts = causal_lm.preprocessor.generate_preprocess(
            prompts
        )
        token_ids = preprocessed_prompts["token_ids"]
        padding_mask = preprocessed_prompts["padding_mask"]

        # Get the scores and assert their shape.
        scores = causal_lm.score(
            token_ids=token_ids,
            padding_mask=padding_mask,
            scoring_mode="logits",
        )

        self.assertEqual(ops.shape(scores), expected_score_shape)

    def test_score_loss(self):
        # Setup prompts, models, and associated expected shapes.
        prompts = ["the quick brown fox", "the quick brown fox"]
        causal_lm = GemmaCausalLM(**self.init_kwargs)
        expected_score_shape = (2, 8)

        # Preprocess prompts to get tokenized representations and padding masks.
        preprocessed_prompts = causal_lm.preprocessor.generate_preprocess(
            prompts
        )
        token_ids = preprocessed_prompts["token_ids"]
        padding_mask = preprocessed_prompts["padding_mask"]
        target_ids = keras.ops.roll(token_ids, shift=-1, axis=1)

        # Get the scores and assert their shape.
        scores = causal_lm.score(
            token_ids=token_ids,
            padding_mask=padding_mask,
            scoring_mode="loss",
            target_ids=target_ids,
        )

        self.assertEqual(ops.shape(scores), expected_score_shape)

    def test_score_layer_intercept_fn_exfiltration(self):
        # Setup prompts, models, and associated expected shapes.
        prompts = ["the quick brown fox", "the quick brown fox"]
        causal_lm = GemmaCausalLM(**self.init_kwargs)
        expected_embedded_shape = (2, 8, 8)
        expected_score_shape = (2, 8, 11)

        # Preprocess prompts to get tokenized representations and padding masks.
        preprocessed_prompts = causal_lm.preprocessor.generate_preprocess(
            prompts
        )
        token_ids = preprocessed_prompts["token_ids"]
        padding_mask = preprocessed_prompts["padding_mask"]

        # Setup a custom intercept function that extracts the embeddings to a
        # a variable from the embeddings layer and otherwise asserts on shapes.
        embedded_prompts = None

        def layer_intercept_fn_for_testing(x, i):
            if i == -1:
                nonlocal embedded_prompts
                embedded_prompts = x
            else:
                nonlocal expected_embedded_shape
                self.assertEqual(ops.shape(x), expected_embedded_shape)
            return x

        # Get the scores.
        scores = causal_lm.score(
            token_ids=token_ids,
            padding_mask=padding_mask,
            scoring_mode="logits",
            layer_intercept_fn=layer_intercept_fn_for_testing,
        )

        # Assert shapes for info exfiltrated into the parent context.
        self.assertEqual(ops.shape(embedded_prompts), expected_embedded_shape)
        self.assertEqual(ops.shape(scores), expected_score_shape)

    def test_get_quantization_layer_structure(self):
        causal_lm = GemmaCausalLM(**self.init_kwargs)
        structure = causal_lm.get_quantization_layer_structure("gptq")
        self.assertIsInstance(structure, dict)
        self.assertIn("pre_block_layers", structure)
        self.assertIn("sequential_blocks", structure)
        self.assertLen(structure["pre_block_layers"], 1)
        self.assertIsInstance(structure["pre_block_layers"][0], keras.Model)
        self.assertEqual(
            structure["sequential_blocks"], self.backbone.transformer_layers
        )

        self.assertIsNone(causal_lm.get_quantization_layer_structure("int8"))
