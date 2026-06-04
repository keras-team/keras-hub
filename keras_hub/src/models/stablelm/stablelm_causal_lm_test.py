from unittest.mock import patch

import pytest
from keras import ops

from keras_hub.src.models.stablelm.stablelm_backbone import StableLMBackbone
from keras_hub.src.models.stablelm.stablelm_causal_lm import StableLMCausalLM
from keras_hub.src.models.stablelm.stablelm_causal_lm_preprocessor import (
    StableLMCausalLMPreprocessor,
)
from keras_hub.src.models.stablelm.stablelm_tokenizer import StableLMTokenizer
from keras_hub.src.tests.test_case import TestCase


class StableLMCausalLMTest(TestCase):
    def setUp(self):
        self.vocab = [
            "!",
            "air",
            "Ġair",
            "plane",
            "Ġat",
            "port",
            "<|endoftext|>",
        ]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.merges = [
            "Ġ a",
            "Ġ t",
            "Ġ i",
            "Ġ b",
            "a i",
            "p l",
            "n e",
            "Ġa t",
            "p o",
            "r t",
            "Ġt h",
            "ai r",
            "pl a",
            "po rt",
            "Ġai r",
            "Ġa i",
            "pla ne",
        ]

        self.preprocessor = StableLMCausalLMPreprocessor(
            tokenizer=StableLMTokenizer(
                vocabulary=self.vocab, merges=self.merges
            ),
            sequence_length=8,
        )

        # Config
        self.backbone = StableLMBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            num_query_heads=2,
            num_key_value_heads=2,
            hidden_dim=4,
            intermediate_dim=8,
        )

        # Initialization kwargs for the causal LM.
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }

        # Training data for testing.
        self.train_data = ([" airplane at airport", " airplane at airport"],)
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=StableLMCausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 8, 7),
        )

    def test_generate(self):
        causal_lm = StableLMCausalLM(**self.init_kwargs)
        # Test string input.
        prompt = " airplane at airport"
        output = causal_lm.generate(prompt)
        self.assertTrue(prompt in output)
        # Test integer tensor input.
        prompt_ids = self.preprocessor.generate_preprocess([prompt])
        causal_lm.preprocessor = None
        outputs = causal_lm.generate(prompt_ids, stop_token_ids=None)
        # Assert prompt is preserved in output token IDs.
        self.assertAllEqual(
            outputs["token_ids"][:, :5],
            prompt_ids["token_ids"][:, :5],
        )
        self.assertAllEqual(
            outputs["padding_mask"][:, :5],
            prompt_ids["padding_mask"][:, :5],
        )

    def test_early_stopping(self):
        causal_lm = StableLMCausalLM(**self.init_kwargs)
        call_with_cache = causal_lm.call_with_cache

        def wrapper(*args, **kwargs):
            """Modify logits to favor end_token_id for early stopping."""
            logits, hidden_states, cache = call_with_cache(*args, **kwargs)
            index = self.preprocessor.tokenizer.end_token_id
            update = ops.ones_like(logits)[:, :, index] * 1.0e9
            update = ops.expand_dims(update, axis=-1)
            logits = ops.slice_update(logits, (0, 0, index), update)
            return logits, hidden_states, cache

        with patch.object(causal_lm, "call_with_cache", wraps=wrapper):
            prompt = [" airplane at airport", " airplane"]
            output = causal_lm.generate(prompt)
            # We should immediately abort and output the prompt.
            self.assertEqual(prompt, output)

    def test_generate_compilation(self):
        causal_lm = StableLMCausalLM(**self.init_kwargs)
        # Assert we do not recompile with successive calls.
        causal_lm.generate(" airplane at airport")
        first_fn = causal_lm.generate_function
        causal_lm.generate(" airplane at airport")
        second_fn = causal_lm.generate_function
        self.assertEqual(first_fn, second_fn)
        # Assert we do recompile after compile is called.
        causal_lm.compile(sampler="greedy")
        self.assertIsNone(causal_lm.generate_function)

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=StableLMCausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
