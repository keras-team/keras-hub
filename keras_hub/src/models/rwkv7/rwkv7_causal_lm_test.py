from unittest.mock import patch

from keras import ops

from keras_hub.src.models.rwkv7.rwkv7_backbone import RWKV7Backbone
from keras_hub.src.models.rwkv7.rwkv7_causal_lm import RWKV7CausalLM
from keras_hub.src.models.rwkv7.rwkv7_causal_lm_preprocessor import (
    RWKV7CausalLMPreprocessor,
)
from keras_hub.src.models.rwkv7.rwkv7_tokenizer import RWKVTokenizer
from keras_hub.src.tests.test_case import TestCase


class RWKV7CausalLMTest(TestCase):
    def setUp(self):
        """
        Set up the test case with vocabulary, merges, preprocessor, backbone,
        and other initialization parameters.
        """
        # Create a small vocabulary for testing
        self.vocab = [
            "1 ' ' 1",
            "2 '\\n' 1",
            "3 'the' 3",
            "4 'hello' 5",
            "5 'world' 5",
            "6 'python' 6",
        ]

        # Initialize tokenizer with test vocabulary
        self.tokenizer = RWKVTokenizer(vocabulary=self.vocab, end_token_id=1)

        # Create preprocessor with sequence length of 8
        self.preprocessor = RWKV7CausalLMPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=16,
        )

        # Create a small backbone for testing
        self.backbone = RWKV7Backbone(
            vocabulary_size=7,
            hidden_size=16,
            num_layers=2,
            head_size=4,
            intermediate_dim=32,
            gate_lora=8,
            mv_lora=4,
            aaa_lora=4,
            decay_lora=4,
        )

        # Initialize parameters for the causal LM
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = (["hello world", "the python"],)
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_generate(self):
        """
        Test text generation functionality.
        """
        causal_lm = RWKV7CausalLM(self.backbone, self.preprocessor)
        prompt = ["hello world"]
        output = causal_lm.generate(prompt, 16)
        self.assertTrue(isinstance(output[0], str))
        self.assertTrue(isinstance(output, list))

        prompt = "hello world"
        output = causal_lm.generate(prompt, 16)
        self.assertTrue(isinstance(output, str))

    def test_generate_strip_prompt(self):
        """
        Test that generated text can strip the prompt from output.
        """
        prompt = ["hello world"]
        causal_lm = RWKV7CausalLM(self.backbone, self.preprocessor)
        output = causal_lm.generate(prompt, 16, strip_prompt=True)
        self.assertFalse(output[0].startswith(prompt[0]))

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=RWKV7CausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 16, 7),
        )

    def test_generate_compilation(self):
        causal_lm = RWKV7CausalLM(**self.init_kwargs)
        causal_lm.generate("hello world", max_length=16)
        first_fn = causal_lm.generate_function
        causal_lm.generate("hello world", max_length=16)
        second_fn = causal_lm.generate_function
        self.assertEqual(first_fn, second_fn)

        causal_lm.compile(sampler="greedy")
        self.assertIsNone(causal_lm.generate_function)

    def test_early_stopping(self):
        causal_lm = RWKV7CausalLM(**self.init_kwargs)
        call_with_cache = causal_lm.call_with_cache

        def wrapper(*args, **kwargs):
            logits, hidden_states, cache = call_with_cache(*args, **kwargs)
            if logits is not None:
                index = self.preprocessor.tokenizer.end_token_id
                update = ops.ones_like(logits)[:, :, index] * 1.0e9
                update = ops.expand_dims(update, axis=-1)
                logits = ops.slice_update(logits, (0, 0, index), update)
            return logits, hidden_states, cache

        with patch.object(causal_lm, "call_with_cache", wraps=wrapper):
            prompt = ["hello world", "the python"]
            output = causal_lm.generate(prompt, max_length=16)
            except_output = [t + " " for t in prompt]
            self.assertEqual(except_output, output)
