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
            "0 ' ' 1",
            "1 '\\n' 1",
            "2 'the' 3",
            "3 'hello' 5",
            "4 'world' 5",
            "5 'python' 6",
        ]

        # Initialize tokenizer with test vocabulary
        self.tokenizer = RWKVTokenizer(vocabulary=self.vocab)

        # Create preprocessor with sequence length of 8
        self.preprocessor = RWKV7CausalLMPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=16,
        )

        # Create a small backbone for testing
        self.backbone = RWKV7Backbone(
            vocabulary_size=5,
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

    def test_generate_compilation(self):
        """
        Test that the generate function compiles correctly and
        reuses compiled functions.
        """
        causal_lm = RWKV7CausalLM(self.backbone, self.preprocessor)
        causal_lm.generate(["hello world"], 16)
        first_fn = causal_lm.generate_function
        causal_lm.generate(["hello world"], 16)
        second_fn = causal_lm.generate_function
        self.assertEqual(first_fn, second_fn)

        causal_lm.compile(sampler="greedy")
        self.assertIsNone(causal_lm.generate_function)
