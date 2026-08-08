import numpy as np

from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_converter import (
    Qwen3ASRAudioConverter,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_causal_lm_preprocessor import (
    Qwen3ASRCausalLMPreprocessor,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_tokenizer import Qwen3ASRTokenizer
from keras_hub.src.tests.test_case import TestCase


class Qwen3ASRCausalLMPreprocessorTest(TestCase):
    def setUp(self):
        self._allow_python_workflow = True

        # Dummy Vocab
        self.merges = ["! !", "! a", "a b"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += ["<|im_end|>", "<|endoftext|>", "!", "<|AUDIO|>"]
        self.vocab = sorted(set(self.vocab))
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])

        self.tokenizer = Qwen3ASRTokenizer(
            vocabulary=self.vocab, merges=self.merges
        )

        # Audio Converter configured to yield exactly 100 frames (1 chunk)
        self.audio_converter = Qwen3ASRAudioConverter(
            max_audio_length=1,  # 1 second
            sampling_rate=16000,
            stride=160,
        )

        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "audio_converter": self.audio_converter,
            "sequence_length": 32,  # Needs to be large enough for expansion
            "_allow_python_workflow": self._allow_python_workflow,
        }

        self.input_data = {
            "prompts": ["! <|AUDIO|>"],
            "responses": ["a"],
            "audio": [np.zeros([16000])],
        }

    def test_preprocessor_basics(self):
        preprocessor = Qwen3ASRCausalLMPreprocessor(**self.init_kwargs)
        outputs = preprocessor(self.input_data)

        # Verify outputs contain expected keys
        x = outputs[0]
        self.assertIn("token_ids", x)
        self.assertIn("audio_mel", x)
        self.assertIn("audio_indices", x)

        # Verify audio_indices shape matches expanded placeholders (13 for 1
        # chunk).
        self.assertEqual(x["audio_indices"].shape[-1], 13)

        self.assertEqual(x["token_ids"].shape[-1], 32)
