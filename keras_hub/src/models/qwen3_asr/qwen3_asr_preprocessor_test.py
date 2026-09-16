import numpy as np

from keras_hub.src.models.qwen3.qwen3_tokenizer import Qwen3Tokenizer
from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_converter import (
    Qwen3ASRAudioConverter,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_preprocessor import (
    Qwen3ASRPreprocessor,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3ASRPreprocessorTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += [
            "<|audio_pad|>",
            "<|audio_info|>",
            "<|im_end|>",
            "<|endoftext|>",
        ]
        self.vocab = sorted(set(self.vocab))
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.tokenizer = Qwen3Tokenizer(
            vocabulary=self.vocab,
            merges=self.merges,
        )

        self.audio_converter = Qwen3ASRAudioConverter(
            max_audio_length=1.05,
        )

        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "audio_converter": self.audio_converter,
            "sequence_length": 40,
        }

    def test_preprocessor_basics(self):
        input_data = {
            "prompts": [" airplane at airport"],
            "responses": [" airplane at airport"],
            "audio": [np.ones((16000,))],
        }
        preprocessor = Qwen3ASRPreprocessor(**self.init_kwargs)
        output = preprocessor(input_data)
        x, y, sw = output

        # Check keys
        self.assertIn("token_ids", x)
        self.assertIn("padding_mask", x)
        self.assertIn("audio_mel", x)
        self.assertIn("audio_mask", x)

        # Check shapes
        self.assertEqual(x["token_ids"].shape, (1, 40))
        self.assertEqual(x["padding_mask"].shape, (1, 40))
        self.assertEqual(len(x["audio_mel"].shape), 3)
        self.assertEqual(x["audio_mel"].shape[0], 1)
        self.assertEqual(x["audio_mel"].shape[2], 128)

        self.assertEqual(len(x["audio_mask"].shape), 2)
        self.assertEqual(x["audio_mask"].shape[0], 1)

    def test_inference_basics(self):
        input_data = {
            "prompts": [" airplane at airport"],
            "audio": [np.ones((16000,))],
        }
        preprocessor = Qwen3ASRPreprocessor(**self.init_kwargs)
        output = preprocessor(input_data)

        # Check keys
        self.assertIn("token_ids", output)
        self.assertIn("padding_mask", output)
        self.assertIn("audio_mel", output)
        self.assertIn("audio_mask", output)

        # Check shapes
        self.assertEqual(output["token_ids"].shape, (1, 40))
        self.assertEqual(output["padding_mask"].shape, (1, 40))
        self.assertEqual(len(output["audio_mel"].shape), 3)
        self.assertEqual(output["audio_mel"].shape[0], 1)
        self.assertEqual(output["audio_mel"].shape[2], 128)

        self.assertEqual(len(output["audio_mask"].shape), 2)
        self.assertEqual(output["audio_mask"].shape[0], 1)

    def test_generate_preprocess(self):
        input_data = {
            "prompts": " airplane",
            "audio": np.ones((16000,)),
        }
        preprocessor = Qwen3ASRPreprocessor(**self.init_kwargs)
        output = preprocessor.generate_preprocess(input_data)

        self.assertIn("token_ids", output)
        self.assertIn("padding_mask", output)
        self.assertIn("audio_mel", output)
        self.assertIn("audio_mask", output)

        # Output should be unbatched if input was unbatched
        self.assertEqual(output["token_ids"].shape, (40,))
        self.assertEqual(output["padding_mask"].shape, (40,))
        self.assertEqual(len(output["audio_mel"].shape), 2)
        self.assertEqual(output["audio_mel"].shape[1], 128)
        self.assertEqual(len(output["audio_mask"].shape), 1)

    def test_generate_postprocess(self):
        input_data = {
            "token_ids": [5, 17, 27, 26, 19, 1, 1, 1],
            "padding_mask": [1, 1, 1, 1, 1, 0, 0, 0],
        }
        preprocessor = Qwen3ASRPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_postprocess(input_data)
        # It should decode to string
        self.assertTrue(isinstance(x, str) or isinstance(x, list))
