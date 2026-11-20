import pytest
from keras import ops

from keras_hub.src.models.qwen3_asr.qwen3_asr_tokenizer import Qwen3ASRTokenizer
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.tensor_utils import tf


class Qwen3ASRTokenizerTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab = sorted(set(self.vocab))  # Remove duplicates
        self.vocab += [
            "<|audio_pad|>",
            "<|audio_info|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<|audio_start|>",
            "<|audio_end|>",
            "<asr_text>",
        ]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])

        self.init_kwargs = {
            "vocabulary": self.vocab,
            "merges": self.merges,
            "unsplittable_tokens": [
                "<|im_start|>",
                "<|im_end|>",
                "<|endoftext|>",
                "<|audio_start|>",
                "<|audio_end|>",
                "<|audio_pad|>",
            ],
        }
        self.input_data = [
            " airplane at airport<|endoftext|>",
            " airplane <|audio_start|><|audio_pad|><|audio_end|>",
        ]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=Qwen3ASRTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_detokenize_skip_special_tokens_keeps_asr_text(self):
        tokenizer = Qwen3ASRTokenizer(**self.init_kwargs)
        self.assertEqual(tokenizer.end_token, "<|im_end|>")
        self.assertEqual(tokenizer.pad_token, "<|endoftext|>")
        self.assertFalse(hasattr(tokenizer, "end_token2"))
        self.assertEqual(
            list(ops.convert_to_numpy(tokenizer("<asr_text>"))),
            [self.vocab["<asr_text>"]],
        )

        token_ids = [
            self.vocab["<|audio_start|>"],
            self.vocab["<|audio_pad|>"],
            self.vocab["<|audio_end|>"],
            self.vocab["<asr_text>"],
            self.vocab["<|im_end|>"],
            self.vocab["<|endoftext|>"],
        ]
        decoded = tokenizer.detokenize(token_ids, skip_special_tokens=True)
        self.assertEqual(decoded, "<asr_text>")

        none_tok = Qwen3ASRTokenizer(
            vocabulary=self.vocab,
            merges=self.merges,
            unsplittable_tokens=None,
        )
        self.assertIn("<asr_text>", none_tok.unsplittable_tokens)

    def test_detokenize_keeps_non_special_unsplittable_tokens(self):
        tokenizer = Qwen3ASRTokenizer(
            vocabulary=self.vocab,
            merges=self.merges,
            unsplittable_tokens=["<|audio_info|>"],
        )
        token_ids = [
            self.vocab["<|im_start|>"],
            self.vocab["<|audio_info|>"],
            self.vocab["<asr_text>"],
            self.vocab["<|im_end|>"],
        ]
        self.assertEqual(
            tokenizer.detokenize(token_ids, skip_special_tokens=True),
            "<|audio_info|><asr_text>",
        )

    def test_detokenize_skip_special_tokens_ragged_batch(self):
        tokenizer = Qwen3ASRTokenizer(**self.init_kwargs)
        batch = [
            tokenizer.tokenize(" airplane"),
            tokenizer.tokenize(" airplane at"),
        ]
        decoded = tokenizer.detokenize(batch, skip_special_tokens=True)
        self.assertEqual(decoded, [" airplane", " airplane at"])

        if tf is not None:
            ragged = tf.ragged.constant(
                [list(ops.convert_to_numpy(seq)) for seq in batch]
            )
            self.assertEqual(
                tokenizer.detokenize(ragged, skip_special_tokens=True),
                [" airplane", " airplane at"],
            )

    @pytest.mark.skipif(tf is None, reason="TensorFlow is not installed")
    def test_detokenize_tf_workflow_and_dataset(self):
        tokenizer = Qwen3ASRTokenizer(**self.init_kwargs)
        token_ids = [
            self.vocab["<|im_start|>"],
            self.vocab["<asr_text>"],
            self.vocab["<|im_end|>"],
        ]
        tokenizer._allow_python_workflow = False
        try:
            eager_out = tokenizer.detokenize(
                tf.constant(token_ids), skip_special_tokens=True
            )
            if tf.is_tensor(eager_out):
                eager_out = eager_out.numpy().decode("utf-8")
            self.assertEqual(eager_out, "<asr_text>")
        finally:
            tokenizer._allow_python_workflow = True

        ds = tf.data.Dataset.from_tensor_slices([token_ids]).map(
            lambda x: tokenizer.detokenize(x, skip_special_tokens=True)
        )
        results = [item.numpy().decode("utf-8") for item in ds]
        self.assertEqual(results, ["<asr_text>"])

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            Qwen3ASRTokenizer(
                vocabulary={"a": 0, "b": 1, "ab": 2}, merges=["a b"]
            )
