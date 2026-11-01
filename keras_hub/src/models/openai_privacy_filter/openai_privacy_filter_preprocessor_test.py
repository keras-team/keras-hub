import keras
import numpy as np
import pytest

from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_preprocessor import (  # noqa: E501
    OpenAIPrivacyFilterPreprocessor,
)
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_tokenizer import (  # noqa: E501
    OpenAIPrivacyFilterTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class OpenAIPrivacyFilterPreprocessorTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += ["!", "<|endoftext|>"]
        self.vocab = sorted(set(self.vocab))
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.tokenizer = OpenAIPrivacyFilterTokenizer(
            vocabulary=self.vocab, merges=self.merges
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 8,
        }
        self.input_data = (
            ["airplane at airport"],
            [[1, 2, 0, 0, 0]],  # Per-token labels, padded to 8 by the layer.
            [[1.0, 1.0, 1.0, 1.0, 1.0]],  # Padded to 8 by the layer.
        )

    def test_preprocessor_basics(self):
        self.run_preprocessor_test(
            cls=OpenAIPrivacyFilterPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_sample_weight_handling(self):
        # An explicit `sample_weight` is aligned with the packed `y`: per-token
        # weights are padded or truncated to `sequence_length` (floats kept),
        # a per-sample weight is broadcast and masked. Each batched case runs
        # on both the Python and the TF workflow via `run_preprocessor_test`.
        pad = self.tokenizer.pad_token_id
        ids = [4, 16, 26, 25, 18, pad, pad, pad]
        mask = [True, True, True, True, True, False, False, False]
        labels = [1, 2, 0, 0, 0, 0, 0, 0]
        padded_half = [1.0, 0.5] + [0.0] * 6
        cases = [
            # (name, sample_weight, expected sample_weight)
            ("numpy_2d", np.array([[1.0, 0.5]]), [padded_half]),
            ("ragged_2d", [[1.0, 0.5]], [padded_half]),
            ("truncated", [[1.0] * 10], [[1.0] * 8]),
            ("per_sample", [2.0], [[2.0] * 5 + [0.0] * 3]),
        ]
        for name, sw, expected_sw in cases:
            with self.subTest(name):
                self.run_preprocessor_test(
                    cls=OpenAIPrivacyFilterPreprocessor,
                    init_kwargs=self.init_kwargs,
                    input_data=(
                        ["airplane at airport"],
                        [[1, 2, 0, 0, 0]],
                        sw,
                    ),
                    expected_output=(
                        {"token_ids": [ids], "padding_mask": [mask]},
                        [labels],
                        expected_sw,
                    ),
                )
        # Unbatched: `tf.data.Dataset.from_tensor_slices` cannot slice a
        # scalar string, so check the Python call and a `from_tensors`
        # pipeline directly.
        import tensorflow as tf

        preprocessor = OpenAIPrivacyFilterPreprocessor(**self.init_kwargs)
        sample = ("airplane at airport", [1, 2, 0, 0, 0], [1.0, 0.5])
        expected = (
            {"token_ids": ids, "padding_mask": mask},
            labels,
            padded_half,
        )
        self.assertAllClose(preprocessor(*sample), expected)
        ds = tf.data.Dataset.from_tensors(sample).map(preprocessor)
        self.assertAllClose(ds.get_single_element(), expected)

    def test_output_structure(self):
        preprocessor = OpenAIPrivacyFilterPreprocessor(**self.init_kwargs)
        output = preprocessor(["airplane at airport"])
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(output)
        self.assertIn("token_ids", x)
        self.assertIn("padding_mask", x)

    def test_sequence_length_override(self):
        preprocessor = OpenAIPrivacyFilterPreprocessor(**self.init_kwargs)
        output = preprocessor(["airplane at airport"], sequence_length=16)
        x, _, _ = keras.utils.unpack_x_y_sample_weight(output)
        from keras import ops

        self.assertEqual(ops.shape(x["token_ids"])[-1], 16)

    def test_no_special_tokens(self):
        preprocessor = OpenAIPrivacyFilterPreprocessor(**self.init_kwargs)
        x = preprocessor(["airplane at airport"])
        pad = self.tokenizer.pad_token_id
        self.assertAllEqual(
            x["token_ids"], [[4, 16, 26, 25, 18, pad, pad, pad]]
        )
        self.assertAllEqual(
            x["padding_mask"],
            [[True, True, True, True, True, False, False, False]],
        )

    def test_get_config(self):
        preprocessor = OpenAIPrivacyFilterPreprocessor(**self.init_kwargs)
        config = preprocessor.get_config()
        self.assertEqual(config["sequence_length"], 8)
        self.assertEqual(config["truncate"], "round_robin")

    def test_backend_tensor_sample_weight(self):
        # A backend tensor passed as sample_weight on the Python path
        # should be converted safely.
        preprocessor = OpenAIPrivacyFilterPreprocessor(**self.init_kwargs)
        sw = keras.ops.convert_to_tensor([1.0, 0.5])
        if keras.backend.backend() == "torch":
            sw = sw.requires_grad_()
        sample = ("airplane at airport", [1, 2, 0, 0, 0], sw)
        # Check it does not raise.
        x, y, sample_weight = preprocessor(*sample)
        self.assertEqual(list(sample_weight), [1.0, 0.5] + [0.0] * 6)

    def test_multi_segment_raises(self):
        preprocessor = OpenAIPrivacyFilterPreprocessor(**self.init_kwargs)
        # Multi-segment input is a tuple of batches, one per segment. It has
        # to be passed as `x=`: a positional 2-tuple is unpacked as `(x, y)`.
        with self.assertRaisesRegex(ValueError, "Multi-segment"):
            preprocessor(x=(["airplane at airport"], ["airport"]))

    def test_short_y_raises(self):
        # A `y` row with fewer labels than its tokens would otherwise be
        # padded with label 0 at weight 1. Both workflows refuse it.
        import tensorflow as tf

        preprocessor = OpenAIPrivacyFilterPreprocessor(**self.init_kwargs)
        short = "shorter than its unpadded token count"
        with self.assertRaisesRegex(ValueError, short):
            preprocessor(["airplane at airport"], [[1]])
        with self.assertRaisesRegex(ValueError, short):
            preprocessor("airplane at airport", [1])
        with self.assertRaisesRegex(ValueError, short):
            preprocessor(["airplane at airport", "at"], [[1, 2, 0, 0, 0], [1]])

        ds = tf.data.Dataset.from_tensors(("airplane at airport", [1]))
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError, short):
            ds.map(preprocessor).get_single_element()
        ds = tf.data.Dataset.from_tensor_slices(
            (["airplane at airport", "at"], tf.ragged.constant([[1], [1]]))
        ).batch(2)
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError, short):
            list(ds.map(preprocessor))

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in OpenAIPrivacyFilterPreprocessor.presets:
            self.run_preset_test(
                cls=OpenAIPrivacyFilterPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
