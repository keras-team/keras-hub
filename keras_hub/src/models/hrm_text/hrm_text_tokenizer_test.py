from keras_hub.src.models.hrm_text.hrm_text_tokenizer import HrmTextTokenizer
from keras_hub.src.tests.test_case import TestCase


def make_tokenizer_assets():
    merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
    merges += [
        "Ġa t",
        "p o",
        "r t",
        "Ġt h",
        "ai r",
        "pl a",
        "po rt",
    ]
    merges += ["Ġai r", "Ġa i", "pla ne"]
    vocabulary = []
    for merge in merges:
        left, right = merge.split(" ")
        vocabulary.extend([left, right, left + right])
    vocabulary += ["!", "<|im_start|>", "<|box_end|>", "<|endoftext|>"]
    vocabulary = {
        token: index for index, token in enumerate(sorted(set(vocabulary)))
    }
    return vocabulary, merges


class HrmTextTokenizerTest(TestCase):
    def setUp(self):
        self.vocabulary, self.merges = make_tokenizer_assets()
        self.tokenizer = HrmTextTokenizer(
            vocabulary=self.vocabulary, merges=self.merges
        )

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=HrmTextTokenizer,
            init_kwargs={
                "vocabulary": self.vocabulary,
                "merges": self.merges,
            },
            input_data=[" airplane", " at airport"],
            expected_detokenize_output=[" airplane", " at airport"],
        )

    def test_tokenize_and_detokenize(self):
        token_ids = self.tokenizer([" airplane", " at airport"])
        self.assertAllEqual(token_ids, [[27, 18], [28, 27, 20]])
        self.assertAllEqual(
            self.tokenizer.detokenize(token_ids),
            [" airplane", " at airport"],
        )

    def test_special_tokens(self):
        self.assertEqual(self.tokenizer.start_token, "<|im_start|>")
        self.assertEqual(self.tokenizer.end_token, "<|box_end|>")
        self.assertEqual(self.tokenizer.pad_token, "<|endoftext|>")
        self.assertEqual(
            self.tokenizer.start_token_id,
            self.vocabulary["<|im_start|>"],
        )
        self.assertEqual(
            self.tokenizer.end_token_id, self.vocabulary["<|box_end|>"]
        )
        self.assertEqual(
            self.tokenizer.pad_token_id, self.vocabulary["<|endoftext|>"]
        )

    def test_config_round_trip(self):
        config = self.tokenizer.get_config()
        restored = HrmTextTokenizer.from_config(config)
        restored.set_vocabulary_and_merges(self.vocabulary, self.merges)
        self.assertAllEqual(
            self.tokenizer([" airplane"]), restored([" airplane"])
        )
