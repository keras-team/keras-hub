import keras
from keras_hub.src.models.stablelm.stablelm_tokenizer import StableLMTokenizer
from keras_hub.src.tests.test_case import TestCase

class StableLMTokenizerTest(TestCase):
    def setUp(self):
       
        self.vocab = ["!", "air", "Ġair", "plane", "Ġat", "port", "<|endoftext|>"]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.merges = [
            "Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e",
            "Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt",
            "Ġai r", "Ġa i", "pla ne"
        ]
        self.init_kwargs = {"vocabulary": self.vocab, "merges": self.merges}
        self.input_data = [
            " airplane at airport<|endoftext|>",
            " airplane airport",
        ]

    def test_tokenizer_basics(self):
        expected_output = [[2, 3, 4, 2, 5, 6],[2, 3, 2, 5]]
        
        # Run the preprocessing layer test to verify tokenization
        self.run_preprocessing_layer_test(
            cls=StableLMTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=expected_output,
        )

    def test_errors_missing_special_tokens(self):
        # Test that an error is raised if "<|endoftext|>" is missing from the vocabulary
        with self.assertRaises(ValueError):
            StableLMTokenizer(vocabulary=["a", "b", "c"], merges=[])