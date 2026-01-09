import os
import numpy as np
import keras
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.models.modernbert.modernbert_tokenizer import ModernBertTokenizer


class ModernBertTokenizerTest(TestCase):
    def setUp(self):
        self.vocab = {
            "<|endoftext|>": 0,
            "<|padding|>": 1,
            "t": 2,
            "h": 3,
            "e": 4,
            "th": 5,
            "the": 6,
            "<|file_separator|>": 50279,
            "<mask>": 50284,
        }
        self.merges = ["t h", "th e"]
        self.tokenizer = ModernBertTokenizer(
            vocabulary=self.vocab, 
            merges=self.merges
        )

    def test_tokenize(self):
        output = self.tokenizer.tokenize("the")
        self.assertAllEqual(output, [6])

    def test_serialization(self):
        config = self.tokenizer.get_config()
        new_tokenizer = ModernBertTokenizer.from_config(config)
        
        input_str = "the"
        self.assertAllEqual(
            self.tokenizer.tokenize(input_str),
            new_tokenizer.tokenize(input_str)
        )

    def test_save_and_load(self):
        input_data = keras.Input(shape=(), dtype="string", name="input_str")
        output = self.tokenizer(input_data)
        model = keras.Model(input_data, output)
        
        path = os.path.join(self.get_temp_dir(), "tokenizer_model.keras")
        model.save(path)

        reloaded_model = keras.models.load_model(path)
 
        test_input = np.array(["the"], dtype=object)
        
        original_output = model(test_input)
        reloaded_output = reloaded_model(test_input)
        
        self.assertAllEqual(original_output, reloaded_output)