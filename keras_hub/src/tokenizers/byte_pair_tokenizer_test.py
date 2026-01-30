import keras
import tensorflow as tf
from keras.src.saving import serialization_lib

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer

VOCAB_PATH = keras.utils.get_file(
    None,
    "https://storage.googleapis.com/keras-nlp/models/roberta_base/vocab.json",
)
MERGE_PATH = keras.utils.get_file(
    None,
    "https://storage.googleapis.com/keras-nlp/models/roberta_base/merges.txt",
)


class BytePairTokenizerTest(TestCase):
    def setUp(self):
        super().setUp()
        self.tokenizer = BytePairTokenizer(
            vocabulary=VOCAB_PATH, merges=MERGE_PATH
        )

    def test_tokenize_list_input(self):
        input_data = ["brown.", "black."]
        call_output = self.tokenizer(input_data)
        tokenize_output = self.tokenizer.tokenize(input_data)
        expected = [[31876, 4], [14178, 4]]
        self.assertAllEqual(call_output, expected)
        self.assertAllEqual(tokenize_output, expected)

        input_data = tf.convert_to_tensor(["brown.", "black."])
        encoded = self.tokenizer(input_data)
        self.assertAllEqual(encoded, expected)

    def test_tokenize_string_output(self):
        input_data = ["quick brown fox.", "slow black bear."]
        tokenizer = BytePairTokenizer(
            vocabulary=VOCAB_PATH, merges=MERGE_PATH, dtype="string"
        )
        call_output = tokenizer(input_data)
        expected = [
            ["quick", "Ġbrown", "Ġfox", "."],
            ["slow", "Ġblack", "Ġbear", "."],
        ]
        self.assertAllEqual(call_output, expected)

    def test_tokenize_with_special_tokens(self):
        vocab = {"sp": 0, "s": 1, "p": 2}
        merges = ["s p"]
        tokenizer = BytePairTokenizer(
            vocabulary=vocab,
            merges=merges,
            unsplittable_tokens=["s", "p"],
        )
        output = tokenizer("sp")
        self.assertAllEqual(output, [1, 2])

        # If not setting special tokens, "sp" is one token.
        tokenizer = BytePairTokenizer(
            vocabulary=vocab,
            merges=merges,
        )
        output = tokenizer("sp")
        self.assertAllEqual(output, [0])

    def test_tokenize_prefix_space(self):
        input_data = ["brown.", "black."]
        tokenizer = BytePairTokenizer(
            vocabulary=VOCAB_PATH,
            merges=MERGE_PATH,
            dtype="string",
            add_prefix_space=True,
        )
        call_output = tokenizer(input_data)

        expected = [["Ġbrown", "."], ["Ġblack", "."]]
        self.assertAllEqual(call_output, expected)

    def test_tokenize_scalar_input(self):
        input_data = "brown."
        encoded = self.tokenizer.tokenize(input_data)
        self.assertAllEqual(encoded, [31876, 4])

    def test_detokenize_scalar_input(self):
        input_data = ["quick brown fox."]
        encoded = self.tokenizer.tokenize(input_data)
        decoded = self.tokenizer.detokenize(encoded)
        self.assertAllEqual(input_data, decoded)

    def test_detokenize_list_input(self):
        input_data = ["quick brown fox.", "slow bear"]
        encoded = self.tokenizer.tokenize(input_data)
        decoded = self.tokenizer.detokenize(encoded)
        self.assertAllEqual(input_data, decoded)

    def test_error_id_out_of_vocabulary(self):
        with self.assertRaises(ValueError):
            self.tokenizer.id_to_token(self.tokenizer.vocabulary_size())
        with self.assertRaises(ValueError):
            self.tokenizer.id_to_token(-1)

    def test_whitespace_split(self):
        input_data = "\n\n\n  s"
        encoded = self.tokenizer(input_data)
        self.assertAllEqual(encoded, [50140, 50118, 1437, 579])

        input_data = "  \n\n\ns"
        encoded = self.tokenizer(input_data)
        self.assertAllEqual(encoded, [1437, 1437, 50140, 50118, 29])

        # This is important for Llama3 which uses the \n\n sequence in chat
        # templates: \n\n must be tokenized as a single token
        input_data = "Hello\n\nHello"
        encoded = self.tokenizer(input_data)
        self.assertAllEqual(encoded, [31414, 50140, 31414])

        input_data = "Hello\n\n\n\nHello"
        encoded = self.tokenizer(input_data)
        self.assertAllEqual(encoded, [31414, 50140, 50140, 31414])

        input_data = "Hello\n\n"
        encoded = self.tokenizer(input_data)
        self.assertAllEqual(encoded, [31414, 50140])

        input_data = "Hello\n\n\n\n"
        encoded = self.tokenizer(input_data)
        self.assertAllEqual(encoded, [31414, 50140, 50140])

    def test_special_whitespace(self):
        input_data = "\xa0 \xa0 \x3000 s"
        encoded = self.tokenizer(input_data)
        self.assertAllEqual(encoded, [50141, 50143, 12096, 579])

    def test_cjk_input(self):
        input_data = "素晴らしい！芭比Q啦～"
        # Black formats long list by one element per line, which is bad to read.
        expected = [36714, 20024, 21402, 37127, 27, 20024, 48945, 47918]
        expected += [47780, 43251, 4394, 10172, 36484, 27969, 12410, 37127]
        expected += [10965, 10674, 1864, 42393, 15722, 18164, 43251, 10809]
        expected += [17772]
        encoded = self.tokenizer(input_data)
        self.assertAllEqual(encoded, expected)

    def test_tokenize_with_tf_data(self):
        data = [
            "I am just a test string",
            "I am also a test string",
            "I am still a test string",
            "me too",
            "I am not a test string (joking)",
            "You guys should add punctuation!",
            "Period matters!",
        ]
        ds = tf.data.Dataset.from_tensor_slices(data)
        ds = ds.batch(2).map(self.tokenizer)
        encoded = next(iter(ds))
        expected = [
            [100, 524, 95, 10, 1296, 6755],
            [100, 524, 67, 10, 1296, 6755],
        ]
        self.assertAllEqual(encoded, expected)

    def test_config(self):
        input_data = ["the quick brown whale."]
        cloned_tokenizer = BytePairTokenizer.from_config(
            self.tokenizer.get_config()
        )
        cloned_tokenizer.set_vocabulary_and_merges(
            self.tokenizer.vocabulary, self.tokenizer.merges
        )
        self.assertAllEqual(
            self.tokenizer(input_data),
            cloned_tokenizer(input_data),
        )

    def test_safe_mode_vocabulary_file_disallowed(self):
        import os

        temp_dir = self.get_temp_dir()
        vocab_path = os.path.join(temp_dir, "vocab.json")
        merges_path = os.path.join(temp_dir, "merges.txt")

        with open(vocab_path, "w") as file:
            file.write('{"<|endoftext|>": 0, "the": 1, "quick": 2}')
        with open(merges_path, "w") as file:
            file.write("t h\nthe quick")

        tokenizer = BytePairTokenizer()
        with serialization_lib.SafeModeScope(True):
            with self.assertRaisesRegex(
                ValueError,
                r"Requested the loading of a vocabulary file outside of the "
                r"model archive.*Vocabulary file: .*vocab\.json",
            ):
                tokenizer.set_vocabulary_and_merges(vocab_path, merges_path)
