import os

from keras.src.saving import serialization_lib

from keras_hub.src.models.parseq.parseq_tokenizer import PARSeqTokenizer
from keras_hub.src.tests.test_case import TestCase


class PARSeqTokenizerTest(TestCase):
    def test_safe_mode_vocabulary_file_disallowed(self):
        temp_dir = self.get_temp_dir()
        vocab_path = os.path.join(temp_dir, "vocab.txt")
        with open(vocab_path, "w") as file:
            file.write("a\nb\nc\n")

        tokenizer = PARSeqTokenizer()
        with serialization_lib.SafeModeScope(True):
            with self.assertRaisesRegex(
                ValueError,
                r"Requested the loading of a vocabulary file outside of the "
                r"model archive.*Vocabulary file: .*vocab\.txt",
            ):
                tokenizer.set_vocabulary(vocab_path)
