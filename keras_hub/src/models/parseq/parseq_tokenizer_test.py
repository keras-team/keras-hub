import os

import keras
from keras.src.saving import serialization_lib

from keras_hub.src.models.parseq.parseq_tokenizer import PARSEQ_VOCAB
from keras_hub.src.models.parseq.parseq_tokenizer import PARSeqTokenizer
from keras_hub.src.tests.test_case import TestCase


class PARSeqTokenizerTest(TestCase):
    def setUp(self):
        self.tokenizer = PARSeqTokenizer()
        self.input_data = ["Google", "Keras"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=PARSeqTokenizer,
            init_kwargs={},
            input_data=self.input_data,
            expected_output=[
                [43, 25, 25, 17, 22, 15],
                [47, 15, 28, 11, 29],
            ],
            expected_detokenize_output=[
                ["G", "o", "o", "g", "l", "e"],
                ["K", "e", "r", "a", "s"],
            ],
        )

    def test_safe_mode_vocabulary_file_disallowed(self):
        temp_dir = self.get_temp_dir()
        vocab_path = os.path.join(temp_dir, "vocabulary.txt")
        with open(vocab_path, "w", encoding="utf-8") as file:
            for token in PARSEQ_VOCAB:
                file.write(f"{token}\n")

        with serialization_lib.SafeModeScope(True):
            with self.assertRaisesRegex(
                ValueError,
                r"Requested the loading of a vocabulary file outside of the "
                r"model archive.*Vocabulary file: .*vocabulary\.txt",
            ):
                PARSeqTokenizer(vocabulary=vocab_path)

            tokenizer = PARSeqTokenizer()
            with self.assertRaisesRegex(
                ValueError,
                r"Requested the loading of a vocabulary file outside of the "
                r"model archive.*Vocabulary file: .*vocabulary\.txt",
            ):
                tokenizer.set_vocabulary(vocab_path)

        serialized = keras.saving.serialize_keras_object(PARSeqTokenizer())
        serialized["config"]["vocabulary"] = vocab_path
        with self.assertRaisesRegex(
            TypeError,
            r"Requested the loading of a vocabulary file outside of the "
            r"model archive.*Vocabulary file: .*vocabulary\.txt",
        ):
            keras.saving.deserialize_keras_object(serialized)

        tokenizer = PARSeqTokenizer(vocabulary=vocab_path)
        self.assertAllEqual(
            tokenizer(self.input_data),
            [[43, 25, 25, 17, 22, 15], [47, 15, 28, 11, 29]],
        )

        with serialization_lib.SafeModeScope(False):
            tokenizer = PARSeqTokenizer(vocabulary=vocab_path)
            self.assertAllEqual(
                tokenizer(self.input_data),
                [[43, 25, 25, 17, 22, 15], [47, 15, 28, 11, 29]],
            )
            restored = keras.saving.deserialize_keras_object(
                serialized, safe_mode=False
            )
            self.assertAllEqual(
                restored(self.input_data),
                [[43, 25, 25, 17, 22, 15], [47, 15, 28, 11, 29]],
            )
