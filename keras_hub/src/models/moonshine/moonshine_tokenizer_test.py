import os

from keras_hub.src.models.moonshine.moonshine_tokenizer import (
    MoonshineTokenizer,
)


class MoonshineTokenizerTest:
    def setUp(self):
        self.init_kwargs = {
            # Generated using create_moonshine_test_proto.py
            "proto": os.path.join(
                self.get_test_data_dir(), "moonshine_test_vocab.spm"
            )
        }
        self.input_data = [
            "Hello world!",
            "Test with <<ST_42>>",
            "Hex test <0x1F>",
        ]

    def test_tokenizer_basics(self):
        tokenizer = MoonshineTokenizer(**self.init_kwargs)
        outputs = tokenizer(self.input_data)
        self.assertIsNotNone(outputs)

    def test_special_tokens(self):
        tokenizer = MoonshineTokenizer(**self.init_kwargs)
        st_token = "<<ST_42>>"
        output = tokenizer(st_token)
        self.assertIsNotNone(output)
        hex_token = "<0x1F>"
        output = tokenizer(hex_token)
        self.assertIsNotNone(output)
