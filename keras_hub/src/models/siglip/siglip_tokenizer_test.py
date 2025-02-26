import os

import pytest

from keras_hub.src.models.siglip.siglip_tokenizer import SigLIPTokenizer
from keras_hub.src.tests.test_case import TestCase


class SigLIPTokenizerTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            # Generated using create_siglip_test_proto.py
            "proto": os.path.join(
                self.get_test_data_dir(), "siglip_test_vocab.spm"
            )
        }
        self.input_data = ["the quick brown fox", "the earth is round"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=SigLIPTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[4, 9, 5, 7], [4, 6, 8, 10]],
        )

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            SigLIPTokenizer(
                # Generated using create_no_special_token_proto.py
                proto=os.path.join(
                    self.get_test_data_dir(), "no_special_token_vocab.spm"
                )
            )

    @pytest.mark.large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=SigLIPTokenizer,
            preset="clip_vit_base_patch16",
            input_data=["The quick brown fox."],
            expected_output=[[651, 4320, 8426, 25341, 235265]],
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in SigLIPTokenizer.presets:
            self.run_preset_test(
                cls=SigLIPTokenizer,
                preset=preset,
                input_data=self.input_data,
            )
