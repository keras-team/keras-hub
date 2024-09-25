import os

import pytest

from keras_hub.src.models.f_net.f_net_backbone import FNetBackbone
from keras_hub.src.models.f_net.f_net_masked_lm import FNetMaskedLM
from keras_hub.src.models.f_net.f_net_masked_lm_preprocessor import (
    FNetMaskedLMPreprocessor,
)
from keras_hub.src.models.f_net.f_net_tokenizer import FNetTokenizer
from keras_hub.src.tests.test_case import TestCase


class FNetMaskedLMTest(TestCase):
    def setUp(self):
        # Setup model.
        self.preprocessor = FNetMaskedLMPreprocessor(
            FNetTokenizer(
                # Generated using create_f_net_test_proto.py
                proto=os.path.join(
                    self.get_test_data_dir(), "f_net_test_vocab.spm"
                )
            ),
            # Simplify our testing by masking every available token.
            mask_selection_rate=1.0,
            mask_token_rate=1.0,
            random_token_rate=0.0,
            mask_selection_length=5,
            sequence_length=5,
        )
        self.backbone = FNetBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            hidden_dim=2,
            intermediate_dim=4,
            max_sequence_length=self.preprocessor.sequence_length,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = (
            ["the quick brown fox.", "the slow brown fox."],  # Features.
        )
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_masked_lm_basics(self):
        self.run_task_test(
            cls=FNetMaskedLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 5, 12),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=FNetMaskedLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in FNetMaskedLM.presets:
            self.run_preset_test(
                cls=FNetMaskedLM,
                preset=preset,
                input_data=self.input_data,
            )
