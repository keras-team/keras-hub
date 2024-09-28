import os

import pytest

from keras_hub.src.models.albert.albert_backbone import AlbertBackbone
from keras_hub.src.models.albert.albert_masked_lm import AlbertMaskedLM
from keras_hub.src.models.albert.albert_masked_lm_preprocessor import (
    AlbertMaskedLMPreprocessor,
)
from keras_hub.src.models.albert.albert_tokenizer import AlbertTokenizer
from keras_hub.src.tests.test_case import TestCase


class AlbertMaskedLMTest(TestCase):
    def setUp(self):
        # Setup model.
        self.preprocessor = AlbertMaskedLMPreprocessor(
            AlbertTokenizer(
                # Generated using create_albert_test_proto.py
                proto=os.path.join(
                    self.get_test_data_dir(), "albert_test_vocab.spm"
                ),
                sequence_length=5,
            ),
            # Simplify our testing by masking every available token.
            mask_selection_rate=1.0,
            mask_token_rate=1.0,
            random_token_rate=0.0,
            mask_selection_length=5,
            sequence_length=5,
        )
        self.backbone = AlbertBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            num_heads=2,
            hidden_dim=2,
            embedding_dim=2,
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
            cls=AlbertMaskedLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 5, 12),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=AlbertMaskedLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in AlbertMaskedLM.presets:
            self.run_preset_test(
                cls=AlbertMaskedLM,
                preset=preset,
                input_data=self.input_data,
            )
