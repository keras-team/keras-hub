import os

import pytest

from keras_hub.src.models.f_net.f_net_backbone import FNetBackbone
from keras_hub.src.models.f_net.f_net_text_classifier import FNetTextClassifier
from keras_hub.src.models.f_net.f_net_text_classifier_preprocessor import (
    FNetTextClassifierPreprocessor,
)
from keras_hub.src.models.f_net.f_net_tokenizer import FNetTokenizer
from keras_hub.src.tests.test_case import TestCase


class FNetTextClassifierTest(TestCase):
    def setUp(self):
        # Setup model.
        self.preprocessor = FNetTextClassifierPreprocessor(
            FNetTokenizer(
                # Generated using create_f_net_test_proto.py
                proto=os.path.join(
                    self.get_test_data_dir(), "f_net_test_vocab.spm"
                )
            ),
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
            "num_classes": 2,
        }
        self.train_data = (
            ["the quick brown fox.", "the slow brown fox."],  # Features.
            [1, 0],  # Labels.
        )
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_classifier_basics(self):
        self.run_task_test(
            cls=FNetTextClassifier,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 2),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=FNetTextClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        # F-Net does NOT use padding_mask - it only uses token_ids and
        # segment_ids. Don't add padding_mask to input_data.
        self.run_litert_export_test(
            cls=FNetTextClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            comparison_mode="statistical",
            output_thresholds={
                "*": {"max": 0.01, "mean": 0.005},
            },
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in FNetTextClassifier.presets:
            self.run_preset_test(
                cls=FNetTextClassifier,
                preset=preset,
                init_kwargs={"num_classes": 2},
                input_data=self.input_data,
                expected_output_shape=(2, 2),
            )
