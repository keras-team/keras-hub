import os

import pytest

from keras_hub.src.models.xlm_roberta.xlm_roberta_backbone import (
    XLMRobertaBackbone,
)
from keras_hub.src.models.xlm_roberta.xlm_roberta_text_classifier import (
    XLMRobertaTextClassifier,
)
from keras_hub.src.models.xlm_roberta.xlm_roberta_text_classifier_preprocessor import (  # noqa: E501
    XLMRobertaTextClassifierPreprocessor,
)
from keras_hub.src.models.xlm_roberta.xlm_roberta_tokenizer import (
    XLMRobertaTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class XLMRobertaTextClassifierTest(TestCase):
    def setUp(self):
        # Setup model.
        self.preprocessor = XLMRobertaTextClassifierPreprocessor(
            XLMRobertaTokenizer(
                # Generated using create_xlm_roberta_test_proto.py
                proto=os.path.join(
                    self.get_test_data_dir(), "xlm_roberta_test_vocab.spm"
                )
            ),
            sequence_length=5,
        )
        self.backbone = XLMRobertaBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            num_heads=2,
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
            cls=XLMRobertaTextClassifier,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 2),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=XLMRobertaTextClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=XLMRobertaTextClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in XLMRobertaTextClassifier.presets:
            self.run_preset_test(
                cls=XLMRobertaTextClassifier,
                preset=preset,
                init_kwargs={"num_classes": 2},
                input_data=self.input_data,
                expected_output_shape=(2, 2),
            )
