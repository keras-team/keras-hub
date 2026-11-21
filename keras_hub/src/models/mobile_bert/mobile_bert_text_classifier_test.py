import numpy as np
import pytest

from keras_hub.src.models.mobile_bert.mobile_bert_backbone import (
    MobileBertBackbone,
)
from keras_hub.src.models.mobile_bert.mobile_bert_text_classifier import (
    MobileBertTextClassifier,
)
from keras_hub.src.models.mobile_bert.mobile_bert_text_classifier_preprocessor import (  # noqa: E501
    MobileBertTextClassifierPreprocessor,
)
from keras_hub.src.models.mobile_bert.mobile_bert_tokenizer import (
    MobileBertTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class MobileBertTextClassifierTest(TestCase):
    def setUp(self):
        # Setup model.
        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.vocab += ["the", "quick", "brown", "fox", "."]
        self.preprocessor = MobileBertTextClassifierPreprocessor(
            MobileBertTokenizer(vocabulary=self.vocab),
            sequence_length=5,
        )
        self.backbone = MobileBertBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            num_heads=2,
            hidden_dim=4,
            embedding_dim=2,
            intra_bottleneck_dim=2,
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
            cls=MobileBertTextClassifier,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 2),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MobileBertTextClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=MobileBertTextClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=MobileBertTextClassifier,
            preset="mobilebert_uncased",
            init_kwargs={"num_classes": 2},
            input_data=self.input_data,
            expected_output_shape=(2, 2),
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in MobileBertTextClassifier.presets:
            self.run_preset_test(
                cls=MobileBertTextClassifier,
                preset=preset,
                init_kwargs={"num_classes": 2},
                input_data=self.input_data,
                expected_output_shape=(2, 2),
            )

    def test_custom_backbone_and_default_dropout(self):
        features = ["The quick brown fox jumped.", "I forgot my homework."]
        vocab = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        vocab += ["The", "quick", "brown", "fox", "jumped", "."]
        tokenizer = MobileBertTokenizer(vocabulary=vocab)
        preprocessor = MobileBertTextClassifierPreprocessor(
            tokenizer=tokenizer,
            sequence_length=16,
        )
        backbone = MobileBertBackbone(
            vocabulary_size=len(vocab),
            num_layers=2,
            num_heads=2,
            hidden_dim=16,
            embedding_dim=8,
            intra_bottleneck_dim=8,
            intermediate_dim=16,
            max_sequence_length=16,
        )
        classifier = MobileBertTextClassifier(
            backbone=backbone,
            preprocessor=preprocessor,
            num_classes=4,
        )
        self.assertEqual(classifier.dropout, 0.0)
        output = classifier.predict(features, batch_size=2)
        self.assertEqual(output.shape, (2, 4))
        self.assertTrue(np.isfinite(output).all())
