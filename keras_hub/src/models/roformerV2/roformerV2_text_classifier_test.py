from keras_hub.src.models.roformerV2 import (
    roformerV2_text_classifier_preprocessor as r,
)
from keras_hub.src.models.roformerV2.roformerV2_backbone import (
    RoformerV2Backbone,
)
from keras_hub.src.models.roformerV2.roformerV2_text_classifier import (
    RorformerV2TextClassifier,
)
from keras_hub.src.models.roformerV2.roformerV2_tokenizer import (
    RoformerV2Tokenizer,
)
from keras_hub.src.tests.test_case import TestCase

RoformerV2TextClassifierPreprocessor = r.RoformerV2TextClassifierPreprocessor


class RoformerVTextClassifierTest(TestCase):
    def setUp(self):
        # Setup model.
        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.vocab += ["the", "quick", "brown", "fox", "."]
        self.preprocessor = RoformerV2TextClassifierPreprocessor(
            RoformerV2Tokenizer(vocabulary=self.vocab),
            sequence_length=5,
        )
        self.backbone = RoformerV2Backbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            num_heads=2,
            hidden_dim=4,
            intermediate_dim=8,
            head_size=2,
            max_wavelength=self.preprocessor.sequence_length,
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
            cls=RorformerV2TextClassifier,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 2),
        )
