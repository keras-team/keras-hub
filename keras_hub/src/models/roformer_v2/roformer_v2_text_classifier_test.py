import keras

from keras_hub.src.models.roformer_v2 import (
    roformer_v2_text_classifier_preprocessor as r,
)
from keras_hub.src.models.roformer_v2.roformer_v2_backbone import (
    RoformerV2Backbone,
)
from keras_hub.src.models.roformer_v2.roformer_v2_text_classifier import (
    RoformerV2TextClassifier,
)
from keras_hub.src.models.roformer_v2.roformer_v2_tokenizer import (
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
        if keras.__version__ < "3.6":
            self.skipTest("Failing on keras lower version")
        self.run_task_test(
            cls=RoformerV2TextClassifier,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 2),
        )
