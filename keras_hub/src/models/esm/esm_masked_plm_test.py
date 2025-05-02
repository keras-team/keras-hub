import keras

from keras_hub.src.models.esm.esm_backbone import (
    ESMBackbone,
)
from keras_hub.src.models.esm.esm_masked_plm import (
    ESMMaskedPLM
)
from keras_hub.src.models.esm.esm_masked_plm_preprocessor import (
    ESMMaskedPLMPreprocessor,
)
from keras_hub.src.models.esm.esm_tokenizer import (
    ESMTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class ESMMaskedLMTest(TestCase):
    def setUp(self):
        # Setup model.
        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.vocab += ["the", "quick", "brown", "fox", "."]
        self.preprocessor = ESMMaskedPLMPreprocessor(
            ESMTokenizer(vocabulary=self.vocab),
            # Simplify our testing by masking every available token.
            mask_selection_rate=1.0,
            mask_token_rate=1.0,
            random_token_rate=0.0,
            mask_selection_length=5,
            sequence_length=5,
        )
        self.backbone = ESMBackbone(
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
        }
        self.train_data = (
            ["the quick brown fox.", "the slow brown fox."],  # Features.
        )
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_masked_lm_basics(self):
        if keras.__version__ < "3.6":
            self.skipTest("Failing on keras lower version")
        self.run_task_test(
            cls=ESMMaskedPLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 5, 10),
        )
