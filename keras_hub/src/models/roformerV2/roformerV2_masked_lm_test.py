from keras_hub.src.models.roformerV2.roformerV2_backbone import (
    RoformerV2Backbone,
)
from keras_hub.src.models.roformerV2.roformerV2_masked_lm import (
    RoformerV2MaskedLM,
)
from keras_hub.src.models.roformerV2.roformerV2_masked_lm_preprocessor import (
    RoformerV2MaskedLMPreprocessor,
)
from keras_hub.src.models.roformerV2.roformerV2_tokenizer import (
    RoformerV2Tokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class RoformerV2MaskedLMTest(TestCase):
    def setUp(self):
        # Setup model.
        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.vocab += ["the", "quick", "brown", "fox", "."]
        self.preprocessor = RoformerV2MaskedLMPreprocessor(
            RoformerV2Tokenizer(vocabulary=self.vocab),
            # Simplify our testing by masking every available token.
            mask_selection_rate=1.0,
            mask_token_rate=1.0,
            random_token_rate=0.0,
            mask_selection_length=5,
            sequence_length=5,
        )
        self.backbone = RoformerV2Backbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            num_heads=2,
            hidden_dim=2,
            intermediate_dim=4,
            head_size=1,
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
            cls=RoformerV2MaskedLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 5, 10),
        )
