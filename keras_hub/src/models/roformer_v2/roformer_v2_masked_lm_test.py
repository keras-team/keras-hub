import numpy as np

from keras_hub.src.models.roformer_v2.roformer_v2_backbone import (
    RoformerV2Backbone,
)
from keras_hub.src.models.roformer_v2.roformer_v2_masked_lm import (
    RoformerV2MaskedLM,
)
from keras_hub.src.models.roformer_v2.roformer_v2_masked_lm_preprocessor import (  # noqa: E501
    RoformerV2MaskedLMPreprocessor,
)
from keras_hub.src.models.roformer_v2.roformer_v2_tokenizer import (
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
        self.run_task_test(
            cls=RoformerV2MaskedLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 5, 10),
        )


class RoformerV2MaskedLMDynamicMaskTest(TestCase):
    def test_dynamic_mask_positions(self):
        backbone = RoformerV2Backbone(
            vocabulary_size=10,
            num_layers=2,
            num_heads=2,
            hidden_dim=4,
            intermediate_dim=8,
            head_size=2,
        )
        model = RoformerV2MaskedLM(
            backbone=backbone,
            preprocessor=None,
        )
        inputs = {
            "token_ids": np.array([[1, 2, 3, 4, 5]] * 2, dtype="int32"),
            "segment_ids": np.zeros((2, 5), dtype="int32"),
        }

        for mask_count in (2, 4):
            inputs["mask_positions"] = np.tile(
                np.arange(mask_count, dtype="int32"), (2, 1)
            )
            outputs = model(inputs)
            self.assertEqual(outputs.shape, (2, mask_count, 10))
