import keras
import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.mobile_bert.mobile_bert_backbone import (
    MobileBertBackbone,
)
from keras_hub.src.models.mobile_bert.mobile_bert_masked_lm import (
    MobileBertMaskedLM,
)
from keras_hub.src.models.mobile_bert.mobile_bert_masked_lm_preprocessor import (  # noqa: E501
    MobileBertMaskedLMPreprocessor,
)
from keras_hub.src.models.mobile_bert.mobile_bert_tokenizer import (
    MobileBertTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class MobileBertMaskedLMTest(TestCase):
    def setUp(self):
        # Setup model.
        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.vocab += ["the", "quick", "brown", "fox", "."]
        self.preprocessor = MobileBertMaskedLMPreprocessor(
            MobileBertTokenizer(vocabulary=self.vocab),
            # Simplify our testing by masking every available token.
            mask_selection_rate=1.0,
            mask_token_rate=1.0,
            random_token_rate=0.0,
            mask_selection_length=4,
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
        }
        self.train_data = (
            ["the quick brown fox.", "the slow brown fox."],  # Features.
        )
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_masked_lm_basics(self):
        self.run_task_test(
            cls=MobileBertMaskedLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 4, 10),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MobileBertMaskedLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in MobileBertMaskedLM.presets:
            self.run_preset_test(
                cls=MobileBertMaskedLM,
                preset=preset,
                input_data=self.input_data,
            )

    def test_layer_norm_epsilon_matches_backbone(self):
        kwargs = self.init_kwargs.copy()
        kwargs["backbone"] = MobileBertBackbone.from_config(
            {**self.backbone.get_config(), "layer_norm_epsilon": 1e-5}
        )
        model = MobileBertMaskedLM(**kwargs)
        self.assertEqual(
            model.masked_lm_head.transform_layer_norm.epsilon,
            1e-5,
        )

    def test_head_dtype_follows_backbone(self):
        backbone = MobileBertBackbone.from_config(
            {**self.backbone.get_config(), "dtype": "mixed_float16"}
        )
        model = MobileBertMaskedLM(backbone=backbone, preprocessor=None)
        self.assertEqual(model.masked_lm_head.compute_dtype, "float16")
        self.assertEqual(model.masked_lm_head.variable_dtype, "float32")
        output = model(self.input_data)
        self.assertEqual(
            keras.backend.standardize_dtype(output.dtype), "float16"
        )
        self.assertTrue(np.isfinite(ops.convert_to_numpy(output)).all())
