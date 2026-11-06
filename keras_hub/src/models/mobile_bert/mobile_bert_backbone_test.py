import numpy as np
import pytest

from keras_hub.src.models.mobile_bert.mobile_bert_backbone import (
    MobileBertBackbone,
)
from keras_hub.src.tests.test_case import TestCase


class MobileBertBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 16,
            "embedding_dim": 8,
            "intra_bottleneck_dim": 8,
            "intermediate_dim": 16,
            "max_sequence_length": 16,
            "num_segments": 2,
            "use_pooler_dense": True,
        }
        self.input_data = {
            "token_ids": np.ones((2, 4), dtype="int32"),
            "segment_ids": np.zeros((2, 4), dtype="int32"),
            "padding_mask": np.ones((2, 4), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=MobileBertBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "sequence_output": (2, 4, 16),
                "pooled_output": (2, 16),
            },
            variable_length_data=(
                {
                    "token_ids": np.ones((2, 8), dtype="int32"),
                    "segment_ids": np.zeros((2, 8), dtype="int32"),
                    "padding_mask": np.ones((2, 8), dtype="int32"),
                },
            ),
            run_mixed_precision_check=True,
            run_quantization_check=True,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MobileBertBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
