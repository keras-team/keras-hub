import pytest
from keras import ops

from keras_hub.src.models.edrec.edrec_backbone import EdRecBackbone
from keras_hub.src.tests.test_case import TestCase


class EdRecBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocab_size": 10,
            "num_layers_enc": 2,
            "num_layers_dec": 2,
            "num_heads": 2,
            "hidden_dim": 4,
            "intermediate_dim": 8,
            "dropout": 0.0,
        }
        self.input_data = {
            "encoder_token_ids": ops.ones((2, 5), dtype="int32"),
            "encoder_padding_mask": ops.zeros((2, 5), dtype="int32"),
            "decoder_token_ids": ops.ones((2, 5), dtype="int32"),
            "decoder_padding_mask": ops.zeros((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=EdRecBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            run_mixed_precision_check=False,
            expected_output_shape={
                "encoder_sequence_output": (2, 5, 4),
                "decoder_sequence_output": (2, 5, 4),
            },
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=EdRecBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
