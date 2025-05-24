import pytest
from keras import ops

from keras_hub.src.models.stablelm.stablelm_backbone import StableLMBackbone
from keras_hub.src.tests.test_case import TestCase


class StableLMBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_query_heads": 2,
            "hidden_dim": 2,
            "intermediate_dim": 4,
            "num_key_value_heads": 2
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        """Test that the backbone processes with expected shape."""
        self.run_backbone_test(
            cls=StableLMBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 2), 
        )

    @pytest.mark.large
    def test_saved_model(self):
        """Test that the model can be saved and loaded successfully."""
        self.run_model_saving_test(
            cls=StableLMBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

   