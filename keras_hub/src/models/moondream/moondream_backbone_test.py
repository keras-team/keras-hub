import numpy as np
import pytest

from keras_hub.src.models.moondream.moondream_backbone import MoondreamBackbone
from keras_hub.src.tests.test_case import TestCase


class MoondreamBackboneTest(TestCase):
    def setUp(self):
        self.image_size = 28
        self.vision_patch_size = 14
        self.vocabulary_size = 256
        self.text_sequence_length = 16
        self.image_sequence_length = (
            self.image_size // self.vision_patch_size
        ) ** 2  # 4

        self.init_kwargs = {
            "vocabulary_size": self.vocabulary_size,
            "image_size": self.image_size,
            "vision_patch_size": self.vision_patch_size,
            "vision_num_layers": 2,
            "vision_num_heads": 2,
            "vision_hidden_dim": 8,
            "vision_intermediate_dim": 16,
            "projection_dim": 8,
            "text_num_layers": 2,
            "text_hidden_dim": 8,
            "text_intermediate_dim": 16,
            "text_num_query_heads": 2,
            "text_num_key_value_heads": 1,
        }

        batch_size = 2
        self.input_data = {
            "images": np.random.uniform(
                size=(batch_size, self.image_size, self.image_size, 3)
            ).astype("float32"),
            "token_ids": np.random.randint(
                0,
                self.vocabulary_size,
                (batch_size, self.text_sequence_length),
            ),
            "padding_mask": np.ones(
                (batch_size, self.text_sequence_length), dtype="int32"
            ),
        }

    def test_backbone_basics(self):
        batch_size = 2
        expected_seq_len = self.image_sequence_length + self.text_sequence_length
        self.run_backbone_test(
            cls=MoondreamBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(
                batch_size,
                expected_seq_len,
                8,  # text_hidden_dim
            ),
            variable_length_data=[self.input_data],
        )

    def test_image_sequence_length_property(self):
        backbone = MoondreamBackbone(**self.init_kwargs)
        self.assertEqual(
            backbone.image_sequence_length, self.image_sequence_length
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MoondreamBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
