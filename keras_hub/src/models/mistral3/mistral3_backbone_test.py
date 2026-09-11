import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.mistral3.mistral3_backbone import Mistral3Backbone
from keras_hub.src.models.mistral3.mistral3_vision_encoder import (
    Mistral3MultiModalProjector,
)
from keras_hub.src.models.mistral3.mistral3_vision_encoder import (
    Mistral3VisionEncoder,
)
from keras_hub.src.models.mistral3.mistral3_vision_encoder import (
    compute_image_placeholder_indices,
)
from keras_hub.src.tests.test_case import TestCase


class Mistral3BackboneTest(TestCase):
    def setUp(self):
        self.text_init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_query_heads": 8,
            "num_key_value_heads": 4,
            "hidden_dim": 16,
            "intermediate_dim": 8,
            "sliding_window": 2,
        }

        vision_encoder = Mistral3VisionEncoder(
            image_size=8,
            patch_size=4,
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            head_dim=4,
            intermediate_dim=8,
        )
        multimodal_projector = Mistral3MultiModalProjector(
            vision_hidden_dim=8,
            text_hidden_dim=self.text_init_kwargs["hidden_dim"],
            spatial_merge_size=1,
            patch_size=4,
            image_size=8,
        )
        # Must stay inside `vocabulary_size`: the embedding lookup happens
        # before the image-text merger overwrites these positions.
        self.image_token_index = 9
        self.init_kwargs = {
            **self.text_init_kwargs,
            "vision_encoder": vision_encoder,
            "multimodal_projector": multimodal_projector,
            "image_token_index": self.image_token_index,
        }
        # Two 8x8 images, each a 2x2 patch grid; `spatial_merge_size=1`
        # makes every patch its own merge window (4 rows per image).
        token_ids = ops.array(
            [
                [self.image_token_index] * 4 + [3],
                [self.image_token_index] * 4 + [4],
            ],
            dtype="int32",
        )
        placeholder_indices = compute_image_placeholder_indices(
            token_ids, image_token_index=self.image_token_index
        )
        self.input_data = {
            "token_ids": token_ids,
            "padding_mask": ops.ones((2, 5), dtype="int32"),
            "pixel_values": ops.convert_to_tensor(
                np.random.rand(2, 3, 8, 8).astype("float32")
            ),
            "image_sizes": ops.array([[8, 8], [8, 8]], dtype="int32"),
            "placeholder_indices": ops.convert_to_tensor(placeholder_indices),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=Mistral3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(
                2,
                5,
                self.text_init_kwargs["hidden_dim"],
            ),
            # Image inputs have no sequence axis to slice, so skip the
            # default variable-length sweep.
            variable_length_data=[self.input_data],
            # `run_quantization_test` rebuilds `vision_encoder`/
            # `multimodal_projector` as standalone objects to apply a
            # path-keyed `DTypePolicyMap`, but their sublayer paths change
            # once they're no longer nested under the backbone -- the same
            # structural mismatch `gemma3_backbone_test.py` works around for
            # its own vision-encoder-bearing backbone.
            run_quantization_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Mistral3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_variable_images_per_prompt(self):
        # One prompt with one image, one with two.
        token_ids = ops.array(
            [
                [self.image_token_index] * 4 + [3, 0, 0, 0, 0],
                [self.image_token_index] * 8 + [4],
            ],
            dtype="int32",
        )
        placeholder_indices = compute_image_placeholder_indices(
            token_ids, image_token_index=self.image_token_index
        )
        input_data = {
            "token_ids": token_ids,
            "padding_mask": ops.ones((2, 9), dtype="int32"),
            "pixel_values": ops.convert_to_tensor(
                np.random.rand(3, 3, 8, 8).astype("float32")
            ),
            "image_sizes": ops.array([[8, 8], [8, 8], [8, 8]], dtype="int32"),
            "placeholder_indices": ops.convert_to_tensor(placeholder_indices),
        }
        model = Mistral3Backbone(**self.init_kwargs)
        output = model(input_data)
        self.assertEqual(
            ops.shape(output),
            (2, 9, self.text_init_kwargs["hidden_dim"]),
        )

    def test_num_parameters(self):
        model = Mistral3Backbone(**self.init_kwargs)
        self.assertEqual(model.count_params(), 4016)
        self.assertEqual(len(model.layers), 11)

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        token_ids = ops.array([[1, 1824, 349, 524, 11234, 28804]])
        input_data = {
            "token_ids": token_ids,
            "padding_mask": ops.ones_like(token_ids),
            "pixel_values": ops.zeros((0, 3, 14, 14), dtype="float32"),
            "image_sizes": ops.zeros((0, 2), dtype="int32"),
            "placeholder_indices": ops.zeros((1, 0), dtype="int32"),
        }
        for preset in Mistral3Backbone.presets:
            self.run_preset_test(
                cls=Mistral3Backbone,
                preset=preset,
                input_data=input_data,
            )
