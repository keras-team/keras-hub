import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.mistral.mistral_backbone import MistralBackbone
from keras_hub.src.models.mistral.mistral_vision_encoder import (
    Mistral3MultiModalProjector,
)
from keras_hub.src.models.mistral.mistral_vision_encoder import (
    Mistral3VisionEncoder,
)
from keras_hub.src.models.mistral.mistral_vision_encoder import (
    compute_image_placeholder_indices,
)
from keras_hub.src.tests.test_case import TestCase


class MistralBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_query_heads": 8,
            "num_key_value_heads": 4,
            "hidden_dim": 16,
            "intermediate_dim": 8,
            "sliding_window": 2,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=MistralBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 16),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MistralBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_num_parameters(self):
        model = MistralBackbone(**self.init_kwargs)
        # Reference value calculated using the PyTorch model
        self.assertEqual(model.count_params(), 2704)

    def test_explicit_head_dim(self):
        # Magistral-style config: `head_dim` is set explicitly and does not
        # equal `hidden_dim // num_query_heads`. `sliding_window=None` is
        # also exercised here. Run the full backbone test so the new path
        # gets serialization and precision coverage.
        init_kwargs = {
            **self.init_kwargs,
            "sliding_window": None,
            "head_dim": 4,
        }
        self.run_backbone_test(
            cls=MistralBackbone,
            init_kwargs=init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 16),
        )
        model = MistralBackbone(**init_kwargs)
        attention = model.transformer_layers[0]._self_attention_layer
        self.assertEqual(attention._head_dim, 4)

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=MistralBackbone,
            preset="mistral_7b_en",
            input_data={
                "token_ids": ops.array([[1, 1824, 349, 524, 11234, 28804]]),
                "padding_mask": ops.ones((1, 6), dtype="int32"),
            },
            expected_output_shape=(1, 6, 4096),
            # The forward pass from a preset should be stable!
            # Reference values computed using PyTorch HF model.
            expected_partial_output=ops.array(
                [-1.6875, 0.5117, -1.7188, 2.3125, -0.0996]
            ),
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in MistralBackbone.presets:
            self.run_preset_test(
                cls=MistralBackbone,
                preset=preset,
                input_data=self.input_data,
            )


class MistralMultimodalBackboneTest(TestCase):
    """Tests for `MistralBackbone` configured with a vision encoder."""

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
            "placeholder_indices": ops.reshape(
                ops.convert_to_tensor(placeholder_indices), (2, 4)
            ),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=MistralBackbone,
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

    def test_vision_projector_must_be_paired(self):
        with self.assertRaises(ValueError):
            MistralBackbone(
                **self.text_init_kwargs,
                vision_encoder=self.init_kwargs["vision_encoder"],
            )
        with self.assertRaises(ValueError):
            MistralBackbone(
                **self.text_init_kwargs,
                multimodal_projector=self.init_kwargs["multimodal_projector"],
            )
