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

    def _build_multimodal_kwargs_and_data(self):
        # `head_dim` must be a multiple of 4: Pixtral's 2D rotary embedding
        # splits it into height/width halves, each further halved for
        # sin/cos, so anything else produces mismatched weight shapes.
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
            text_hidden_dim=self.init_kwargs["hidden_dim"],
            spatial_merge_size=1,
            patch_size=4,
            image_size=8,
        )
        # `image_token_index` must stay inside `vocabulary_size`, since the
        # token embedding lookup happens before the image-text merger
        # overwrites those positions.
        image_token_index = 9
        init_kwargs = {
            **self.init_kwargs,
            "vision_encoder": vision_encoder,
            "multimodal_projector": multimodal_projector,
            "image_token_index": image_token_index,
        }
        # Two images, each an 8x8 canvas patchified into a 2x2 grid (4
        # patches); with `spatial_merge_size=1` every patch is its own
        # merge window, so each image contributes 4 image-feature rows.
        token_ids = ops.array(
            [
                [image_token_index] * 4 + [3],
                [image_token_index] * 4 + [4],
            ],
            dtype="int32",
        )
        placeholder_indices = compute_image_placeholder_indices(
            token_ids, image_token_index=image_token_index
        )
        input_data = {
            "token_ids": token_ids,
            "padding_mask": ops.ones((2, 5), dtype="int32"),
            # Keras rejects a single nested `call()` argument (here, the
            # `inputs` dict) that mixes backend tensors with plain
            # NumPy/non-tensor values, so every entry must be a backend
            # tensor even though `pixel_values`/`placeholder_indices` are
            # naturally produced as NumPy arrays.
            "pixel_values": ops.convert_to_tensor(
                np.random.rand(2, 3, 8, 8).astype("float32")
            ),
            "image_sizes": ops.array([[8, 8], [8, 8]], dtype="int32"),
            # `placeholder_indices_input` is a `keras.Input(shape=(None,))`,
            # i.e. rank 2 `(batch, N)`, even though the values themselves are
            # flat global indices with no real per-example meaning —
            # `Mistral3ImageTextEmbeddingMerger` flattens whatever batch
            # dimension it's given right back out. Add a leading batch-of-1
            # dim here to match that expected input rank.
            "placeholder_indices": ops.convert_to_tensor(placeholder_indices)[
                None, :
            ],
        }
        return init_kwargs, input_data

    def test_multimodal_backbone_forward_pass(self):
        init_kwargs, input_data = self._build_multimodal_kwargs_and_data()
        model = MistralBackbone(**init_kwargs)
        output = model(input_data)
        self.assertEqual(output.shape, (2, 5, self.init_kwargs["hidden_dim"]))

    def test_multimodal_backbone_serialization(self):
        init_kwargs, _ = self._build_multimodal_kwargs_and_data()
        model = MistralBackbone(**init_kwargs)
        self.run_serialization_test(model)

    def test_vision_projector_must_be_paired(self):
        # `head_dim` must be a multiple of 4: Pixtral's 2D rotary embedding
        # splits it into height/width halves, each further halved for
        # sin/cos, so anything else produces mismatched weight shapes.
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
            text_hidden_dim=self.init_kwargs["hidden_dim"],
            spatial_merge_size=1,
            patch_size=4,
            image_size=8,
        )
        with self.assertRaises(ValueError):
            MistralBackbone(**self.init_kwargs, vision_encoder=vision_encoder)
        with self.assertRaises(ValueError):
            MistralBackbone(
                **self.init_kwargs, multimodal_projector=multimodal_projector
            )
