import numpy as np
from keras import ops

from keras_hub.src.models.hrm_text.hrm_text_backbone import HrmTextBackbone
from keras_hub.src.models.hrm_text.hrm_text_backbone import (
    make_hrm_text_attention_mask,
)
from keras_hub.src.tests.test_case import TestCase


class HrmTextBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 32,
            "hidden_dim": 16,
            "intermediate_dim": 32,
            "num_layers_per_stack": 2,
            "num_attention_heads": 4,
            "head_dim": 4,
            "h_cycles": 2,
            "l_cycles": 2,
            "max_sequence_length": 8,
        }
        self.input_data = {
            "token_ids": np.array([[1, 2, 3, 4]], dtype="int32"),
            "padding_mask": np.array([[1, 1, 1, 1]], dtype="int32"),
            "token_type_ids": np.array([[1, 1, 0, 0]], dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=HrmTextBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(1, 4, 16),
        )

    def test_prefix_lm_attention_mask(self):
        mask = make_hrm_text_attention_mask(
            np.array([[1, 1, 0, 0]], dtype="int32"),
            np.array([[1, 1, 1, 1]], dtype="int32"),
        )
        self.assertAllEqual(
            mask,
            np.array(
                [
                    [
                        [1, 1, 0, 0],
                        [1, 1, 0, 0],
                        [1, 1, 1, 0],
                        [1, 1, 1, 1],
                    ]
                ],
                dtype="bool",
            ),
        )

    def test_cached_forward_matches_full_forward(self):
        backbone = HrmTextBackbone(**self.init_kwargs)
        full_output = backbone(self.input_data)
        cache = ops.zeros(
            (1, backbone.cache_slots, 2, 4, 4, 4), dtype=backbone.compute_dtype
        )
        _, cache = backbone.call_with_cache(
            self.input_data["token_ids"][:, :3],
            cache,
            cache_update_index=0,
            token_type_ids=self.input_data["token_type_ids"][:, :3],
        )
        cached_output, _ = backbone.call_with_cache(
            self.input_data["token_ids"][:, 3:],
            cache,
            cache_update_index=3,
        )
        self.assertAllClose(full_output[:, 3:], cached_output, atol=1e-5)
