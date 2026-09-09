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
        input_data = {
            "token_ids": np.array([[1, 2, 3, 4, 5, 6]], dtype="int32"),
            "padding_mask": np.ones((1, 6), dtype="int32"),
            "token_type_ids": np.array([[1, 1, 1, 0, 0, 0]], dtype="int32"),
        }
        full_output = backbone(input_data)
        cache = ops.zeros(
            (1, backbone.cache_slots, 2, 6, 4, 4), dtype=backbone.compute_dtype
        )
        _, cache = backbone.call_with_cache(
            input_data["token_ids"][:, :3],
            cache,
            cache_update_index=0,
            token_type_ids=input_data["token_type_ids"][:, :3],
        )
        for index in range(3, 6):
            cached_output, cache = backbone.call_with_cache(
                input_data["token_ids"][:, index : index + 1],
                cache,
                cache_update_index=index,
            )
            self.assertAllClose(
                full_output[:, index : index + 1], cached_output, atol=1e-5
            )

    def test_frozen_initial_state(self):
        backbone = HrmTextBackbone(**self.init_kwargs)
        self.assertFalse(backbone.initial_state.z_L_init.trainable)
        self.assertIn(
            backbone.initial_state.z_L_init, backbone.non_trainable_weights
        )

    def test_l_bp_cycles_config(self):
        backbone = HrmTextBackbone(
            **self.init_kwargs,
            l_bp_cycles=[0, 2],
            initializer_range=0.03,
            embedding_scale=None,
        )
        config = backbone.get_config()
        self.assertEqual(config["l_bp_cycles"], [0, 2])
        self.assertEqual(config["initializer_range"], 0.03)
        self.assertAllClose(config["embedding_scale"], 1.0 / 0.03)

    def test_l_bp_cycles_validation(self):
        with self.assertRaises(ValueError):
            HrmTextBackbone(**self.init_kwargs, l_bp_cycles=[1, 1, 1])
        with self.assertRaises(ValueError):
            HrmTextBackbone(**self.init_kwargs, l_bp_cycles=[-1])
        with self.assertRaises(ValueError):
            HrmTextBackbone(**self.init_kwargs, l_bp_cycles=[1.5])
