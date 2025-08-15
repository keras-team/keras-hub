import keras
import numpy as np
import pytest
from absl.testing import parameterized

from keras_hub.src.models.d_fine.d_fine_backbone import DFineBackbone
from keras_hub.src.models.hgnetv2.hgnetv2_backbone import HGNetV2Backbone
from keras_hub.src.tests.test_case import TestCase


class DFineBackboneTest(TestCase):
    def setUp(self):
        self.labels = [
            {
                "boxes": np.array([[0.5, 0.5, 0.2, 0.2], [0.4, 0.4, 0.1, 0.1]]),
                "labels": np.array([1, 10]),
            },
            {
                "boxes": np.array([[0.6, 0.6, 0.3, 0.3]]),
                "labels": np.array([20]),
            },
        ]
        hgnetv2_backbone = HGNetV2Backbone(
            stem_channels=[3, 16, 16],
            stackwise_stage_filters=[
                [16, 16, 64, 1, 3, 3],
                [64, 32, 256, 1, 3, 3],
                [256, 64, 512, 2, 3, 5],
                [512, 128, 1024, 1, 3, 5],
            ],
            apply_downsample=[False, True, True, True],
            use_lightweight_conv_block=[False, False, True, True],
            depths=[1, 1, 2, 1],
            hidden_sizes=[64, 256, 512, 1024],
            embedding_size=16,
            use_learnable_affine_block=True,
            hidden_act="relu",
            image_shape=(None, None, 3),
            out_features=["stage3", "stage4"],
        )
        self.base_init_kwargs = {
            "backbone": hgnetv2_backbone,
            "decoder_in_channels": [128, 128],
            "encoder_hidden_dim": 128,
            "num_denoising": 100,
            "num_labels": 80,
            "hidden_dim": 128,
            "learn_initial_query": False,
            "num_queries": 300,
            "anchor_image_size": (256, 256),
            "feat_strides": [16, 32],
            "num_feature_levels": 2,
            "encoder_in_channels": [512, 1024],
            "encode_proj_layers": [1],
            "num_attention_heads": 8,
            "encoder_ffn_dim": 512,
            "num_encoder_layers": 1,
            "hidden_expansion": 0.34,
            "depth_multiplier": 0.5,
            "eval_idx": -1,
            "num_decoder_layers": 3,
            "decoder_attention_heads": 8,
            "decoder_ffn_dim": 512,
            "decoder_n_points": [6, 6],
            "lqe_hidden_dim": 64,
            "num_lqe_layers": 2,
            "out_features": ["stage3", "stage4"],
            "image_shape": (None, None, 3),
            "seed": 0,
        }
        self.input_data = keras.random.uniform((2, 256, 256, 3))

    @parameterized.named_parameters(
        ("default_eval_last", False, 300, -1, 4),
        ("denoising_eval_last", True, 500, -1, 4),
        ("default_eval_first", False, 300, 0, 4),
        ("denoising_eval_first", True, 500, 0, 4),
        ("default_eval_middle", False, 300, 1, 4),
        ("denoising_eval_middle", True, 500, 1, 4),
    )
    def test_backbone_basics(
        self, use_noise_and_labels, total_queries, eval_idx, num_logit_layers
    ):
        init_kwargs = self.base_init_kwargs.copy()
        init_kwargs["eval_idx"] = eval_idx
        if use_noise_and_labels:
            init_kwargs["box_noise_scale"] = 1.0
            init_kwargs["label_noise_ratio"] = 0.5
            init_kwargs["labels"] = self.labels
        expected_output_shape = {
            "last_hidden_state": (2, total_queries, 128),
            "intermediate_hidden_states": (2, 3, total_queries, 128),
            "intermediate_logits": (2, num_logit_layers, total_queries, 80),
            "intermediate_reference_points": (
                2,
                num_logit_layers,
                total_queries,
                4,
            ),
            "intermediate_predicted_corners": (
                2,
                num_logit_layers,
                total_queries,
                132,
            ),
            "initial_reference_points": (
                2,
                num_logit_layers,
                total_queries,
                4,
            ),
            "encoder_last_hidden_state": (2, 16, 16, 128),
            "init_reference_points": (2, total_queries, 4),
            "enc_topk_logits": (2, 300, 80),
            "enc_topk_bboxes": (2, 300, 4),
            "enc_outputs_class": (2, 320, 80),
            "enc_outputs_coord_logits": (2, 320, 4),
        }
        # NOTE: The `run_vision_backbone_test` helper's `channels_first`
        # check transposes all 3D / 4D outputs by default, which is incorrect
        # for `DFineBackbone` non-spatial outputs like
        # `intermediate_hidden_states` (shape: `(batch_size, num_decoder_layers,
        # num_queries, hidden_dim)`). Use `spatial_output_keys` to specify
        # spatial outputs (e.g., `encoder_last_hidden_state`) for transposition,
        # ensuring congruence with reference outputs.
        # https://github.com/huggingface/transformers/blob/d37f7517972f67e3f2194c000ed0f87f064e5099/src/transformers/models/d_fine/modeling_d_fine.py#L1595-L1614
        # NOTE: `last_hidden_state`, `intermediate_hidden_states`, and
        # `decoder_hidden_state` are non-spatial object query embeddings,
        # despite their names, and should not be transposed. Other outputs
        # not listed are visibly non-spatial.
        self.run_vision_backbone_test(
            cls=DFineBackbone,
            init_kwargs=init_kwargs,
            input_data=self.input_data,
            expected_output_shape=expected_output_shape,
            spatial_output_keys=[
                "encoder_last_hidden_state",
                "encoder_hidden_states",
            ],
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=DFineBackbone,
            init_kwargs=self.base_init_kwargs,
            input_data=self.input_data,
        )
