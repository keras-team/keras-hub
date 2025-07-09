import keras
import numpy as np
import pytest
from absl.testing import parameterized

from keras_hub.src.models.d_fine.d_fine_backbone import DFineBackbone
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
        self.stackwise_stage_filters = [
            [16, 16, 64, 1, 3, 3],
            [64, 32, 256, 1, 3, 3],
            [256, 64, 512, 2, 3, 5],
            [512, 128, 1024, 1, 3, 5],
        ]
        self.apply_downsample = [False, True, True, True]
        self.use_lightweight_conv_block = [False, False, True, True]
        self.base_init_kwargs = {
            "decoder_in_channels": [128, 128],
            "encoder_hidden_dim": 128,
            "num_denoising": 100,
            "num_labels": 80,
            "hidden_dim": 128,
            "learn_initial_query": False,
            "num_queries": 300,
            "anchor_image_size": (256, 256),
            "feat_strides": [16, 32],
            "batch_norm_eps": 1e-5,
            "num_feature_levels": 2,
            "layer_norm_eps": 1e-5,
            "encoder_in_channels": [512, 1024],
            "encode_proj_layers": [1],
            "positional_encoding_temperature": 10000,
            "eval_size": None,
            "normalize_before": False,
            "num_attention_heads": 8,
            "dropout": 0.0,
            "encoder_activation_function": "gelu",
            "activation_dropout": 0.0,
            "encoder_ffn_dim": 512,
            "encoder_layers": 1,
            "hidden_expansion": 0.34,
            "depth_mult": 0.5,
            "eval_idx": -1,
            "decoder_layers": 3,
            "reg_scale": 4.0,
            "max_num_bins": 32,
            "up": 0.5,
            "decoder_attention_heads": 8,
            "attention_dropout": 0.0,
            "decoder_activation_function": "relu",
            "decoder_ffn_dim": 512,
            "decoder_offset_scale": 0.5,
            "decoder_method": "default",
            "decoder_n_points": [6, 6],
            "top_prob_values": 4,
            "lqe_hidden_dim": 64,
            "lqe_layers_count": 2,
            "hidden_act": "relu",
            "stem_channels": [3, 16, 16],
            "use_learnable_affine_block": True,
            "num_channels": 3,
            "stackwise_stage_filters": self.stackwise_stage_filters,
            "apply_downsample": self.apply_downsample,
            "use_lightweight_conv_block": self.use_lightweight_conv_block,
            "layer_scale": 1.0,
            "out_features": ["stage3", "stage4"],
            "image_shape": (None, None, 3),
            "data_format": "channels_last",
            "depths": [1, 1, 2, 1],
            "hidden_sizes": [64, 256, 512, 1024],
            "embedding_size": 16,
            "seed": 0,
        }
        self.input_data = {
            "pixel_values": keras.random.uniform((2, 256, 256, 3)),
            "pixel_mask": keras.ops.ones((2, 256, 256), dtype="bool"),
        }

    @parameterized.named_parameters(
        ("default", False),
        ("denoising", True),
    )
    def test_backbone_channels_first(self, use_noise_and_labels):
        init_kwargs = self.base_init_kwargs.copy()
        if use_noise_and_labels:
            init_kwargs["box_noise_scale"] = 1.0
            init_kwargs["label_noise_ratio"] = 0.5
            init_kwargs["labels"] = self.labels
        num_queries = init_kwargs["num_queries"]
        num_denoising = (
            init_kwargs["num_denoising"] if use_noise_and_labels else 0
        )
        total_queries = num_queries + 2 * num_denoising
        expected_output_shape = {
            "last_hidden_state": (2, total_queries, 128),
            "intermediate_hidden_states": (2, 3, total_queries, 128),
            "intermediate_logits": (2, 4, total_queries, 80),
            "intermediate_reference_points": (2, 4, total_queries, 4),
            "intermediate_predicted_corners": (2, 3, total_queries, 132),
            "initial_reference_points": (2, 3, total_queries, 4),
            "encoder_last_hidden_state": (2, 16, 16, 128),
            "init_reference_points": (2, total_queries, 4),
            "enc_topk_logits": (2, 300, 80),
            "enc_topk_bboxes": (2, 300, 4),
            "enc_outputs_class": (2, 320, 80),
            "enc_outputs_coord_logits": (2, 320, 4),
        }
        self.run_vision_backbone_test(
            cls=DFineBackbone,
            init_kwargs=init_kwargs,
            input_data=self.input_data,
            expected_output_shape=expected_output_shape,
            expected_pyramid_output_keys=None,
            expected_pyramid_image_sizes=None,
            run_mixed_precision_check=False,
            run_quantization_check=False,
            run_data_format_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=DFineBackbone,
            init_kwargs=self.base_init_kwargs,
            input_data=self.input_data,
        )
