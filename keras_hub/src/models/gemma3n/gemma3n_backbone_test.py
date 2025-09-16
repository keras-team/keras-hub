from copy import deepcopy

import numpy as np
import pytest
from absl.testing import parameterized

try:
    from keras_hub.src.models.mobilenetv5.mobilenetv5_backbone import (
        MobileNetV5Backbone,
    )
    from keras_hub.src.models.mobilenetv5.mobilenetv5_builder import (
        decode_arch_def,
    )

    mobilenetv5 = True
except ImportError:
    mobilenetv5 = False

from keras_hub.src.models.gemma3n.gemma3n_audio_encoder import (
    Gemma3nAudioEncoder,
)
from keras_hub.src.models.gemma3n.gemma3n_backbone import Gemma3nBackbone
from keras_hub.src.tests.test_case import TestCase


@pytest.mark.skipif(
    not mobilenetv5,
    reason="The pull request for MobileNetV5 is still open.",
)
class Gemma3nBackboneTest(TestCase):
    def setUp(self):
        self.batch_size = 1
        self.text_vocab_size = 50
        self.text_sequence_length = 16
        self.image_height = 224
        self.image_width = 224
        self.audio_sequence_length = 16
        self.audio_feature_size = 32
        # === Vision Encoder ===
        if mobilenetv5:
            vision_arch_def = [["er_r1_k3_s1_e1_c16"]]
            vision_block_args = decode_arch_def(vision_arch_def)
            vision_encoder = MobileNetV5Backbone(
                block_args=vision_block_args,
                num_features=4,
                image_shape=(self.image_height, self.image_width, 3),
                use_msfa=False,
            )
            vision_encoder_config = vision_encoder.get_config()
        else:
            vision_encoder_config = None
        # === Audio Encoder ===
        audio_encoder = Gemma3nAudioEncoder(
            hidden_size=8,
            input_feat_size=self.audio_feature_size,
            sscp_conv_channel_size=[4, 8],
            sscp_conv_kernel_size=[(3, 3), (3, 3)],
            sscp_conv_stride_size=[(2, 2), (2, 2)],
            sscp_conv_group_norm_eps=1e-5,
            conf_num_hidden_layers=1,
            rms_norm_eps=1e-6,
            gradient_clipping=1.0,
            conf_residual_weight=0.5,
            conf_num_attention_heads=1,
            conf_attention_chunk_size=4,
            conf_attention_context_right=5,
            conf_attention_context_left=5,
            conf_attention_logit_cap=50.0,
            conf_conv_kernel_size=5,
            conf_reduction_factor=1,
        )
        # === Multimodal ===
        self.multimodal_init_kwargs = {
            "text_vocab_size": self.text_vocab_size,
            "text_hidden_size": 8,
            "num_hidden_layers": 1,
            "pad_token_id": 0,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 8,  # hidden_size / num_attention_heads
            "intermediate_size": [16],
            "hidden_activation": "gelu_approximate",
            "layer_types": ["full_attention"],
            "sliding_window": 4,
            "rope_theta": 10000.0,
            "max_position_embeddings": self.text_sequence_length,
            "vocab_size_per_layer_input": 50,
            "hidden_size_per_layer_input": 2,
            "altup_num_inputs": 2,
            "laurel_rank": 1,
            "vision_encoder_config": vision_encoder_config,
            "vision_hidden_size": 16,
            "audio_encoder_config": audio_encoder.get_config(),
            "audio_hidden_size": 8,
        }
        self.multimodal_input_data = {
            "token_ids": np.random.randint(
                0,
                self.text_vocab_size,
                size=(self.batch_size, self.text_sequence_length),
                dtype="int32",
            ),
            "attention_mask": np.ones(
                (
                    self.batch_size,
                    1,
                    self.text_sequence_length,
                    self.text_sequence_length,
                ),
                dtype=bool,
            ),
            "pixel_values": np.random.rand(
                self.batch_size, 1, self.image_height, self.image_width, 3
            ).astype("float32"),
            "input_features": np.random.rand(
                self.batch_size,
                self.audio_sequence_length,
                self.audio_feature_size,
            ).astype("float32"),
            "input_features_mask": np.zeros(
                (self.batch_size, self.audio_sequence_length), dtype=bool
            ),
        }
        # === Text-Only ===
        self.text_init_kwargs = deepcopy(self.multimodal_init_kwargs)
        del self.text_init_kwargs["vision_encoder_config"]
        del self.text_init_kwargs["audio_encoder_config"]
        del self.text_init_kwargs["vision_hidden_size"]
        del self.text_init_kwargs["audio_hidden_size"]
        self.text_input_data = deepcopy(self.multimodal_input_data)
        del self.text_input_data["pixel_values"]
        del self.text_input_data["input_features"]
        del self.text_input_data["input_features_mask"]

    @parameterized.named_parameters(
        ("multimodal", "multimodal"), ("text_only", "text_only")
    )
    def test_backbone_basics(self, backbone_type):
        if backbone_type == "multimodal":
            init_kwargs = self.multimodal_init_kwargs
            input_data = self.multimodal_input_data
        else:
            init_kwargs = self.text_init_kwargs
            input_data = self.text_input_data
        self.run_backbone_test(
            cls=Gemma3nBackbone,
            init_kwargs=init_kwargs,
            input_data=input_data,
            expected_output_shape=(
                self.batch_size,
                self.text_sequence_length,
                init_kwargs["text_hidden_size"],
            ),
        )

    @parameterized.named_parameters(
        ("multimodal", "multimodal"), ("text_only", "text_only")
    )
    def test_saved_model(self, backbone_type):
        if backbone_type == "multimodal":
            init_kwargs = self.multimodal_init_kwargs
            input_data = self.multimodal_input_data
        else:
            init_kwargs = self.text_init_kwargs
            input_data = self.text_input_data
        self.run_model_saving_test(
            cls=Gemma3nBackbone,
            init_kwargs=init_kwargs,
            input_data=input_data,
        )

    @parameterized.named_parameters(
        ("multimodal", "multimodal", 10354, 7),
        ("text_only", "text_only", 1450, 4),
    )
    def test_architecture_characteristics(
        self, backbone_type, num_params, num_layers
    ):
        if backbone_type == "multimodal":
            init_kwargs = self.multimodal_init_kwargs
        else:
            init_kwargs = self.text_init_kwargs
        model = Gemma3nBackbone(**init_kwargs)
        self.assertEqual(model.count_params(), num_params)
        self.assertEqual(len(model.layers), num_layers)
