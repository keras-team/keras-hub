import numpy as np
import pytest
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.qwen3_omni.qwen3_omni_audio_encoder import (
    Qwen3OmniAudioEncoder,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_backbone import (
    Qwen3OmniBackbone,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_vision_encoder import (
    Qwen3OmniVisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3OmniBackboneTest(TestCase, parameterized.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.sequence_length = 8

        # === Vision Encoder ===
        vision_encoder = Qwen3OmniVisionEncoder(
            depth=2,
            hidden_size=32,
            num_heads=4,
            intermediate_size=64,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=16,
            num_position_embeddings=49,
            deepstack_visual_indexes=[0],
            dtype="float32",
        )

        # === Audio Encoder ===
        audio_encoder = Qwen3OmniAudioEncoder(
            num_mel_bins=80,
            d_model=32,
            encoder_layers=2,
            encoder_attention_heads=4,
            encoder_ffn_dim=64,
            output_dim=16,
            max_source_positions=100,
            scale_embedding=False,
            dtype="float32",
        )

        # === Text + Vision + Audio Backbone ===
        self.init_kwargs = {
            "vocabulary_size": 20,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "hidden_dim": 16,
            "intermediate_dim": 32,
            "head_dim": 4,
            "moe_intermediate_dim": 16,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "norm_topk_prob": True,
            "decoder_sparse_step": 1,
            "mrope_section": (1, 1, 0),
            "layer_norm_epsilon": 1e-6,
            "rope_max_wavelength": 10000,
            "rope_scaling_factor": 1.0,
            "dropout": 0.0,
            "tie_word_embeddings": False,
            "sliding_window_size": None,
            "router_aux_loss_coefficient": 0.001,
            "mlp_only_layers": [],
            "vision_encoder": vision_encoder,
            "audio_encoder": audio_encoder,
            "image_token_id": 5,
            "video_token_id": 6,
            "audio_token_id": 7,
            "dtype": "float32",
        }

        self.input_data = {
            "token_ids": ops.ones(
                (self.batch_size, self.sequence_length), dtype="int32"
            ),
            "padding_mask": ops.ones(
                (self.batch_size, self.sequence_length), dtype="int32"
            ),
        }

        # === Text-Only Backbone ===
        self.text_init_kwargs = self.init_kwargs.copy()
        self.text_init_kwargs["vision_encoder"] = None
        self.text_init_kwargs["audio_encoder"] = None

    @parameterized.named_parameters(
        ("multimodal", "multimodal"), ("text_only", "text_only")
    )
    def test_backbone_basics(self, backbone_type):
        if backbone_type == "multimodal":
            init_kwargs = self.init_kwargs
        else:
            init_kwargs = self.text_init_kwargs

        self.run_backbone_test(
            cls=Qwen3OmniBackbone,
            init_kwargs=init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(
                self.batch_size,
                self.sequence_length,
                16,
            ),
            run_quantization_check=(backbone_type == "text_only"),
        )

    @parameterized.named_parameters(
        ("multimodal", "multimodal"), ("text_only", "text_only")
    )
    @pytest.mark.large
    def test_saved_model(self, backbone_type):
        if backbone_type == "multimodal":
            init_kwargs = self.init_kwargs
        else:
            init_kwargs = self.text_init_kwargs

        self.run_model_saving_test(
            cls=Qwen3OmniBackbone,
            init_kwargs=init_kwargs,
            input_data=self.input_data,
        )

    def test_architecture_characteristics(self):
        model = Qwen3OmniBackbone(**self.init_kwargs)
        self.assertEqual(len(model.transformer_layers), 2)
        self.assertIsNotNone(model.vision_encoder)
        self.assertIsNotNone(model.audio_encoder)

        text_model = Qwen3OmniBackbone(**self.text_init_kwargs)
        self.assertIsNone(text_model.vision_encoder)
        self.assertIsNone(text_model.audio_encoder)

    def test_auxiliary_loss(self):
        model = Qwen3OmniBackbone(**self.init_kwargs)
        _ = model(self.input_data, training=True)
        self.assertTrue(
            len(model.losses) > 0, "Auxiliary losses should be present"
        )
        for loss in model.losses:
            self.assertGreater(loss, 0.0, "Auxiliary loss should be positive")

    def test_vision_fusion_forward(self):
        model = Qwen3OmniBackbone(**self.init_kwargs)

        token_ids = np.ones((1, self.sequence_length), dtype="int32")
        token_ids[0, 3] = 5
        padding_mask = np.ones((1, self.sequence_length), dtype="int32")
        pixel_values = (
            np.random.RandomState(0).randn(1, 2, 4, 4, 3).astype("float32")
        )
        grid_thw = np.array([[1, 2, 2]], dtype="int32")

        output_fused = model(
            {
                "token_ids": token_ids,
                "padding_mask": padding_mask,
                "pixel_values": pixel_values,
                "grid_thw": grid_thw,
            }
        )
        self.assertEqual(ops.shape(output_fused), (1, self.sequence_length, 16))

        output_text = model(
            {
                "token_ids": token_ids,
                "padding_mask": padding_mask,
            }
        )
        self.assertEqual(ops.shape(output_text), (1, self.sequence_length, 16))

        fused_np = ops.convert_to_numpy(output_fused)
        text_np = ops.convert_to_numpy(output_text)
        self.assertFalse(
            np.allclose(fused_np, text_np, atol=1e-5),
            "Vision fusion should change the output.",
        )
