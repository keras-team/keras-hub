import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.gemma3n.gemma3n_backbone import Gemma3nBackbone
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_gemma3n


class TestGemma3nConverter(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = Gemma3nBackbone.from_preset(
            "hf://yujiepan/gemma-3n-tiny-random-dim4",
            load_weights=False,
        )
        self.assertIsInstance(model, Gemma3nBackbone)

    @pytest.mark.large
    def test_class_detection(self):
        model = Backbone.from_preset(
            "hf://yujiepan/gemma-3n-tiny-random-dim4",
            load_weights=False,
        )
        self.assertIsInstance(model, Gemma3nBackbone)

    def test_gemma3n_rope_theta(self):
        # transformers < 5 format
        text_cfg = {
            "vocab_size": 100,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "intermediate_size": 48,
            "layer_types": ["full_attention", "full_attention"],
            "sliding_window": 4096,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 32,
            "vocab_size_per_layer_input": 100,
            "hidden_size_per_layer_input": 32,
            "altup_num_inputs": 1,
            "laurel_rank": 1,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "altup_coef_clip": 1.0,
            "altup_active_idx": 0,
            "altup_correct_scale": False,
            "num_kv_shared_layers": 1,
        }
        transformers_config = {
            "text_config": text_cfg,
            "vision_config": {
                "hidden_size": 32,
                "vocab_size": 100,
                "vocab_offset": 0,
            },
            "audio_config": {
                "hidden_size": 32,
                "input_feat_size": 32,
                "sscp_conv_channel_size": 32,
                "sscp_conv_kernel_size": 3,
                "sscp_conv_stride_size": 2,
                "sscp_conv_group_norm_eps": 1e-5,
                "conf_num_hidden_layers": 2,
                "rms_norm_eps": 1e-5,
                "gradient_clipping": 1.0,
                "conf_residual_weight": 0.1,
                "conf_num_attention_heads": 4,
                "conf_attention_chunk_size": 16,
                "conf_attention_context_right": 8,
                "conf_attention_context_left": 8,
                "conf_attention_logit_cap": 10.0,
                "conf_conv_kernel_size": 3,
                "conf_reduction_factor": 2,
                "vocab_size": 100,
                "vocab_offset": 0,
            },
            "vision_soft_tokens_per_image": 16,
            "image_token_id": 1,
            "audio_soft_tokens_per_image": 16,
            "audio_token_id": 2,
        }
        keras_config = convert_gemma3n.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(keras_config["rope_theta"], 10000.0)

        # transformers >= 5 format
        text_cfg["rope_parameters"] = {"rope_theta": 20000.0}
        keras_config = convert_gemma3n.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(keras_config["rope_theta"], 20000.0)
