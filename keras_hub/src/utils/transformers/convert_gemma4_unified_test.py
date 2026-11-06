from keras_hub.src.models.gemma4_unified.gemma4_unified_backbone import (
    Gemma4UnifiedBackbone,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers.convert_gemma4_unified import (
    convert_backbone_config,
)
from keras_hub.src.utils.transformers.convert_gemma4_unified import (
    convert_weights,
)


class FakeLoader:
    def port_weight(self, keras_variable, hf_weight_key, hook_fn=None):
        pass


class TestConvertGemma4Unified(TestCase):
    def test_loader_entry_points_present(self):
        # Every attribute `TransformersPresetLoader` resolves on a converter
        # module. The de-duplication imports shared helpers from
        # `convert_gemma4`; a dropped re-export only fails at `from_preset`.
        from keras_hub.src.utils.transformers import convert_gemma4_unified

        for name in (
            "backbone_cls",
            "convert_backbone_config",
            "convert_weights",
            "convert_tokenizer",
            "load_image_converter_config",
            "load_audio_converter_config",
            "load_video_converter_config",
            "load_preprocessor_config",
        ):
            self.assertTrue(
                callable(getattr(convert_gemma4_unified, name, None)),
                f"convert_gemma4_unified.{name} is missing",
            )

    def test_convert_weights_fake_loader(self):
        config = {
            "model_type": "gemma4_unified",
            "text_config": {
                "vocab_size": 100,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "hidden_size": 32,
                "intermediate_size": 64,
                "head_dim": 16,
                "global_head_dim": 16,
                "num_global_key_value_heads": 1,
                "layer_types": ["full_attention", "full_attention"],
                "attention_k_eq_v": True,
                "rope_parameters": {
                    "full_attention": {
                        "rope_type": "proportional",
                        "rope_theta": 1000000.0,
                        "partial_rotary_factor": 0.25,
                    },
                    "sliding_attention": {
                        "rope_type": "default",
                        "rope_theta": 10000.0,
                    },
                },
            },
            "vision_config": {
                "mm_embed_dim": 32,
                "model_patch_size": 48,
                "mm_posemb_size": 4,
                "num_soft_tokens": 16,
                "pooling_kernel_size": 3,
                "patch_size": 16,
            },
            "audio_config": {
                "audio_embed_dim": 32,
            },
        }
        kwargs = convert_backbone_config(config)
        backbone = Gemma4UnifiedBackbone(**kwargs)
        loader = FakeLoader()
        convert_weights(backbone, loader, config)
