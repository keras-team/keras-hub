import json
import os

import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.embedding_gemma2.embedding_gemma2_backbone import (
    EmbeddingGemma2Backbone,
)
from keras_hub.src.models.embedding_gemma2.embedding_gemma2_text_embedder import (  # noqa: E501
    EmbeddingGemma2TextEmbedder,
)
from keras_hub.src.models.embedding_gemma2.embedding_gemma2_tokenizer import (
    EmbeddingGemma2Tokenizer,
)
from keras_hub.src.models.text_embedder import TextEmbedder
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers.convert_embedding_gemma2 import (
    convert_backbone_config,
)
from keras_hub.src.utils.transformers.convert_embedding_gemma2 import (
    convert_tokenizer,
)


def _tiny_text_config():
    return {
        "vocab_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "hidden_size": 16,
        "intermediate_size": 32,
        "head_dim": 8,
        "sliding_window": 4,
        "layer_types": ["sliding_attention", "full_attention"],
        "hidden_size_per_layer_input": 0,
        "embedding_dim": 8,
        "rms_norm_eps": 1e-6,
        "attention_dropout": 0.0,
        "per_layer_config": {},
        "rope_parameters": {
            "sliding_attention": {"rope_theta": 10000.0},
            "full_attention": {"rope_theta": 1000000.0},
        },
    }


class ConvertEmbeddingGemma2Test(TestCase):
    @pytest.mark.extra_large
    def test_backbone_from_hf_preset(self):
        model = EmbeddingGemma2Backbone.from_preset(
            "hf://google/embeddinggemma-2",
            load_weights=False,
        )
        self.assertEqual(model.vocabulary_size, 262144)
        self.assertEqual(model.hidden_dim, 512)
        self.assertEqual(model.num_layers, 24)
        self.assertEqual(model.embedding_dim, 768)

    @pytest.mark.extra_large
    def test_class_detection(self):
        preset_name = "hf://google/embeddinggemma-2"
        model = TextEmbedder.from_preset(
            preset_name,
            load_weights=False,
        )
        self.assertIsInstance(model, EmbeddingGemma2TextEmbedder)
        model = Backbone.from_preset(
            preset_name,
            load_weights=False,
        )
        self.assertIsInstance(model, EmbeddingGemma2Backbone)

    def test_audio_input_feat_size_follows_conv_channels(self):
        # HF sizes the SSCP projection as
        # `(conv_channels[0] // 4) * conv_channels[1]`; the Keras side must
        # derive the mel-bin count the same way rather than default to 128.
        text_cfg = _tiny_text_config()
        audio_cfg = {
            "hidden_size": 32,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "attention_chunk_size": 12,
            "attention_context_left": 13,
            "attention_context_right": 0,
            "attention_logit_cap": 50.0,
            "attention_invalid_logits_value": -1e9,
            "conv_kernel_size": 5,
            "residual_weight": 0.5,
            "gradient_clipping": 1e10,
            "subsampling_conv_channels": [8, 8],
            "output_proj_dims": 48,
            "rms_norm_eps": 1e-6,
        }
        kwargs = convert_backbone_config(
            {"text_config": text_cfg, "audio_config": audio_cfg}
        )
        audio_encoder = kwargs["audio_encoder"]
        self.assertEqual(audio_encoder.input_feat_size, 8)
        input_proj = audio_encoder.subsample_conv_projection.input_proj
        # (8 // 4) * 8 = 16 input features, projected to hidden_size 32.
        self.assertEqual(tuple(input_proj.kernel.shape), (16, 32))

    def test_backbone_config_reads_rope_and_audio_token_counts(self):
        text_cfg = _tiny_text_config()
        kwargs = convert_backbone_config({"text_config": text_cfg})
        # Absent in the HF config: keep the backbone defaults.
        self.assertEqual(kwargs["global_rope_partial_rotary_factor"], 1.0)
        self.assertIsNone(kwargs["num_audio_tokens_per_clip"])
        text_cfg["rope_parameters"]["full_attention"][
            "partial_rotary_factor"
        ] = 0.25
        kwargs = convert_backbone_config(
            {"text_config": text_cfg, "audio_soft_tokens_per_image": 188}
        )
        self.assertEqual(kwargs["global_rope_partial_rotary_factor"], 0.25)
        self.assertEqual(kwargs["num_audio_tokens_per_clip"], 188)

    def test_backbone_config_reads_per_layer_config(self):
        text_cfg = _tiny_text_config()
        entry = {"head_dim": 32, "num_key_value_heads": 2}
        text_cfg["per_layer_config"] = {"1": dict(entry), "3": dict(entry)}
        kwargs = convert_backbone_config({"text_config": text_cfg})
        self.assertEqual(kwargs["global_head_dim"], 32)
        self.assertEqual(kwargs["num_global_key_value_heads"], 2)
        text_cfg["per_layer_config"]["3"]["head_dim"] = 64
        with self.assertRaisesRegex(ValueError, "disagree"):
            convert_backbone_config({"text_config": text_cfg})

    def test_vision_encoder_follows_hf_vision_config(self):
        # Tier 1 parity found the encoder pooling with the default
        # `pool_size=3` while HF used `pooling_kernel_size=2`, so it emitted
        # 31 soft tokens where the preprocessor had placed 64.
        text_cfg = {
            "vocab_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "head_dim": 8,
            "sliding_window": 4,
            "layer_types": ["full_attention"],
            "hidden_size_per_layer_input": 0,
            "embedding_dim": 8,
            "rms_norm_eps": 1e-6,
            "attention_dropout": 0.0,
            "per_layer_config": {},
            "rope_parameters": {
                "sliding_attention": {"rope_theta": 10000.0},
                "full_attention": {"rope_theta": 1000000.0},
            },
        }
        vis_cfg = {
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "intermediate_size": 64,
            "patch_size": 16,
            "pooling_kernel_size": 2,
            "position_embedding_size": 1024,
            "rms_norm_eps": 1e-5,
            "rope_parameters": {"rope_theta": 100.0},
            "use_clipped_linears": False,
            "standardize": True,
        }
        kwargs = convert_backbone_config(
            {"text_config": text_cfg, "vision_config": vis_cfg}
        )
        cfg = kwargs["vision_encoder"].get_config()
        self.assertEqual(cfg["pool_size"], 2)
        self.assertEqual(cfg["num_key_value_heads"], 1)
        self.assertEqual(cfg["position_embedding_size"], 1024)
        self.assertEqual(cfg["layer_norm_epsilon"], 1e-5)
        self.assertFalse(cfg["use_clipped_linears"])
        self.assertTrue(cfg["standardize"])

    def test_tokenizer_added_tokens_are_atomic(self):
        # Tier 1 parity found `<|image|>` tokenizing as `<`,`|`,`image`,`|>`.
        # HF matches added_tokens before BPE, so they must survive as one id.
        base = ["<pad>", "<eos>", "<bos>", "<unk>", "<", "|", ">", "i", "m"]
        base += ["a", "g", "e", "u", "d", "o", "v", "▁", "<|image>"]
        base += ["<|image|>", "<image|>", "<|audio>", "<|audio|>"]
        base += ["<audio|>", "<|video|>"]
        # byte_fallback protos must carry all 256 byte pieces.
        base += [f"<0x{i:02X}>" for i in range(256)]
        vocab = {tok: i for i, tok in enumerate(base)}
        added = [
            {"id": vocab[t], "content": t, "special": True}
            for t in base
            if t.startswith("<") and t.endswith(">") and len(t) > 1
        ]
        preset = self.get_temp_dir()
        with open(os.path.join(preset, "tokenizer.json"), "w") as f:
            json.dump(
                {
                    "model": {"vocab": vocab, "merges": []},
                    "added_tokens": added,
                },
                f,
            )
        tokenizer = convert_tokenizer(EmbeddingGemma2Tokenizer, preset)
        ids = [int(t) for t in tokenizer("<|image|><|audio|><|video|>")]
        self.assertEqual(
            ids, [vocab["<|image|>"], vocab["<|audio|>"], vocab["<|video|>"]]
        )
