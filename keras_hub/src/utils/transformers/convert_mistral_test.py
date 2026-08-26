import base64
import json
import os
import tempfile

import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.mistral.mistral_backbone import MistralBackbone
from keras_hub.src.models.mistral.mistral_causal_lm import MistralCausalLM
from keras_hub.src.models.mistral.mistral_tokenizer import (
    MistralTekkenTokenizer,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_mistral


def _write_tekken_file(dir_path):
    """Write a tiny synthetic `tekken.json` for offline conversion tests."""
    vocab = [
        {
            "rank": i,
            "token_bytes": base64.b64encode(bytes([i])).decode(),
            "token_str": None,
        }
        for i in range(256)
    ]
    for rank, piece in [
        (256, b"th"),
        (257, b"the"),
        (258, b"in"),
        (259, b" t"),
        (260, b" th"),
    ]:
        vocab.append(
            {
                "rank": rank,
                "token_bytes": base64.b64encode(piece).decode(),
                "token_str": piece.decode("latin-1"),
            }
        )
    config = {
        "pattern": (
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
            r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|\p{N}| ?[^\s\p{L}\p{N}]+"
            r"[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        ),
        "num_vocab_tokens": 261,
        "default_vocab_size": 266,
        "default_num_special_tokens": 5,
        "version": "v7",
    }
    special_tokens = [
        {"rank": 0, "token_str": "<unk>", "is_control": True},
        {"rank": 1, "token_str": "<s>", "is_control": True},
        {"rank": 2, "token_str": "</s>", "is_control": True},
        {"rank": 3, "token_str": "<pad>", "is_control": True},
        {"rank": 4, "token_str": "[INST]", "is_control": True},
    ]
    path = os.path.join(dir_path, "tekken.json")
    with open(path, "w") as f:
        json.dump(
            {
                "config": config,
                "vocab": vocab,
                "special_tokens": special_tokens,
            },
            f,
        )
    return path


class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = MistralCausalLM.from_preset("hf://cosmo3769/tiny-mistral-test")
        prompt = "What is your favorite condiment?"
        model.generate([prompt], max_length=15)

    @pytest.mark.large
    def test_class_detection(self):
        model = CausalLM.from_preset(
            "hf://cosmo3769/tiny-mistral-test",
            load_weights=False,
        )
        self.assertIsInstance(model, MistralCausalLM)
        model = Backbone.from_preset(
            "hf://cosmo3769/tiny-mistral-test",
            load_weights=False,
        )
        self.assertIsInstance(model, MistralBackbone)

    @pytest.mark.large
    def test_explicit_head_dim_matches_hf(self):
        # Magistral-style config: `head_dim` is set explicitly and does not
        # equal `hidden_size // num_attention_heads`, and sliding window is
        # disabled. Build a small HF Mistral, convert, and check that the
        # keras-hub forward pass matches the HF reference to within the
        # precision of the fp16 hops used by the converter.
        torch = pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")

        cfg = transformers.MistralConfig(
            vocab_size=100,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=12,
            sliding_window=None,
            rope_theta=1_000_000.0,
            rms_norm_eps=1e-5,
        )
        self.assertNotEqual(
            cfg.head_dim, cfg.hidden_size // cfg.num_attention_heads
        )
        torch.manual_seed(0)
        hf_model = transformers.MistralForCausalLM(cfg).eval()

        with tempfile.TemporaryDirectory() as preset_dir:
            hf_model.save_pretrained(preset_dir)
            keras_backbone = MistralBackbone.from_preset(preset_dir)

        input_ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype="int32")
        padding = np.ones_like(input_ids)
        keras_out = ops.convert_to_numpy(
            keras_backbone({"token_ids": input_ids, "padding_mask": padding})
        )
        with torch.no_grad():
            hf_out = (
                hf_model.model(torch.tensor(input_ids))
                .last_hidden_state.detach()
                .cpu()
                .numpy()
            )
        self.assertEqual(keras_out.shape, hf_out.shape)
        # The converter stores weights in float16, so the parity bound is
        # dominated by fp16 quantization rather than implementation error.
        self.assertAllClose(keras_out, hf_out, atol=1e-2)

    @pytest.mark.large
    def test_convert_multimodal_preset_matches_hf(self):
        # Build a tiny Mistral3 (Pixtral vision tower + Mistral text model)
        # checkpoint and check that the converted `MistralBackbone` matches
        # HF's reference forward pass end to end, including the vision
        # tower, multimodal projector, and image/text embedding merge.
        torch = pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")

        text_config = transformers.MistralConfig(
            vocab_size=100,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            sliding_window=None,
            rope_theta=1_000_000.0,
            rms_norm_eps=1e-5,
        )
        vision_config = transformers.PixtralVisionConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_channels=3,
            image_size=16,
            patch_size=4,
            rope_parameters={"rope_theta": 10000.0},
        )
        config = transformers.Mistral3Config(
            text_config=text_config,
            vision_config=vision_config,
            image_token_index=10,
            spatial_merge_size=2,
        )
        torch.manual_seed(0)
        hf_model = transformers.Mistral3ForConditionalGeneration(config).eval()

        with tempfile.TemporaryDirectory() as preset_dir:
            hf_model.save_pretrained(preset_dir)
            keras_backbone = MistralBackbone.from_preset(preset_dir)

        self.assertFalse(keras_backbone.text_only_model)
        self.assertIsNotNone(keras_backbone.vision_encoder)

        # A single 16x16 image with a 4x4 patch size produces a 4x4 patch
        # grid (16 patches); a spatial merge size of 2 merges these into a
        # 2x2 grid of 4 tokens, so 4 placeholder tokens at id 10 are needed.
        input_ids = np.array([[1, 10, 10, 10, 10, 3, 4]], dtype="int32")
        padding_mask = np.ones_like(input_ids)
        pixel_values = np.random.rand(1, 3, 16, 16).astype("float32")
        image_sizes = np.array([[16, 16]], dtype="int32")
        placeholder_indices = np.array([[1, 2, 3, 4]], dtype="int32")

        keras_out = ops.convert_to_numpy(
            keras_backbone(
                {
                    "token_ids": input_ids,
                    "padding_mask": padding_mask,
                    "pixel_values": pixel_values,
                    "image_sizes": image_sizes,
                    "placeholder_indices": placeholder_indices,
                }
            )
        )
        with torch.no_grad():
            hf_out = (
                hf_model.model(
                    input_ids=torch.tensor(input_ids),
                    attention_mask=torch.tensor(padding_mask),
                    pixel_values=torch.tensor(pixel_values),
                    image_sizes=torch.tensor(image_sizes),
                )
                .last_hidden_state.detach()
                .cpu()
                .numpy()
            )
        self.assertEqual(keras_out.shape, hf_out.shape)
        # fp16 weight storage dominates the parity bound, as in the
        # text-only converter test above.
        self.assertAllClose(keras_out, hf_out, atol=1e-2)

    def test_convert_backbone_config_detects_mistral3(self):
        from keras_hub.src.models.mistral.mistral_vision_encoder import (
            Mistral3MultiModalProjector,
        )
        from keras_hub.src.models.mistral.mistral_vision_encoder import (
            Mistral3VisionEncoder,
        )

        transformers_config = {
            "text_config": {
                "vocab_size": 100,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "hidden_size": 32,
                "intermediate_size": 48,
                "num_key_value_heads": 2,
                "rope_theta": 1_000_000.0,
                "rms_norm_eps": 1e-5,
                "sliding_window": None,
            },
            "vision_config": {
                "hidden_size": 16,
                "intermediate_size": 24,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_channels": 3,
                "image_size": 32,
                "patch_size": 8,
                "hidden_act": "gelu",
                "rope_parameters": {"rope_theta": 10000.0},
            },
            "image_token_index": 10,
            "spatial_merge_size": 2,
        }
        keras_config = convert_mistral.convert_backbone_config(
            transformers_config
        )
        self.assertIsInstance(
            keras_config["vision_encoder"], Mistral3VisionEncoder
        )
        self.assertIsInstance(
            keras_config["multimodal_projector"],
            Mistral3MultiModalProjector,
        )
        self.assertEqual(keras_config["image_token_index"], 10)
        self.assertEqual(keras_config["rope_max_wavelength"], 1_000_000.0)
        self.assertEqual(
            keras_config["vision_encoder"].get_config()["rope_theta"],
            10000.0,
        )

    def test_convert_backbone_config_rope_theta(self):
        # transformers < 5 format
        transformers_config = {
            "vocab_size": 100,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_key_value_heads": 2,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
            "sliding_window": 4096,
        }
        keras_config = convert_mistral.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(keras_config["rope_max_wavelength"], 10000.0)

        # transformers >= 5 format
        transformers_config = {
            "vocab_size": 100,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_key_value_heads": 2,
            "rope_parameters": {"rope_theta": 20000.0},
            "rms_norm_eps": 1e-5,
            "sliding_window": 4096,
        }
        # In the real transformers >= 5, rope_theta might still be present at
        # top level for some models, but the source of truth moved to
        # rope_parameters.
        keras_config = convert_mistral.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(keras_config["rope_max_wavelength"], 20000.0)

    def test_convert_tekken_tokenizer(self):
        pytest.importorskip("mistral_common")
        with tempfile.TemporaryDirectory() as dir_path:
            path = _write_tekken_file(dir_path)
            vocabulary, merges, split_pattern, control_tokens = (
                convert_mistral._convert_tekken_tokenizer(path)
            )
        # 5 special tokens + 256 bytes + 5 merges.
        self.assertEqual(len(vocabulary), 266)
        self.assertEqual(len(merges), 5)
        # Special tokens keep their reserved rank as id.
        self.assertEqual(vocabulary["<unk>"], 0)
        self.assertEqual(vocabulary["<s>"], 1)
        self.assertEqual(vocabulary["</s>"], 2)
        self.assertEqual(vocabulary["<pad>"], 3)
        self.assertEqual(len(control_tokens), 5)
        self.assertIn("<unk>", control_tokens)

        tokenizer = MistralTekkenTokenizer(
            vocabulary=vocabulary,
            merges=merges,
            split_pattern=split_pattern,
            control_tokens=control_tokens,
        )
        self.assertEqual(tokenizer.start_token_id, 1)
        self.assertEqual(tokenizer.end_token_id, 2)
        self.assertEqual(tokenizer.pad_token_id, 0)
        self.assertEqual(tokenizer.vocabulary_size(), 266)
        output = tokenizer("the tin")
        self.assertEqual(tokenizer.detokenize(output), "the tin")
