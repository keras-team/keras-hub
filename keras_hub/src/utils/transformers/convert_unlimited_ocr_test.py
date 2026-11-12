import json
import os
import tempfile

import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_backbone import (
    UnlimitedOCRBackbone,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_causal_lm import (
    UnlimitedOCRCausalLM,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_image_converter import (
    UnlimitedOCRImageConverter,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_tokenizer import (
    UnlimitedOCRTokenizer,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_unlimited_ocr

IMAGE_TOKEN_ID = 99
GLOBAL_SIZE = 64
LOCAL_SIZE = 40

# The `config.json` of `baidu/Unlimited-OCR`. It uses the original schema.
REAL_CONFIG = {
    "language_config": {
        "first_k_dense_replace": 1,
        "hidden_size": 1280,
        "intermediate_size": 6848,
        "moe_intermediate_size": 896,
        "n_routed_experts": 64,
        "n_shared_experts": 2,
        "num_attention_heads": 10,
        "num_experts_per_tok": 6,
        "num_hidden_layers": 12,
        "vocab_size": 129280,
        "sliding_window_size": 128,
    },
    "model_type": "unlimited_ocr",
    "projector_config": {"input_dim": 2048, "n_embed": 1280},
    "vision_config": {
        "image_size": 1024,
        "width": {
            "clip-l-14-224": {
                "heads": 16,
                "image_size": 224,
                "layers": 24,
                "patch_size": 14,
                "width": 1024,
            },
            "sam_vit_b": {
                "downsample_channels": [512, 1024],
                "global_attn_indexes": [2, 5, 8, 11],
                "heads": 12,
                "layers": 12,
                "width": 768,
            },
        },
    },
    "sliding_window": 128,
}

REAL_PREPROCESSOR_CONFIG = {
    "crop_to_patches": True,
    "min_patches": 2,
    "max_patches": 32,
    "tile_size": 640,
    "background_color": [127, 127, 127],
    "size": {"height": 1024, "width": 1024},
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "rescale_factor": 0.00392156862745098,
}

REAL_PROCESSOR_CONFIG = {"patch_size": 16, "downsample_ratio": 4}


def _write_json(dir_path, name, data):
    with open(os.path.join(dir_path, name), "w", encoding="utf-8") as f:
        json.dump(data, f)


def _build_tiny_hf_model(transformers):
    """Builds a tiny random native model.

    The config exercises every mechanism of the model. The SAM position and
    relative position tables resize for the 64 pixel global view and the 40
    pixel tile. The tile needs window padding. The CLIP position table
    resizes. The text decoder has a dense layer, a sparse layer, shared
    experts and a rotary base other than the default.
    """
    torch = pytest.importorskip("torch")
    config = transformers.UnlimitedOcrConfig(
        vision_config={
            "sam_config": {
                "hidden_size": 32,
                "output_channels": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "image_size": 128,
                "patch_size": 8,
                "window_size": 4,
                "global_attn_indexes": [1],
                "mlp_dim": 64,
                "downsample_channels": [24, 32],
            },
            "encoder_config": {
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "image_size": 48,
                "patch_size": 16,
            },
        },
        text_config={
            "vocab_size": 100,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "moe_intermediate_size": 16,
            "n_routed_experts": 4,
            "n_shared_experts": 2,
            "num_experts_per_tok": 2,
            "sliding_window": 4,
            "rope_parameters": {"rope_type": "default", "rope_theta": 20000.0},
        },
        image_token_id=IMAGE_TOKEN_ID,
    )
    torch.manual_seed(0)
    model = transformers.UnlimitedOcrForConditionalGeneration(config)
    # The default initialization leaves some tables at zero. Random values
    # make every weight matter.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            values = 0.1 * torch.randn_like(parameter)
            if parameter.ndim == 1 and name.endswith("weight"):
                values = values + 1.0
            parameter.copy_(values)
    return model.eval()


class TestUnlimitedOCRConverter(TestCase):
    def test_convert_backbone_config_original_schema(self):
        config = convert_unlimited_ocr.convert_backbone_config(REAL_CONFIG)
        vision_encoder = config.pop("vision_encoder")
        self.assertEqual(
            config,
            {
                "vocabulary_size": 129280,
                "num_layers": 12,
                "num_query_heads": 10,
                "hidden_dim": 1280,
                "intermediate_dim": 6848,
                "moe_intermediate_dim": 896,
                "num_shared_experts": 2,
                "num_experts": 64,
                "top_k": 6,
                "num_dense_layers": 1,
                "routed_scaling_factor": 1.0,
                "rope_max_wavelength": 10000.0,
                "layer_norm_epsilon": 1e-6,
                "sliding_window": 128,
                "image_token_id": 128815,
                "global_image_size": 1024,
            },
        )
        self.assertEqual(vision_encoder.sam_hidden_dim, 768)
        self.assertEqual(vision_encoder.sam_mlp_dim, 3072)
        self.assertEqual(vision_encoder.sam_window_size, 14)
        self.assertEqual(vision_encoder.sam_global_attn_indexes, (2, 5, 8, 11))
        self.assertEqual(vision_encoder.sam_downsample_channels, (512, 1024))
        self.assertEqual(vision_encoder.clip_hidden_dim, 1024)
        self.assertEqual(vision_encoder.clip_num_layers, 24)
        self.assertEqual(vision_encoder.clip_intermediate_dim, 4096)
        self.assertEqual(vision_encoder.clip_image_size, 224)
        self.assertEqual(vision_encoder.clip_patch_size, 14)

    def test_convert_backbone_config_native_schema(self):
        transformers_config = {
            "image_token_id": 7,
            "text_config": {
                "vocab_size": 100,
                "hidden_size": 32,
                "intermediate_size": 48,
                "num_hidden_layers": 3,
                "num_attention_heads": 4,
                "moe_intermediate_size": 16,
                "n_routed_experts": 4,
                "n_shared_experts": 2,
                "num_experts_per_tok": 2,
                "mlp_layer_types": ["dense", "dense", "sparse"],
                "rms_norm_eps": 1e-5,
                "rope_parameters": {"rope_theta": 20000.0},
                "sliding_window": 4,
            },
            "vision_config": {
                "sam_config": {
                    "hidden_size": 32,
                    "output_channels": 16,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "image_size": 128,
                    "patch_size": 8,
                    "window_size": 4,
                    "global_attn_indexes": [1],
                    "mlp_dim": 64,
                    "downsample_channels": [24, 32],
                },
                "encoder_config": {
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "image_size": 48,
                    "patch_size": 16,
                },
            },
        }
        config = convert_unlimited_ocr.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(config["num_dense_layers"], 2)
        self.assertEqual(config["rope_max_wavelength"], 20000.0)
        self.assertEqual(config["layer_norm_epsilon"], 1e-5)
        self.assertEqual(config["sliding_window"], 4)
        self.assertEqual(config["image_token_id"], 7)
        self.assertEqual(config["global_image_size"], 128)
        vision_encoder = config["vision_encoder"]
        self.assertEqual(vision_encoder.sam_patch_size, 8)
        self.assertEqual(vision_encoder.sam_mlp_dim, 64)
        self.assertEqual(vision_encoder.sam_global_attn_indexes, (1,))
        self.assertEqual(vision_encoder.clip_intermediate_dim, 64)

    def test_load_image_converter_config(self):
        with tempfile.TemporaryDirectory() as preset_dir:
            _write_json(
                preset_dir, "preprocessor_config.json", REAL_PREPROCESSOR_CONFIG
            )
            _write_json(
                preset_dir, "processor_config.json", REAL_PROCESSOR_CONFIG
            )
            config = convert_unlimited_ocr.load_image_converter_config(
                preset_dir, REAL_CONFIG
            )
        converter = UnlimitedOCRImageConverter(**config)
        self.assertEqual(converter.global_size, 1024)
        self.assertEqual(converter.tile_size, 640)
        self.assertEqual(converter.min_tiles, 2)
        self.assertEqual(converter.max_tiles, 32)
        self.assertTrue(converter.crop_to_patches)
        self.assertEqual(converter.background_color, (127, 127, 127))
        self.assertEqual(converter.patch_size, 16)
        self.assertEqual(converter.downsample_ratio, 4)
        # A 1024 pixel view gives 16x16 features, 16 newlines and 1 separator.
        self.assertEqual(converter.get_num_image_tokens(0, 0), 273)

    def test_convert_tokenizer(self):
        from tokenizers import pre_tokenizers

        alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
        vocab = {token: index + 3 for index, token in enumerate(alphabet)}
        vocab["Ġt"] = len(vocab) + 3
        specials = {
            "<｜begin▁of▁sentence｜>": 0,
            "<｜end▁of▁sentence｜>": 1,
            "<｜▁pad▁｜>": 2,
            "<image>": 300,
            "<|det|>": 301,
        }
        # The original `tokenizer.json` lists the base tokens, the special
        # tokens and the merges.
        tokenizer_json = {
            "model": {"vocab": vocab, "merges": [["Ġ", "t"]]},
            "added_tokens": [
                {"id": index, "content": token}
                for token, index in specials.items()
            ],
        }
        with tempfile.TemporaryDirectory() as preset_dir:
            _write_json(preset_dir, "tokenizer.json", tokenizer_json)
            tokenizer = convert_unlimited_ocr.convert_tokenizer(
                UnlimitedOCRTokenizer, preset_dir
            )
        self.assertEqual(tokenizer.start_token_id, 0)
        self.assertEqual(tokenizer.end_token_id, 1)
        self.assertEqual(tokenizer.pad_token_id, 2)
        self.assertEqual(tokenizer.token_to_id("<image>"), 300)
        output = [int(i) for i in tokenizer("a<image>b <|det|>")]
        self.assertIn(300, output)
        self.assertIn(301, output)
        self.assertEqual(
            tokenizer.detokenize(tokenizer("a<image>b <|det|>")),
            "a<image>b <|det|>",
        )

    @pytest.mark.large
    def test_class_detection(self):
        transformers = pytest.importorskip("transformers")
        model = _build_tiny_hf_model(transformers)
        with tempfile.TemporaryDirectory() as preset_dir:
            model.save_pretrained(preset_dir)
            backbone = Backbone.from_preset(
                preset_dir, load_weights=False, local_image_size=LOCAL_SIZE
            )
        self.assertIsInstance(backbone, UnlimitedOCRBackbone)

    @pytest.mark.large
    def test_converted_backbone_matches_hf(self):
        torch = pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")
        hf_model = _build_tiny_hf_model(transformers)

        with tempfile.TemporaryDirectory() as preset_dir:
            hf_model.save_pretrained(preset_dir)
            keras_backbone = UnlimitedOCRBackbone.from_preset(
                preset_dir,
                global_image_size=GLOBAL_SIZE,
                local_image_size=LOCAL_SIZE,
            )
        # Image 0 has two tiles in a row: 2 * (2 * 2 + 1) = 10 local tokens
        # and 7 global tokens. Image 1 has no tile: 7 tokens.
        rng = np.random.RandomState(0)
        token_ids = rng.randint(3, 90, size=(2, 22)).astype("int32")
        token_ids[0, 1:18] = IMAGE_TOKEN_ID
        token_ids[1, 2:9] = IMAGE_TOKEN_ID
        padding_mask = np.ones((2, 22), dtype="int32")
        # Right padding on the second row.
        padding_mask[1, 18:] = 0
        pixel_values = rng.uniform(
            -1, 1, size=(2, GLOBAL_SIZE, GLOBAL_SIZE, 3)
        ).astype("float32")
        pixel_values_local = rng.uniform(
            -1, 1, size=(2, LOCAL_SIZE, LOCAL_SIZE, 3)
        ).astype("float32")
        patches_grid = np.array([[2, 1], [0, 0]], dtype="int32")
        # The Keras backbone takes one row of tiles for each sample. The HF
        # model takes the tiles of all images in one array.
        keras_local = np.zeros((2, 2, LOCAL_SIZE, LOCAL_SIZE, 3), "float32")
        keras_local[0] = pixel_values_local

        keras_out = ops.convert_to_numpy(
            keras_backbone(
                {
                    "token_ids": token_ids,
                    "padding_mask": padding_mask,
                    "pixel_values": pixel_values[:, None],
                    "pixel_values_local": keras_local,
                    "patches_grid": patches_grid[:, None],
                }
            )
        )
        with torch.no_grad():
            hf_out = (
                hf_model.model(
                    input_ids=torch.tensor(token_ids, dtype=torch.long),
                    pixel_values=torch.tensor(pixel_values).permute(0, 3, 1, 2),
                    pixel_values_local=torch.tensor(pixel_values_local).permute(
                        0, 3, 1, 2
                    ),
                    num_local_patches=[2, 0],
                    patches_grid=[[2, 1], [0, 0]],
                    attention_mask=torch.tensor(padding_mask),
                )
                .last_hidden_state.detach()
                .cpu()
                .numpy()
            )
        self.assertEqual(keras_out.shape, hf_out.shape)
        # Compare the real tokens. Padding rows differ by design.
        valid = padding_mask.astype(bool)
        self.assertAllClose(keras_out[valid], hf_out[valid], atol=1e-4)

    @pytest.mark.large
    def test_converted_decode_matches_hf(self):
        torch = pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")
        hf_model = _build_tiny_hf_model(transformers)
        with tempfile.TemporaryDirectory() as preset_dir:
            hf_model.save_pretrained(preset_dir)
            keras_backbone = UnlimitedOCRBackbone.from_preset(
                preset_dir,
                global_image_size=GLOBAL_SIZE,
                local_image_size=LOCAL_SIZE,
            )
        task = UnlimitedOCRCausalLM(keras_backbone)

        prompt_length = 22
        num_steps = 10
        rng = np.random.RandomState(0)
        token_ids = rng.randint(3, 90, size=(1, prompt_length)).astype("int32")
        token_ids[0, 1:18] = IMAGE_TOKEN_ID
        pixel_values = rng.uniform(
            -1, 1, size=(1, GLOBAL_SIZE, GLOBAL_SIZE, 3)
        ).astype("float32")
        pixel_values_local = rng.uniform(
            -1, 1, size=(2, LOCAL_SIZE, LOCAL_SIZE, 3)
        ).astype("float32")
        patches_grid = np.array([[2, 1]], dtype="int32")

        # The decode steps run past the sliding window of 4 positions. HF
        # keeps the prefill keys and drops the oldest decode keys.
        with torch.no_grad():
            output = hf_model.generate(
                input_ids=torch.tensor(token_ids, dtype=torch.long),
                attention_mask=torch.ones(1, prompt_length, dtype=torch.long),
                pixel_values=torch.tensor(pixel_values).permute(0, 3, 1, 2),
                pixel_values_local=torch.tensor(pixel_values_local).permute(
                    0, 3, 1, 2
                ),
                num_local_patches=[2],
                patches_grid=[[2, 1]],
                max_new_tokens=num_steps,
                do_sample=False,
                suppress_tokens=[1],
                pad_token_id=0,
                output_logits=True,
                return_dict_in_generate=True,
            )
        hf_logits = torch.stack(output.logits, dim=0)[:, 0].numpy()
        generated = output.sequences[0, prompt_length:].numpy().astype("int32")
        self.assertEqual(hf_logits.shape[0], num_steps)

        # Feed the HF tokens one at a time. The first step recomputes the
        # last prompt token, as `generate_step` does.
        all_ids = np.concatenate([token_ids[0], generated])[None, :]
        image_features = keras_backbone.image_feature_extractor(
            pixel_values[:, None], pixel_values_local[None]
        )
        _, cache = task._build_cache(
            ops.convert_to_tensor(all_ids),
            image_features=image_features,
            patches_grid=patches_grid[:, None],
        )
        keras_logits = []
        for step in range(num_steps):
            index = prompt_length - 1 + step
            logits, _, cache = task.call_with_cache(
                ops.convert_to_tensor(all_ids[:, index : index + 1]),
                cache,
                index,
                prefill_length=ops.convert_to_tensor(
                    [prompt_length], dtype="int32"
                ),
            )
            keras_logits.append(ops.convert_to_numpy(logits)[0, 0])
        self.assertAllClose(np.stack(keras_logits), hf_logits, atol=1e-4)
