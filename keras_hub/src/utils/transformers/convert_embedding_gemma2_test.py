import json
import os
import tempfile

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
from keras_hub.src.utils.transformers.convert_embedding_gemma2 import (
    load_preprocessor_config,
)
from keras_hub.src.utils.transformers.convert_embedding_gemma2 import (
    load_task_config,
)
from keras_hub.src.utils.transformers.convert_embedding_gemma2 import (
    load_video_converter_config,
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


def _get_tokenizer_dict():
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
    return {
        "model": {"vocab": vocab, "merges": []},
        "added_tokens": added,
    }


class ConvertEmbeddingGemma2Test(TestCase):
    @pytest.mark.large
    def test_backbone_from_hf_preset(self):
        model = EmbeddingGemma2Backbone.from_preset(
            "hf://google/embeddinggemma-2",
            load_weights=False,
        )
        self.assertEqual(model.vocabulary_size, 262144)
        self.assertEqual(model.hidden_dim, 512)
        self.assertEqual(model.num_layers, 24)
        self.assertEqual(model.embedding_dim, 768)

    @pytest.mark.large
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

    def _write_sentence_transformers_files(
        self, modules, pooling=None, sentence_bert=None
    ):
        """A preset directory holding real sentence-transformers files."""
        preset = tempfile.mkdtemp(dir=self.get_temp_dir())
        with open(os.path.join(preset, "modules.json"), "w") as f:
            json.dump(modules, f)
        if pooling is not None:
            os.makedirs(os.path.join(preset, "1_Pooling"))
            with open(os.path.join(preset, "1_Pooling/config.json"), "w") as f:
                json.dump(pooling, f)
        if sentence_bert is not None:
            path = os.path.join(preset, "sentence_bert_config.json")
            with open(path, "w") as f:
                json.dump(sentence_bert, f)
        return preset

    def _write_tiny_hf_preset(self, processor_config=None, sentence_bert=None):
        preset = tempfile.mkdtemp(dir=self.get_temp_dir())
        with open(os.path.join(preset, "config.json"), "w") as f:
            json.dump(
                {
                    "model_type": "embedding_gemma2",
                    "architectures": ["EmbeddingGemma2Model"],
                    "text_config": _tiny_text_config(),
                },
                f,
            )
        with open(os.path.join(preset, "tokenizer.json"), "w") as f:
            json.dump(_get_tokenizer_dict(), f)
        if processor_config is not None:
            with open(os.path.join(preset, "processor_config.json"), "w") as f:
                json.dump(processor_config, f)
        if sentence_bert is not None:
            with open(
                os.path.join(preset, "sentence_bert_config.json"), "w"
            ) as f:
                json.dump(sentence_bert, f)
        return preset

    def test_reads_sentence_transformers_files(self):
        transformer = {"type": "sentence_transformers.models.Transformer"}
        normalize = {"type": "sentence_transformers.models.Normalize"}

        # No sentence-transformers files at all: nothing is overridden.
        self.assertEqual(load_task_config(self.get_temp_dir(), {}), {})

        # Modern export: `pooling_mode`, no Normalize module.
        preset = self._write_sentence_transformers_files(
            [transformer], pooling={"pooling_mode": "mean"}
        )
        self.assertEqual(
            load_task_config(preset, {}),
            {"normalize": False, "pooling_mode": "mean"},
        )

        # Legacy export: `pooling_mode_mean_tokens`, with Normalize.
        preset = self._write_sentence_transformers_files(
            [transformer, normalize],
            pooling={"pooling_mode_mean_tokens": True},
        )
        self.assertEqual(
            load_task_config(preset, {}),
            {"normalize": True, "pooling_mode": "mean"},
        )

        # Neither pooling key: default to mean.
        preset = self._write_sentence_transformers_files(
            [transformer, normalize], pooling={}
        )
        self.assertEqual(
            load_task_config(preset, {}),
            {"normalize": True, "pooling_mode": "mean"},
        )

        # `max_seq_length` becomes the preprocessor's `sequence_length`;
        # when the file lacks it the class default stays in force.
        preset = self._write_sentence_transformers_files(
            [transformer], sentence_bert={"max_seq_length": 2048}
        )
        self.assertEqual(
            load_preprocessor_config(preset, {})["sequence_length"], 2048
        )
        preset = self._write_sentence_transformers_files(
            [transformer], sentence_bert={"do_lower_case": False}
        )
        self.assertEqual(
            load_preprocessor_config(preset, {})["sequence_length"], 512
        )

    def test_backbone_config_reads_rope_and_audio_token_counts(self):
        text_cfg = _tiny_text_config()
        kwargs = convert_backbone_config({"text_config": text_cfg})
        # Absent in the HF config: keep the backbone defaults.
        self.assertEqual(kwargs["global_rope_partial_rotary_factor"], 1.0)
        text_cfg["rope_parameters"]["full_attention"][
            "partial_rotary_factor"
        ] = 0.25
        kwargs = convert_backbone_config({"text_config": text_cfg})
        self.assertEqual(kwargs["global_rope_partial_rotary_factor"], 0.25)

    def test_backbone_config_reads_per_layer_config(self):
        text_cfg = _tiny_text_config()
        # No `per_layer_config`: global layers use the base attention dims,
        # as HF does.
        kwargs = convert_backbone_config({"text_config": text_cfg})
        self.assertEqual(kwargs["global_head_dim"], text_cfg["head_dim"])
        self.assertEqual(
            kwargs["num_global_key_value_heads"],
            text_cfg["num_key_value_heads"],
        )
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
        tokenizer_dict = _get_tokenizer_dict()
        vocab = tokenizer_dict["model"]["vocab"]
        preset = self.get_temp_dir()
        with open(os.path.join(preset, "tokenizer.json"), "w") as f:
            json.dump(tokenizer_dict, f)
        tokenizer = convert_tokenizer(EmbeddingGemma2Tokenizer, preset)
        ids = [int(t) for t in tokenizer("<|image|><|audio|><|video|>")]
        self.assertEqual(
            ids, [vocab["<|image|>"], vocab["<|audio|>"], vocab["<|video|>"]]
        )

    def test_preprocessor_reads_max_seq_length(self):
        from keras_hub.src.models.embedding_gemma2.embedding_gemma2_text_embedder_preprocessor import (  # noqa: E501
            EmbeddingGemma2TextEmbedderPreprocessor,
        )

        preset = self._write_tiny_hf_preset(
            processor_config={}, sentence_bert={"max_seq_length": 256}
        )
        prep = EmbeddingGemma2TextEmbedderPreprocessor.from_preset(preset)
        self.assertEqual(prep.sequence_length, 256)
        embedder = EmbeddingGemma2TextEmbedder.from_preset(
            preset, load_weights=False
        )
        self.assertEqual(embedder.preprocessor.sequence_length, 256)

    def test_preprocessor_from_preset_without_processor_config(self):
        from keras_hub.src.models.embedding_gemma2.embedding_gemma2_text_embedder_preprocessor import (  # noqa: E501
            EmbeddingGemma2TextEmbedderPreprocessor,
        )

        preset = self._write_tiny_hf_preset(processor_config=None)
        self.assertIsNone(load_video_converter_config(preset, {}))
        # A vision model whose export lacks processor_config.json must also
        # degrade to None, not raise.
        self.assertIsNone(
            load_video_converter_config(preset, {"vision_config": {}})
        )
        prep = EmbeddingGemma2TextEmbedderPreprocessor.from_preset(preset)
        self.assertIsNone(prep.video_converter)
        self.assertIsNone(prep.image_converter)
        self.assertIsNone(prep.audio_converter)

    @pytest.mark.large
    def test_convert_weights_matches_tiny_hf_model(self):
        # Build a tiny HF `EmbeddingGemma2Model` with vision and audio
        # towers, save it, load it through the converter, and compare the
        # per-token hidden states for text, image, audio and mixed prompts.
        # `load_weights=False` elsewhere in this file never runs
        # `convert_weights`; this does.
        torch = pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")
        if not hasattr(transformers, "EmbeddingGemma2Config"):
            self.skipTest("transformers has no EmbeddingGemma2 yet")
        import numpy as np
        from keras import ops

        text_cfg = _tiny_text_config()
        # The special tokens must be real vocabulary ids for the tiny model.
        tokens = {
            "boi_token_id": 56,
            "eoi_token_id": 57,
            "image_token_id": 58,
            "video_token_id": 59,
            "boa_token_id": 60,
            "eoa_token_index": 61,
            "audio_token_id": 62,
        }
        text_cfg["pad_token_id"] = 0
        vision_cfg = {
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "intermediate_size": 32,
            # Patch / pooling geometry of the released checkpoint, so the
            # HF image processor defaults apply; only the widths are tiny.
            "patch_size": 16,
            "pooling_kernel_size": 3,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 100.0},
        }
        audio_cfg = {
            "hidden_size": 16,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "attention_chunk_size": 4,
            "attention_context_left": 5,
            "attention_context_right": 0,
            "attention_logit_cap": 50.0,
            "attention_invalid_logits_value": -1e9,
            "conv_kernel_size": 5,
            "residual_weight": 0.5,
            "gradient_clipping": 1e10,
            "subsampling_conv_channels": [4, 2],
            "output_proj_dims": 48,
            "rms_norm_eps": 1e-6,
        }
        config = transformers.EmbeddingGemma2Config(
            text_config=text_cfg,
            vision_config=vision_cfg,
            audio_config=audio_cfg,
            **tokens,
        )
        torch.manual_seed(0)
        hf_model = transformers.EmbeddingGemma2Model(config).eval()
        preset = self.get_temp_dir()
        hf_model.save_pretrained(preset)
        backbone = EmbeddingGemma2Backbone.from_preset(preset)

        rng = np.random.RandomState(0)
        batch = 2
        # Patches from the HF image processor go into both models, so this
        # compares the towers and the interleave, not the two resize paths.
        image_processor = transformers.Gemma4ImageProcessor(
            patch_size=16, pooling_kernel_size=3, max_soft_tokens=70
        )
        from PIL import Image

        hf_images = image_processor(
            images=[
                Image.fromarray(
                    rng.randint(0, 256, (96, 96, 3), dtype=np.uint8)
                )
                for _ in range(batch)
            ],
            return_tensors="pt",
        )
        n_img = int(hf_images["num_soft_tokens_per_image"][0])
        patch_dim = 3 * vision_cfg["patch_size"] ** 2
        mel_frames, num_mels = 16, backbone.audio_encoder.input_feat_size
        seq = n_img + 24
        n_aud = int(
            ops.convert_to_numpy(
                backbone.audio_encoder(
                    np.zeros((1, mel_frames, num_mels), "float32"),
                    np.ones((1, mel_frames), "bool"),
                )
            ).shape[1]
        )

        def prompt(image=False, audio=False):
            ids = rng.randint(1, 50, size=(seq,))
            pos = 1
            if image:
                ids[pos] = tokens["boi_token_id"]
                ids[pos + 1 : pos + 1 + n_img] = tokens["image_token_id"]
                ids[pos + 1 + n_img] = tokens["eoi_token_id"]
                pos += n_img + 2
            if audio:
                ids[pos] = tokens["boa_token_id"]
                ids[pos + 1 : pos + 1 + n_aud] = tokens["audio_token_id"]
                ids[pos + 1 + n_aud] = tokens["eoa_token_index"]
            return ids

        cases = {
            "text": (False, False),
            "image": (True, False),
            "audio": (False, True),
            "mixed": (True, True),
        }
        for name, (image, audio) in cases.items():
            token_ids = np.stack([prompt(image, audio) for _ in range(batch)])
            keras_inputs = {
                "token_ids": token_ids.astype("int32"),
                "padding_mask": np.ones_like(token_ids, dtype="int32"),
            }
            hf_inputs = {
                "input_ids": torch.tensor(token_ids),
                "attention_mask": torch.ones_like(torch.tensor(token_ids)),
            }
            if image:
                pixel_values = hf_images["pixel_values"]
                pos_ids = hf_images["image_position_ids"]
                keras_inputs["pixel_values"] = pixel_values.numpy()[:, None]
                keras_inputs["pixel_position_ids"] = pos_ids.numpy()[
                    :, None
                ].astype("int32")
                hf_inputs["pixel_values"] = pixel_values
                hf_inputs["image_position_ids"] = pos_ids
            else:
                keras_inputs["pixel_values"] = np.zeros(
                    (batch, 0, 0, patch_dim), "float32"
                )
                keras_inputs["pixel_position_ids"] = np.zeros(
                    (batch, 0, 0, 2), "int32"
                )
            if audio:
                mel = rng.uniform(-1, 1, (batch, mel_frames, num_mels)).astype(
                    "float32"
                )
                keras_inputs["audio_mel"] = mel[:, None]
                keras_inputs["audio_mel_mask"] = np.ones(
                    (batch, 1, mel_frames), "int32"
                )
                hf_inputs["input_features"] = torch.tensor(mel)
                hf_inputs["input_features_mask"] = torch.ones(
                    (batch, mel_frames), dtype=torch.bool
                )
            else:
                keras_inputs["audio_mel"] = np.zeros(
                    (batch, 0, 1, num_mels), "float32"
                )
                keras_inputs["audio_mel_mask"] = np.zeros(
                    (batch, 0, 1), "int32"
                )
            img_id, aud_id = tokens["image_token_id"], tokens["audio_token_id"]
            keras_inputs["vision_indices"] = np.stack(
                [np.flatnonzero(row == img_id) for row in token_ids]
            ).astype("int32")
            keras_inputs["audio_indices"] = np.stack(
                [np.flatnonzero(row == aud_id) for row in token_ids]
            ).astype("int32")

            keras_out = ops.convert_to_numpy(backbone(keras_inputs))
            with torch.no_grad():
                hf_out = hf_model(**hf_inputs).last_hidden_state.numpy()
            self.assertAllClose(keras_out, hf_out, atol=1e-5, msg=name)
