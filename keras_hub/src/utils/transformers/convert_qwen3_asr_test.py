import contextlib
import json
import os
import tempfile

import keras
import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.qwen3_asr.qwen3_asr_backbone import Qwen3ASRBackbone
from keras_hub.src.models.qwen3_asr.qwen3_asr_causal_lm import Qwen3ASRCausalLM
from keras_hub.src.models.qwen3_asr.qwen3_asr_tokenizer import Qwen3ASRTokenizer
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers.convert_qwen3_asr import (
    convert_backbone_config,
)
from keras_hub.src.utils.transformers.convert_qwen3_asr import convert_tokenizer
from keras_hub.src.utils.transformers.convert_qwen3_asr import (
    load_audio_converter_config,
)


@contextlib.contextmanager
def _full_precision_matmul():
    """Disable TF32 on GPU so parity tests compare float32 math accurately.

    Without disabling TF32, A100 JAX shows ~2.016e-3 drift vs PyTorch; with
    full-precision matmul, drift drops to <= 1.27e-6.
    """
    backend = keras.config.backend()
    if backend == "jax":
        import jax

        with jax.default_matmul_precision("highest"):
            yield
    elif backend == "tensorflow":
        import tensorflow as tf

        enabled = tf.config.experimental.tensor_float_32_execution_enabled()
        tf.config.experimental.enable_tensor_float_32_execution(False)
        try:
            yield
        finally:
            tf.config.experimental.enable_tensor_float_32_execution(enabled)
    elif backend == "torch":
        import torch

        matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
            torch.backends.cudnn.allow_tf32 = cudnn_tf32
    else:
        yield


def _write_tiny_hf_tokenizer_and_processor(preset_dir, vocab_size=100):
    vocab = {f"tok_{i}": i for i in range(vocab_size - 8)}
    vocab["tok_0tok_1"] = vocab.pop("tok_2")
    special_tokens = [
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<|audio_start|>",
        "<|audio_end|>",
        "<|audio_pad|>",
        "<asr_text>",
        "<|reserved_special_token_0|>",
    ]
    added_tokens = []
    for offset, tok in enumerate(special_tokens):
        idx = vocab_size - 8 + offset
        if not tok.startswith("<|reserved_special_token_"):
            vocab[tok] = idx
        added_tokens.append({"id": idx, "content": tok})
    tokenizer_json = {
        "model": {
            "vocab": vocab,
            "merges": [["tok_0", "tok_1"]],
        },
        "added_tokens": added_tokens,
    }
    with open(os.path.join(preset_dir, "tokenizer.json"), "w") as f:
        json.dump(tokenizer_json, f)

    processor_config = {
        "feature_extractor": {
            "feature_size": 20,
            "n_fft": 400,
            "hop_length": 160,
            "sampling_rate": 16000,
            "min_length": 8000,
            "n_window": 50,
        }
    }
    with open(os.path.join(preset_dir, "processor_config.json"), "w") as f:
        json.dump(processor_config, f)


class TestQwen3ASRConverter(TestCase):
    @pytest.mark.large
    def test_class_detection(self):
        model = Backbone.from_preset(
            "hf://Qwen/Qwen3-ASR-0.6B-hf",
            load_weights=False,
        )
        self.assertIsInstance(model, Qwen3ASRBackbone)
        self.assertEqual(model.hidden_dim, 1024)
        self.assertEqual(model.num_layers, 28)
        self.assertEqual(model.audio_encoder.num_layers, 18)

    @pytest.mark.large
    def test_load_audio_converter_config(self):
        config = load_audio_converter_config("hf://Qwen/Qwen3-ASR-0.6B-hf", {})
        self.assertEqual(
            config,
            {
                "num_mels": 128,
                "num_fft_bins": 400,
                "stride": 160,
                "sampling_rate": 16000,
                "min_length": 8000,
                "n_window": 50,
            },
        )

    def test_causal_lm_from_tiny_local_hf_preset(self):
        transformers = pytest.importorskip("transformers")

        vocab_size = 100
        text_cfg = transformers.Qwen3Config(
            vocab_size=vocab_size,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            tie_word_embeddings=True,
        )
        audio_cfg = transformers.Qwen3ASREncoderConfig(
            num_mel_bins=20,
            encoder_layers=2,
            encoder_attention_heads=2,
            encoder_ffn_dim=16,
            d_model=16,
            n_window=50,
            n_window_infer=200,
            max_position_embeddings=13,
            downsample_hidden_size=4,
            output_dim=16,
        )
        cfg = transformers.Qwen3ASRConfig(
            text_config=text_cfg,
            audio_config=audio_cfg,
            audio_token_id=vocab_size - 3,
            timestamp_token_id=vocab_size - 1,
            pad_token_id=0,
            eos_token_id=1,
            tie_word_embeddings=True,
        )
        hf_model = transformers.Qwen3ASRForConditionalGeneration(cfg).eval()
        with tempfile.TemporaryDirectory() as preset_dir:
            hf_model.save_pretrained(preset_dir)
            _write_tiny_hf_tokenizer_and_processor(
                preset_dir, vocab_size=vocab_size
            )
            model = Qwen3ASRCausalLM.from_preset(preset_dir)
            self.assertIsInstance(model, Qwen3ASRCausalLM)
            self.assertEqual(model.backbone.hidden_dim, 16)
            self.assertIsInstance(
                model.preprocessor.tokenizer, Qwen3ASRTokenizer
            )
            self.assertNotIn(
                "<|reserved_special_token_0|>",
                model.preprocessor.tokenizer.vocabulary,
            )

    def test_convert_backbone_config_variants(self):
        base_cfg = {
            "audio_token_id": 99,
            "tie_word_embeddings": False,
            "text_config": {
                "vocab_size": 100,
                "head_dim": 8,
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "intermediate_size": 24,
                "rms_norm_eps": 1e-6,
                "attention_dropout": 0.1,
                "rope_parameters": None,
                "rope_theta": 500000,
                "use_sliding_window": True,
                "sliding_window": 4096,
                "tie_word_embeddings": True,
            },
            "audio_config": {
                "num_mel_bins": 20,
                "encoder_layers": 2,
                "encoder_attention_heads": 2,
                "encoder_ffn_dim": 16,
                "d_model": 16,
                "n_window": 50,
                "n_window_infer": 200,
                "downsample_hidden_size": 4,
                "max_position_embeddings": 13,
            },
        }
        converted = convert_backbone_config(base_cfg)
        self.assertEqual(converted["rope_max_wavelength"], 500000)
        self.assertEqual(converted["sliding_window_size"], 4096)
        self.assertFalse(converted["tie_word_embeddings"])
        self.assertEqual(converted["dropout"], 0.1)

    def test_convert_tokenizer_merges_and_unsplittable_tokens(self):
        with tempfile.TemporaryDirectory() as preset_dir:
            _write_tiny_hf_tokenizer_and_processor(preset_dir, vocab_size=100)
            tok_default = convert_tokenizer(Qwen3ASRTokenizer, preset_dir)
            self.assertIn("<|im_start|>", tok_default.unsplittable_tokens)

            tok_custom = convert_tokenizer(
                Qwen3ASRTokenizer,
                preset_dir,
                unsplittable_tokens=["tok_0"],
            )
            self.assertIn("tok_0", tok_custom.unsplittable_tokens)

            # String merges format
            tok_path = os.path.join(preset_dir, "tokenizer.json")
            with open(tok_path) as f:
                data = json.load(f)
            data["model"]["merges"] = ["tok_0 tok_1"]
            with open(tok_path, "w") as f:
                json.dump(data, f)
            tok_str_merges = convert_tokenizer(Qwen3ASRTokenizer, preset_dir)
            self.assertEqual(tok_str_merges.merges, ["tok_0 tok_1"])

    def _run_parity_check(self, tie_word_embeddings=True):
        torch = pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")

        # Tiny configs
        vocab_size = 100
        audio_token_id = 99

        text_cfg = transformers.Qwen3Config(
            vocab_size=vocab_size,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            tie_word_embeddings=tie_word_embeddings,
            rms_norm_eps=1e-5,
        )
        audio_cfg = transformers.Qwen3ASREncoderConfig(
            num_mel_bins=20,
            encoder_layers=2,
            encoder_attention_heads=2,
            encoder_ffn_dim=16,
            d_model=16,
            n_window=50,
            n_window_infer=200,
            max_position_embeddings=13,
            downsample_hidden_size=4,
            output_dim=16,
        )
        cfg = transformers.Qwen3ASRConfig(
            text_config=text_cfg,
            audio_config=audio_cfg,
            audio_token_id=audio_token_id,
            timestamp_token_id=98,
            pad_token_id=0,
            eos_token_id=1,
            tie_word_embeddings=tie_word_embeddings,
        )

        torch.manual_seed(0)
        hf_model = transformers.Qwen3ASRForConditionalGeneration(cfg).eval()
        with torch.no_grad():
            for p in hf_model.parameters():
                if p.ndim == 1:
                    p.copy_(1.0 + 0.3 * torch.randn_like(p))
                elif p.ndim >= 2:
                    p.copy_(torch.randn_like(p) / (p.shape[-1] ** 0.5))

        # Two clips span multiple inference windows and have different lengths.
        batch_size = 2
        audio_lens = [250, 400]
        max_audio_len = max(audio_lens)
        np.random.seed(0)
        audio_mel_keras = np.random.rand(batch_size, max_audio_len, 20).astype(
            "float32"
        )
        audio_mask = np.zeros((batch_size, max_audio_len), dtype="int32")
        for i, audio_len in enumerate(audio_lens):
            audio_mask[i, :audio_len] = 1
        audio_mel_keras *= audio_mask[..., None]
        audio_mel_hf = np.transpose(audio_mel_keras, (0, 2, 1))

        audio_token_lens = [33, 52]
        token_sequences = [
            [1] + [audio_token_id] * token_len + [2]
            for token_len in audio_token_lens
        ]
        max_token_len = max(map(len, token_sequences))
        token_ids = np.zeros((batch_size, max_token_len), dtype="int32")
        padding_mask = np.zeros_like(token_ids)
        for i, token_sequence in enumerate(token_sequences):
            token_ids[i, : len(token_sequence)] = token_sequence
            padding_mask[i, : len(token_sequence)] = 1

        # Keras forward
        keras_inputs = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "audio_mel": audio_mel_keras,
            "audio_mask": audio_mask,
        }
        with _full_precision_matmul():
            with tempfile.TemporaryDirectory() as preset_dir:
                hf_model.save_pretrained(preset_dir)
                keras_model = Qwen3ASRBackbone.from_preset(preset_dir)
            keras_hidden = keras_model(keras_inputs)
            keras_out = ops.convert_to_numpy(
                keras_model.token_embedding(keras_hidden, reverse=True)
            )

        # HF forward
        with torch.no_grad():
            hf_out = (
                hf_model(
                    input_ids=torch.tensor(token_ids),
                    attention_mask=torch.tensor(padding_mask),
                    input_features=torch.tensor(audio_mel_hf),
                    input_features_mask=torch.tensor(audio_mask),
                )
                .logits.detach()
                .cpu()
                .numpy()
            )

        self.assertEqual(keras_out.shape, hf_out.shape)
        self.assertAllClose(keras_out, hf_out, atol=1e-5, rtol=1e-5)

    def test_numerical_parity(self):
        self._run_parity_check(tie_word_embeddings=True)
        self._run_parity_check(tie_word_embeddings=False)
