import json
import os

import keras
import numpy as np
import pytest

from keras_hub.src.models.audio_to_text import AudioToText
from keras_hub.src.models.audio_to_text_preprocessor import (
    AudioToTextPreprocessor,
)
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.whisper.whisper_audio_to_text import (
    WhisperAudioToText,
)
from keras_hub.src.models.whisper.whisper_audio_to_text_preprocessor import (
    WhisperAudioToTextPreprocessor,
)
from keras_hub.src.models.whisper.whisper_backbone import WhisperBackbone
from keras_hub.src.models.whisper.whisper_tokenizer import WhisperTokenizer
from keras_hub.src.tests.mocks.mock_whisper import MERGES
from keras_hub.src.tests.mocks.mock_whisper import _bpe_vocabulary
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_whisper

LANGUAGE_TOKENS = ["<|en|>", "<|fr|>"]
# The tokens Hugging Face adds after `<|endoftext|>`, in id order, including
# three timestamp tokens.
ADDED_TOKENS = (
    ["<|startoftranscript|>"]
    + LANGUAGE_TOKENS
    + [
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nocaptions|>",
        "<|notimestamps|>",
        "<|0.00|>",
        "<|0.02|>",
        "<|0.04|>",
    ]
)
SUPPRESS_TOKENS = [3, 7]


def _token_ids():
    """The tiny checkpoint's ids: BPE `0..29`, `<|endoftext|>` 30, then the
    added tokens from 31."""
    vocabulary = _bpe_vocabulary()
    vocabulary["<|endoftext|>"] = len(vocabulary)
    added = {token: len(vocabulary) + i for i, token in enumerate(ADDED_TOKENS)}
    return vocabulary, added


def _hf_config(multilingual, **overrides):
    vocabulary, added = _token_ids()
    vocabulary_size = len(vocabulary) + len(added)
    if multilingual:
        # Like the released checkpoints, a multilingual vocabulary is one
        # token larger than the English-only one.
        vocabulary_size += 1
    config = {
        "model_type": "whisper",
        "vocab_size": vocabulary_size,
        "d_model": 16,
        "encoder_layers": 2,
        "decoder_layers": 1,
        "encoder_attention_heads": 2,
        "decoder_attention_heads": 4,
        "encoder_ffn_dim": 32,
        "decoder_ffn_dim": 8,
        "num_mel_bins": 80,
        "max_source_positions": 50,
        "max_target_positions": 10,
        "dropout": 0.0,
        "decoder_start_token_id": added["<|startoftranscript|>"],
        "eos_token_id": vocabulary["<|endoftext|>"],
        "pad_token_id": vocabulary["<|endoftext|>"],
        "bos_token_id": vocabulary["<|endoftext|>"],
    }
    config.update(overrides)
    return config


def _generation_config(multilingual):
    vocabulary, added = _token_ids()
    config = {
        "decoder_start_token_id": added["<|startoftranscript|>"],
        "eos_token_id": vocabulary["<|endoftext|>"],
        "pad_token_id": vocabulary["<|endoftext|>"],
        "bos_token_id": vocabulary["<|endoftext|>"],
        "is_multilingual": multilingual,
        "no_timestamps_token_id": added["<|notimestamps|>"],
        "suppress_tokens": SUPPRESS_TOKENS,
        "begin_suppress_tokens": [vocabulary["Ġ"], vocabulary["<|endoftext|>"]],
        "max_length": 10,
    }
    if multilingual:
        config["lang_to_id"] = {
            token: added[token] for token in LANGUAGE_TOKENS
        }
        config["task_to_id"] = {
            "translate": added["<|translate|>"],
            "transcribe": added["<|transcribe|>"],
        }
        config["forced_decoder_ids"] = [
            [1, added["<|fr|>"]],
            [2, added["<|translate|>"]],
        ]
    else:
        config["forced_decoder_ids"] = [[1, added["<|notimestamps|>"]]]
    return config


def _feature_extractor_config(**overrides):
    config = {
        "feature_size": 80,
        "n_fft": 400,
        "hop_length": 160,
        "sampling_rate": 16000,
        "chunk_length": 1,
    }
    config.update(overrides)
    return config


def _write_json(preset, name, data):
    with open(os.path.join(preset, name), "w", encoding="utf-8") as f:
        json.dump(data, f)


def _write_json_preset(
    preset, multilingual=False, tokenizer_json=False, processor_config=False
):
    """Write the JSON files of a tiny Whisper checkpoint, without weights."""
    vocabulary, added = _token_ids()
    _write_json(preset, "config.json", _hf_config(multilingual))
    _write_json(
        preset, "generation_config.json", _generation_config(multilingual)
    )
    if tokenizer_json:
        _write_json(
            preset,
            "tokenizer.json",
            {
                "model": {
                    "vocab": vocabulary,
                    # `tokenizers` writes merges as pairs.
                    "merges": [merge.split(" ") for merge in MERGES],
                },
                "added_tokens": [
                    {"id": token_id, "content": token, "special": True}
                    for token, token_id in added.items()
                ],
            },
        )
    else:
        _write_json(preset, "vocab.json", vocabulary)
        with open(os.path.join(preset, "merges.txt"), "w") as f:
            f.write("#version: 0.2\n" + "\n".join(MERGES) + "\n")
        _write_json(preset, "added_tokens.json", added)
    if processor_config:
        _write_json(
            preset,
            "processor_config.json",
            {"feature_extractor": _feature_extractor_config(feature_size=128)},
        )
    else:
        _write_json(
            preset, "preprocessor_config.json", _feature_extractor_config()
        )


class TestWhisperConverter(TestCase):
    def test_convert_backbone_config(self):
        hf_config = _hf_config(multilingual=False, dropout=0.1)
        config = convert_whisper.convert_backbone_config(hf_config)
        self.assertEqual(config["num_layers"], 2)
        self.assertEqual(config["decoder_num_layers"], 1)
        self.assertEqual(config["num_heads"], 2)
        self.assertEqual(config["decoder_num_heads"], 4)
        self.assertEqual(config["intermediate_dim"], 32)
        self.assertEqual(config["decoder_intermediate_dim"], 8)
        self.assertEqual(config["dropout"], 0.1)
        # The Keras encoder position embedding is twice as long.
        self.assertEqual(config["max_encoder_sequence_length"], 100)
        self.assertEqual(config["max_decoder_sequence_length"], 10)
        backbone = WhisperBackbone(**config)
        self.assertEqual(len(backbone.decoder_transformer_layers), 1)

    def test_convert_tokenizer_english_only(self):
        preset = self.get_temp_dir()
        _write_json_preset(preset)
        tokenizer = convert_whisper.convert_tokenizer(WhisperTokenizer, preset)
        _, added = _token_ids()
        # The whole vocabulary is kept, added tokens included.
        self.assertEqual(tokenizer.vocabulary_size(), 43)
        self.assertEqual(tokenizer.language_tokens, None)
        self.assertEqual(tokenizer.bos_token_id, added["<|startoftranscript|>"])
        self.assertEqual(tokenizer.eos_token_id, 30)
        self.assertEqual(
            tokenizer.no_timestamps_token_id, added["<|notimestamps|>"]
        )
        # Control tokens are atomic, timestamp tokens are not special.
        self.assertAllEqual(
            tokenizer("<|notimestamps|> airport"),
            [added["<|notimestamps|>"], 23, 16],
        )
        self.assertNotIn("<|0.00|>", tokenizer.special_tokens)

    def test_convert_tokenizer_multilingual(self):
        preset = self.get_temp_dir()
        _write_json_preset(preset, multilingual=True)
        tokenizer = convert_whisper.convert_tokenizer(WhisperTokenizer, preset)
        _, added = _token_ids()
        self.assertEqual(
            tokenizer.language_tokens,
            {token: added[token] for token in LANGUAGE_TOKENS},
        )
        self.assertAllEqual(
            tokenizer("<|fr|> airport"), [added["<|fr|>"], 23, 16]
        )

    def test_convert_tokenizer_from_tokenizer_json(self):
        preset = self.get_temp_dir()
        _write_json_preset(preset, multilingual=True, tokenizer_json=True)
        tokenizer = convert_whisper.convert_tokenizer(WhisperTokenizer, preset)
        _, added = _token_ids()
        self.assertEqual(tokenizer.vocabulary_size(), 43)
        self.assertEqual(len(tokenizer.language_tokens), 2)
        self.assertAllEqual(
            tokenizer("<|fr|> airport"), [added["<|fr|>"], 23, 16]
        )

    def test_convert_tokenizer_language_tokens_from_vocabulary(self):
        # Without `lang_to_id`, the language tokens are the ids between
        # `<|startoftranscript|>` and `<|translate|>`.
        preset = self.get_temp_dir()
        _write_json_preset(preset, multilingual=True)
        generation_config = _generation_config(multilingual=True)
        del generation_config["lang_to_id"]
        _write_json(preset, "generation_config.json", generation_config)
        tokenizer = convert_whisper.convert_tokenizer(WhisperTokenizer, preset)
        self.assertEqual(sorted(tokenizer.language_tokens), LANGUAGE_TOKENS)

    def test_load_audio_converter_config(self):
        preset = self.get_temp_dir()
        _write_json_preset(preset)
        config = convert_whisper.load_audio_converter_config(preset, {})
        self.assertEqual(
            config,
            {
                "num_mels": 80,
                "num_fft_bins": 400,
                "stride": 160,
                "sampling_rate": 16000,
                "max_audio_length": 1,
            },
        )

    def test_load_audio_converter_config_from_processor_config(self):
        # `whisper-large-v3` uses 128 mel bins, and `transformers` v5 saves
        # the feature extractor inside `processor_config.json`.
        preset = self.get_temp_dir()
        _write_json_preset(preset, processor_config=True)
        config = convert_whisper.load_audio_converter_config(preset, {})
        self.assertEqual(config["num_mels"], 128)

    def test_load_preprocessor_config(self):
        preset = self.get_temp_dir()
        _write_json_preset(preset)
        hf_config = _hf_config(multilingual=False)
        self.assertEqual(
            convert_whisper.load_preprocessor_config(preset, hf_config),
            {"decoder_sequence_length": 10},
        )
        preset = self.get_temp_dir()
        _write_json_preset(preset, multilingual=True)
        hf_config = _hf_config(multilingual=True)
        self.assertEqual(
            convert_whisper.load_preprocessor_config(preset, hf_config),
            {
                "decoder_sequence_length": 10,
                "language": "<|fr|>",
                "task": "translate",
            },
        )

    def test_load_task_config(self):
        preset = self.get_temp_dir()
        _write_json_preset(preset)
        hf_config = _hf_config(multilingual=False)
        self.assertEqual(
            convert_whisper.load_task_config(preset, hf_config),
            {"suppress_tokens": [3, 7], "begin_suppress_tokens": [20, 30]},
        )
        # Older checkpoints only carry suppression in `config.json`.
        os.remove(os.path.join(preset, "generation_config.json"))
        hf_config["suppress_tokens"] = [5]
        self.assertEqual(
            convert_whisper.load_task_config(preset, hf_config),
            {"suppress_tokens": [5]},
        )

    def test_preprocessor_from_preset(self):
        preset = self.get_temp_dir()
        _write_json_preset(preset, multilingual=True)
        preprocessor = AudioToTextPreprocessor.from_preset(preset)
        self.assertIsInstance(preprocessor, WhisperAudioToTextPreprocessor)
        _, added = _token_ids()
        self.assertEqual(preprocessor.decoder_sequence_length, 10)
        self.assertEqual(
            preprocessor.decoder_prefix_ids,
            [
                added["<|startoftranscript|>"],
                added["<|fr|>"],
                added["<|translate|>"],
                added["<|notimestamps|>"],
            ],
        )

    @pytest.mark.large
    def test_convert_tiny_model_matches_hf(self):
        torch = pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")
        atol = 1e-2 if torch.cuda.is_available() else 1e-5

        for multilingual in (False, True):
            preset = self.get_temp_dir()
            _write_json_preset(preset, multilingual=multilingual)
            hf_config = transformers.WhisperConfig(
                **_hf_config(multilingual=multilingual)
            )
            torch.manual_seed(0)
            hf_model = transformers.WhisperForConditionalGeneration(hf_config)
            hf_model = hf_model.eval()
            hf_model.generation_config = transformers.GenerationConfig(
                **_generation_config(multilingual)
            )
            hf_model.save_pretrained(preset)
            feature_extractor = transformers.WhisperFeatureExtractor(
                **_feature_extractor_config()
            )
            model = WhisperAudioToText.from_preset(preset)
            preprocessor = model.preprocessor

            # One full clip and one that is mostly silence.
            rng = np.random.default_rng(0)
            audio = (rng.standard_normal((2, 16000)) * 0.1).astype("float32")
            audio[1, 4000:] = 0.0
            hf_features = feature_extractor(
                list(audio), sampling_rate=16000, return_tensors="pt"
            ).input_features
            features = preprocessor.audio_converter(audio)
            self.assertAllClose(
                keras.ops.transpose(features, (0, 2, 1)),
                hf_features.numpy(),
                atol=atol,
            )

            # Teacher-forced logits for the whole prefix and three tokens.
            x = preprocessor.generate_preprocess(
                {"audio": audio, "text": [" airplane at"] * 2},
                sequence_length=len(preprocessor.decoder_prefix_ids) + 3,
            )
            with torch.no_grad():
                hf_logits = hf_model(
                    input_features=hf_features,
                    decoder_input_ids=torch.from_numpy(
                        keras.ops.convert_to_numpy(x["decoder_token_ids"])
                    ).long(),
                ).logits
            self.assertAllClose(model(x), hf_logits.numpy(), atol=atol)

            # Greedy decoding with suppression matches `generate`.
            prefix = preprocessor.decoder_prefix_ids
            max_new_tokens = 5
            generate_kwargs = {}
            if multilingual:
                generate_kwargs = {"language": "fr", "task": "translate"}
            with torch.no_grad():
                hf_ids = hf_model.generate(
                    hf_features,
                    max_new_tokens=max_new_tokens,
                    **generate_kwargs,
                )
            hf_ids = hf_ids.numpy()[:, -max_new_tokens:]
            model.preprocessor = None
            inputs = preprocessor.generate_preprocess(
                {"audio": audio},
                sequence_length=len(prefix) + max_new_tokens,
            )
            outputs = model.generate(
                inputs, stop_token_ids=[preprocessor.tokenizer.end_token_id]
            )
            token_ids = keras.ops.convert_to_numpy(outputs["decoder_token_ids"])
            self.assertAllEqual(token_ids[:, : len(prefix)], [prefix] * 2)
            self.assertAllEqual(token_ids[:, len(prefix) :], hf_ids)

    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = WhisperAudioToText.from_preset("hf://openai/whisper-tiny.en")
        audio = keras.random.normal((1, 16000))
        model.generate({"audio": audio}, max_length=10)

    @pytest.mark.large
    def test_class_detection(self):
        model = AudioToText.from_preset(
            "hf://openai/whisper-tiny.en",
            load_weights=False,
        )
        self.assertIsInstance(model, WhisperAudioToText)
        model = Backbone.from_preset(
            "hf://openai/whisper-tiny.en",
            load_weights=False,
        )
        self.assertIsInstance(model, WhisperBackbone)
