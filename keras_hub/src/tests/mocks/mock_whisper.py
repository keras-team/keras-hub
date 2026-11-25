"""Tiny Whisper tokenizer and audio converter shared by the Whisper tests."""

import numpy as np

from keras_hub.src.models.whisper.whisper_audio_converter import (
    WhisperAudioConverter,
)
from keras_hub.src.models.whisper.whisper_tokenizer import WhisperTokenizer

MERGES = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
MERGES += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
MERGES += ["Ġai r", "Ġa i", "pla ne"]

# Control token names, in the order their ids follow the BPE vocabulary.
CONTROL_TOKENS = [
    "<|endoftext|>",
    "<|startoftranscript|>",
    "<|notimestamps|>",
    "<|transcribe|>",
    "<|translate|>",
]
LANGUAGE_TOKENS = ["<|en|>", "<|fr|>"]


def _bpe_vocabulary():
    vocabulary = []
    for merge in MERGES:
        a, b = merge.split(" ")
        vocabulary.extend([a, b, a + b])
    vocabulary = sorted(set(vocabulary)) + ["!"]
    return {token: i for i, token in enumerate(vocabulary)}


def mock_whisper_tokenizer(multilingual=False, **kwargs):
    """A Whisper tokenizer with contiguous ids.

    The BPE tokens take ids `0..29`, the five control tokens `30..34`, and,
    when `multilingual=True`, the language tokens `<|en|>` and `<|fr|>` take
    `35` and `36`.
    """
    vocabulary = _bpe_vocabulary()
    special_tokens = {
        token: len(vocabulary) + i for i, token in enumerate(CONTROL_TOKENS)
    }
    language_tokens = None
    if multilingual:
        offset = len(vocabulary) + len(special_tokens)
        language_tokens = {
            token: offset + i for i, token in enumerate(LANGUAGE_TOKENS)
        }
    return WhisperTokenizer(
        vocabulary=vocabulary,
        merges=MERGES,
        special_tokens=special_tokens,
        language_tokens=language_tokens,
        **kwargs,
    )


def mock_whisper_audio_converter(**kwargs):
    """A Whisper audio converter for one second of 16 kHz audio."""
    return WhisperAudioConverter(
        num_mels=80,
        num_fft_bins=400,
        stride=160,
        sampling_rate=16000,
        max_audio_length=1,
        **kwargs,
    )


def mock_audio(shape, seed=42):
    """Seeded random float32 audio of the given shape."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype("float32")
