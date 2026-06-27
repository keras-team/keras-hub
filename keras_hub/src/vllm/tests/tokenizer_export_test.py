"""CPU unit tests for HF tokenizer export in setup_vllm_model.

Uses a tiny fake KerasHub byte-level BPE tokenizer so no preset download is
needed. Importing registry pulls the adapter (torch/jax), so we skip cleanly if
those aren't installed.
"""

import json
import os
import tempfile

import pytest

# registry imports the adapter, which needs torch + jax.
pytest.importorskip("torch")
pytest.importorskip("jax")

from keras_hub.src.vllm.registry import _export_hf_tokenizer  # noqa: E402


class _FakeBPETokenizer:
    """Minimal stand-in for a KerasHub BytePairTokenizer (GPT-2 style)."""

    proto = None  # marks this as NOT SentencePiece

    def __init__(self):
        # A tiny but valid byte-level vocab + merges.
        self.vocabulary = {
            "<|endoftext|>": 0,
            "Ġ": 1,
            "t": 2,
            "h": 3,
            "e": 4,
            "Ġt": 5,
            "Ġth": 6,
            "Ġthe": 7,
        }
        self.merges = ["Ġ t", "Ġt h", "Ġth e"]

    def save_assets(self, dir_path):
        with open(os.path.join(dir_path, "vocabulary.json"), "w") as f:
            json.dump(self.vocabulary, f)
        with open(os.path.join(dir_path, "merges.txt"), "w") as f:
            for merge in self.merges:
                f.write(merge + "\n")


def test_bpe_export_writes_hf_files():
    tmp = tempfile.mkdtemp()
    ok = _export_hf_tokenizer(_FakeBPETokenizer(), tmp)

    assert ok is True
    # HF byte-level BPE needs vocab.json + merges.txt.
    assert os.path.exists(os.path.join(tmp, "vocab.json"))
    assert os.path.exists(os.path.join(tmp, "merges.txt"))
    # A usable tokenizer config is present (fast tokenizer.json, or the slow
    # GPT2Tokenizer fallback config).
    assert os.path.exists(os.path.join(tmp, "tokenizer.json")) or os.path.exists(
        os.path.join(tmp, "tokenizer_config.json")
    )


def test_merges_txt_has_version_header():
    # HF's BPE parser drops the first merges line; the export must prepend a
    # header so no real merge rule is lost.
    tmp = tempfile.mkdtemp()
    _export_hf_tokenizer(_FakeBPETokenizer(), tmp)
    with open(os.path.join(tmp, "merges.txt")) as f:
        first = f.readline()
    assert first.startswith("#version")


def test_non_bpe_non_sp_returns_false():
    # An object that is neither BPE (no merges) nor SentencePiece (no proto).
    class _Other:
        proto = None

    assert _export_hf_tokenizer(_Other(), tempfile.mkdtemp()) is False
