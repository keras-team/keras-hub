"""KerasHub tokenizers served directly through vLLM's tokenizer registry.

vLLM supports custom tokenizer implementations via its ``TokenizerRegistry``
(``tokenizer_mode``): `KerasHubTokenizer` implements vLLM's tokenizer protocol
on top of the preset's own ``keras_hub.tokenizers.Tokenizer``, so serving uses
the exact tokenizer the model was trained with — for every preset, in every
format — with no HF conversion or export.

``KerasHubLLM`` registers this class under ``tokenizer_mode="keras_hub"``, and
tpu-inference registers it at plugin load so every vLLM process resolves it.
"""

import json
import logging
import os

import numpy as np
from keras import ops
from transformers import BatchEncoding

from keras_hub.src.tokenizers.tokenizer import Tokenizer


def _is_scalar_id(ids):
    """True for a single id: a plain int, numpy integer, or 0-d array."""
    return isinstance(ids, (int, np.integer)) or (
        hasattr(ids, "ndim") and ids.ndim == 0
    )


def _as_id_list(ids):
    """Returns ``ids`` as a list of ids, tolerating a scalar.

    Accepts plain ints, numpy integers, 0-d arrays, sequences, and
    framework tensors (converted through numpy rather than iterated
    element-wise, which is slow on TF/torch tensors).
    """
    # Convert tensors/arrays to numpy first, so a 0-d framework tensor
    # comes out as a plain Python scalar too.
    if hasattr(ids, "shape") or hasattr(ids, "__array__"):
        ids = ops.convert_to_numpy(ids)
    if _is_scalar_id(ids):
        return [ids.item() if hasattr(ids, "item") else int(ids)]
    if hasattr(ids, "tolist"):
        return ids.tolist()
    return list(ids)


def _token_to_str(token):
    """Decodes a token to ``str`` when it is raw bytes.

    Byte-level BPE vocabularies expose some tokens as ``bytes``; vLLM expects
    string keys, so decode them (``errors="replace"`` keeps partial multi-byte
    tokens from raising). ``hasattr(..., "decode")`` also catches
    ``np.bytes_``, which is not an instance of ``bytes``.
    """
    if hasattr(token, "decode"):
        return token.decode("utf-8", errors="replace")
    return token


class KerasHubTokenizer:
    """The preset's own KerasHub tokenizer, speaking vLLM's interface.

    Implements vLLM's ``TokenizerLike`` protocol so it can be registered
    under ``tokenizer_mode="keras_hub"``. All tokenization is done by the
    KerasHub tokenizer itself; this class only translates the calling
    conventions, and reads special tokens from the preset's own metadata
    (nothing is hardcoded).
    """

    def __init__(self, preset_or_tokenizer):
        """Initializes the tokenizer from a preset name or a loaded tokenizer.

        Args:
            preset_or_tokenizer: The name of the KerasHub preset to load, or
                an already-loaded KerasHub `Tokenizer`.
        """
        if isinstance(preset_or_tokenizer, str):
            self.preset_name = preset_or_tokenizer
            self.tokenizer = Tokenizer.from_preset(preset_or_tokenizer)
        else:
            self.preset_name = getattr(preset_or_tokenizer, "name", "unknown")
            self.tokenizer = preset_or_tokenizer
        # Vocabulary caches, built once on first use: id -> token string
        # (list) and token string -> id (dict). Vocabularies run to 256k+
        # entries, so rebuilding them per call is not an option.
        self._vocab_list = None
        self._vocab = None
        self._max_chars_per_token = None
        # Plain attributes (not properties): vLLM's renderer may assign to
        # them, mirroring HF tokenizer semantics.
        self.truncation_side = "right"
        self.model_max_length = int(1e30)  # HF's "no limit" sentinel.

    @classmethod
    def from_pretrained(cls, path_or_repo_id, *args, **kwargs):
        """Loads the tokenizer vLLM asked for.

        vLLM passes the model/tokenizer path from its config. When that path
        is a directory written by ``setup_vllm_model``, the preset name is
        read from its ``config.json`` (the ``keras_hub_preset`` field);
        otherwise the string is treated as a preset name directly.
        """
        preset = str(path_or_repo_id)
        config_path = os.path.join(preset, "config.json")
        if os.path.isfile(config_path):
            with open(config_path, encoding="utf-8") as f:
                preset = json.load(f).get("keras_hub_preset", preset)
        return cls(preset)

    # ------------------------------------------------------------------ #
    # Vocabulary
    # ------------------------------------------------------------------ #
    @property
    def is_fast(self):
        """KerasHub tokenizers are not HF "fast" (Rust) tokenizers."""
        return False

    @property
    def name_or_path(self):
        """The preset name (vLLM uses this for caching / processor lookup)."""
        return self.preset_name

    @property
    def vocab_size(self):
        """Returns the size of the vocabulary."""
        return int(self.tokenizer.vocabulary_size())

    @property
    def max_token_id(self):
        """The largest valid token id."""
        return self.vocab_size - 1

    def __len__(self):
        """Returns the vocabulary size when length is requested."""
        return self.vocab_size

    def __hash__(self):
        """Stable per-preset hash (vLLM caches tokenizers)."""
        return hash((type(self).__name__, self.preset_name))

    @property
    def max_chars_per_token(self):
        """The longest token string in the vocabulary (streaming buffer)."""
        if self._max_chars_per_token is None:
            longest = max((len(token) for token in self.get_vocab()), default=1)
            self._max_chars_per_token = max(longest, 1)
        return self._max_chars_per_token

    def get_vocab(self):
        """Returns a dictionary mapping token strings to integer ids."""
        if self._vocab is None:
            self._vocab_list = self._build_vocab_list()
            self._vocab = {
                token: idx for idx, token in enumerate(self._vocab_list)
            }
        return self._vocab

    def _build_vocab_list(self):
        """Builds the id -> token-string list, once."""
        if hasattr(self.tokenizer, "get_vocabulary"):
            vocabulary = self.tokenizer.get_vocabulary()
            # A None vocabulary is treated like get_vocabulary not being
            # exposed at all: fall through to the id_to_token fallback.
            if vocabulary is not None:
                return [_token_to_str(token) for token in vocabulary]
        # Fallback if get_vocabulary is not exposed. Probe one id first:
        # when id_to_token is unsupported entirely, build the placeholder
        # vocab directly instead of raising once per id.
        try:
            self.tokenizer.id_to_token(0)
        except Exception:  # noqa: BLE001
            return [str(i) for i in range(self.vocab_size)]
        tokens = []
        for i in range(self.vocab_size):
            try:
                tokens.append(_token_to_str(self.tokenizer.id_to_token(i)))
            except Exception:  # noqa: BLE001
                tokens.append(str(i))
        return tokens

    def get_added_vocab(self):
        """KerasHub presets have no separately-added vocabulary."""
        return {}

    # ------------------------------------------------------------------ #
    # Special tokens
    # ------------------------------------------------------------------ #
    @property
    def bos_token_id(self):
        """The beginning-of-sequence token id, or None if the preset has none.

        Read from the KerasHub tokenizer; no hardcoded fallback, since a wrong
        guess corrupts prompts for models whose ids differ.
        """
        return getattr(self.tokenizer, "start_token_id", None)

    @property
    def eos_token_id(self):
        """The end-of-sequence token id, or None if the preset has none."""
        return getattr(self.tokenizer, "end_token_id", None)

    @property
    def pad_token_id(self):
        """The padding token id, or None if the preset has none."""
        return getattr(self.tokenizer, "pad_token_id", None)

    @property
    def all_special_ids(self):
        """The special-token ids the preset actually defines.

        Only ids the tokenizer exposes are included (a missing bos/eos/pad is
        dropped rather than defaulted), so this list drives what
        `all_special_tokens` and `skip_special_tokens` decoding act on.
        """
        special_ids = [self.bos_token_id, self.eos_token_id, self.pad_token_id]
        return [token_id for token_id in special_ids if token_id is not None]

    @property
    def all_special_tokens(self):
        """The special-token strings, aligned one-to-one with
        `all_special_ids`.

        Derived from the tokenizer's own vocabulary rather than hardcoded, so
        the strings match the preset (e.g. `<|endoftext|>`, `</s>`) and never
        desync in length from `all_special_ids` when a preset omits a token.
        """
        return self.convert_ids_to_tokens(self.all_special_ids)

    def num_special_tokens_to_add(self):
        """How many special tokens `encode` adds (the start token, if any)."""
        return 1 if self.bos_token_id is not None else 0

    # ------------------------------------------------------------------ #
    # Encode / decode
    # ------------------------------------------------------------------ #
    def encode(
        self, text, truncation=None, max_length=None, add_special_tokens=True
    ):
        """Converts text to token ids with the preset's own tokenizer.

        When ``add_special_tokens`` is set and the preset defines a start
        token, it is prepended — mirroring KerasHub generation
        (``add_start_token=True``), so served prompts tokenize exactly as
        `model.generate()` would see them.
        """
        if not isinstance(text, str):
            raise TypeError(
                "KerasHubTokenizer.encode expects a single string, got "
                f"{type(text).__name__}. Batch encoding is not supported."
            )
        token_ids = ops.convert_to_numpy(self.tokenizer(text)).tolist()
        bos = self.bos_token_id
        if add_special_tokens and bos is not None:
            if not token_ids or token_ids[0] != bos:
                token_ids = [int(bos)] + token_ids
        if truncation and max_length is not None:
            # Honor `truncation_side`: vLLM's renderer assigns it to keep
            # the end of over-long prompts (left truncation).
            if self.truncation_side == "left":
                token_ids = token_ids[-max_length:]
            else:
                token_ids = token_ids[:max_length]
        return token_ids

    def __call__(
        self,
        text,
        text_pair=None,
        add_special_tokens=True,
        truncation=False,
        max_length=None,
    ):
        """HF-style call returning a ``BatchEncoding`` with ``input_ids``."""
        if text_pair is not None:
            raise ValueError(
                "KerasHubTokenizer does not support text pairs; CausalLM "
                "serving tokenizes single prompts."
            )
        input_ids = self.encode(
            text,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        return BatchEncoding(
            {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
        )

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        """Converts token ids back to text with the preset's own tokenizer.

        Args:
            token_ids: A list of token ids (or a single id).
            skip_special_tokens: When True, drop the preset's special-token
                ids (bos/eos/pad) before detokenizing, so vLLM's final output
                does not leak `</s>`, `<|endoftext|>`, etc.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            The decoded string.
        """
        # Plain Python ints upfront: ids arrive as ints, numpy scalars,
        # or arrays, and downstream (set filtering, detokenize) is simplest
        # against one type.
        token_ids = [int(token_id) for token_id in _as_id_list(token_ids)]
        if skip_special_tokens:
            special_ids = set(self.all_special_ids)
            token_ids = [
                token_id
                for token_id in token_ids
                if token_id not in special_ids
            ]
        # Empty input (or everything filtered as special) decodes to "".
        if not token_ids:
            return ""
        text = self.tokenizer.detokenize(token_ids)
        np_text = ops.convert_to_numpy(text)
        if hasattr(np_text, "item"):
            np_text = np_text.item()
        if isinstance(np_text, bytes):
            # errors="replace": streaming can split a multi-byte UTF-8
            # character across chunks, which would otherwise raise.
            return np_text.decode("utf-8", errors="replace")
        return str(np_text)

    # ------------------------------------------------------------------ #
    # Token <-> id <-> string (vLLM's incremental detokenizer uses these)
    # ------------------------------------------------------------------ #
    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """Maps token ids to their token strings.

        Mirrors the HF tokenizer API: a scalar id returns a single token
        string (or None if it was filtered as special); a sequence returns
        a list.
        """
        is_scalar = _is_scalar_id(ids)
        ids = [int(i) for i in _as_id_list(ids)]
        if skip_special_tokens:
            special_ids = set(self.all_special_ids)
            ids = [i for i in ids if i not in special_ids]
        self.get_vocab()  # ensures the id -> token cache is built
        tokens = [self._vocab_list[i] for i in ids]
        if is_scalar:
            return tokens[0] if tokens else None
        return tokens

    def convert_tokens_to_ids(self, tokens):
        """Maps token strings to their ids.

        Unknown tokens raise `KeyError` rather than mapping to `None`,
        which would silently corrupt downstream id math.
        """
        vocab = self.get_vocab()
        # Normalize through _token_to_str so byte tokens (byte-level BPE
        # vocabularies) look up the same way the vocab dict was built.
        if isinstance(tokens, (str, bytes)):
            return vocab[_token_to_str(tokens)]
        return [vocab[_token_to_str(t)] for t in tokens]

    def convert_tokens_to_string(self, tokens):
        """Joins token strings into text via the tokenizer's detokenizer.

        Ids come from `convert_tokens_to_ids`, so an unknown token raises
        `KeyError` here instead of sending `None` ids into detokenization.
        """
        return self.decode(self.convert_tokens_to_ids(tokens))

    def apply_chat_template(self, messages, tools=None, **kwargs):
        """Chat templates are not part of KerasHub presets."""
        raise ValueError(
            "KerasHub presets do not carry a chat template; pass plain text "
            "prompts (or pre-formatted conversations) to generate()."
        )

    def get_chat_template(self, chat_template=None, tools=None):
        """Returns the caller-provided template; presets carry none."""
        return chat_template

    def add_special_tokens(self, special_tokens):
        """KerasHub vocabularies are fixed; nothing can be added.

        vLLM only calls this for its prompt-embeds placeholder token, an
        opt-in feature this integration does not use, so the request is
        ignored (returning 0 tokens added, matching HF semantics).
        """
        logging.warning(
            "KerasHub tokenizer vocabulary is fixed; ignoring request to "
            "add special tokens %s.",
            special_tokens,
        )
        return 0
