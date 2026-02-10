import base64
import json
import os
from pathlib import Path

import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.tokenizers import tokenizer
from keras_hub.src.utils.tensor_utils import convert_to_ragged_batch
from keras_hub.src.utils.tensor_utils import is_int_dtype
from keras_hub.src.utils.tensor_utils import is_string_dtype
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import tensor_to_list

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import tiktoken
except ImportError:
    tiktoken = None


TIKTOKEN_CONFIG_FILENAME = "tiktoken_config.json"


def _load_json_like(proto):
    """Resolve proto into a Python object (path/bytes/dict)."""
    if isinstance(proto, (str, Path)):
        with open(proto, "r", encoding="utf-8") as f:
            return json.load(f)
    if isinstance(proto, (bytes, bytearray)):
        return json.loads(proto.decode("utf-8"))
    if isinstance(proto, dict):
        return proto
    raise ValueError(f"Unsupported proto type: {type(proto)}")


def _is_normalized_proto(model_data) -> bool:
    """Check whether the proto already has the normalized shape."""
    return "pattern" in model_data and "mergeable_ranks" in model_data


def _normalize_tekken_proto(model_data) -> dict:
    """Convert tekken.json structure into normalized proto for tiktoken."""
    if "config" not in model_data or "vocab" not in model_data:
        raise ValueError(
            "Tekken JSON missing required 'config' or 'vocab' fields."
        )

    cfg = model_data["config"]
    pattern = cfg["pattern"]
    vocab_size = cfg["default_vocab_size"]
    num_special_tokens = cfg["default_num_special_tokens"]

    special_tokens = model_data.get("special_tokens") or []
    special_lookup = {t["token_str"]: t["rank"] for t in special_tokens}

    inner_vocab_size = vocab_size - num_special_tokens
    vocab_slice = model_data["vocab"][:inner_vocab_size]
    mergeable_ranks = {
        entry["token_bytes"]: entry["rank"] for entry in vocab_slice
    }

    normalized = {
        "pattern": pattern,
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": special_tokens,
        "special_lookup": special_lookup,
        "num_special_tokens": num_special_tokens,
        "vocab_size": vocab_size,
        "raw_json": model_data,
    }

    return normalized


def _encode_strings(strings, model):
    """Encode a single string or list of strings using tiktoken."""
    if isinstance(strings, (list, tuple)):
        return model.encode_batch(strings)
    return model.encode(strings)


def _decode_token_ids(token_ids_list, model):
    """Decode token IDs → string(s). Supports nested lists."""
    if not isinstance(token_ids_list, list):
        return model.decode([token_ids_list])

    if len(token_ids_list) == 0:
        return ""

    if isinstance(token_ids_list[0], list):
        return model.decode_batch(token_ids_list)

    return model.decode(token_ids_list)


def _normalize_mergeable_ranks(mr) -> dict:
    """Ensure mergeable_ranks has bytes keys and integer ranks.

    JSON cannot store byte keys, so Tekken-format or similar loaders
    store base64 strings. This function converts those keys → bytes.
    """
    normalized = {}

    for k, v in mr.items():
        rank = int(v)

        # Convert key to bytes
        if isinstance(k, str):
            # Likely base64 encoded bytes
            try:
                key_bytes = base64.b64decode(k)
            except Exception:
                # fallback to utf-8
                key_bytes = k.encode("utf-8", errors="surrogatepass")
        elif isinstance(k, (bytes, bytearray)):
            key_bytes = bytes(k)
        else:
            key_bytes = str(k).encode("utf-8", errors="surrogatepass")

        normalized[key_bytes] = rank

    return normalized


@keras_hub_export("keras_hub.tokenizers.TiktokenTokenizer")
class TiktokenTokenizer(tokenizer.Tokenizer):
    """
    Format-agnostic tiktoken tokenizer with Tekken support.

    The tokenizer can consume:
      • A normalized proto dict with keys:
        {
            "pattern": str,
            "mergeable_ranks": Dict[base64|bytes → rank],
            "special_tokens": list[{token_str, rank}] (optional),
            "special_lookup": Dict[token_str → rank] (optional),
            "num_special_tokens": int,
            "vocab_size": int,
            "raw_json": original JSON (optional)
        }
      • A Tekken JSON (path/bytes/dict). It will be normalized internally.
    """

    def __init__(
        self,
        proto,
        sequence_length=None,
        name="tiktoken",
        dtype="int32",
        add_bos=False,
        add_eos=False,
        **kwargs,
    ):
        if tf is None:
            raise ImportError("TensorFlow is required for TiktokenTokenizer.")
        if tiktoken is None:
            raise ImportError("tiktoken must be installed.")

        if not (is_int_dtype(dtype) or is_string_dtype(dtype)):
            raise ValueError(
                f"Output dtype must be int or string. Received {dtype}."
            )

        super().__init__(dtype=dtype, **kwargs)

        self.name = name
        self.sequence_length = sequence_length
        self.add_bos = add_bos
        self.add_eos = add_eos

        self.proto = None
        self._model = None
        self._num_special_tokens = 0
        self._vocab_size = None
        self._all_special_tokens = []
        self._special_tokens_reverse_vocab = {}

        self.set_proto(proto)

    def _refresh_special_token_ids(self):
        """Update cached special token ids from the current reverse vocab."""
        if not hasattr(self, "_special_token_attrs"):
            return
        for attr in self._special_token_attrs:
            token = getattr(self, attr, None)
            if token is None:
                setattr(self, f"{attr}_id", None)
                continue
            if token in self._special_tokens_reverse_vocab:
                setattr(
                    self,
                    f"{attr}_id",
                    int(self._special_tokens_reverse_vocab[token]),
                )
            else:
                setattr(self, f"{attr}_id", None)

    def _get_special_id(self, attr_name, default_token=None):
        """Return id for a special token attribute if known."""
        token = getattr(self, attr_name, None)
        if token is None:
            token = default_token
        if token and token in self._special_tokens_reverse_vocab:
            return int(self._special_tokens_reverse_vocab[token])
        return getattr(self, f"{attr_name}_id", None)

    def save_assets(self, dir_path):
        """Save the raw normalized proto JSON."""
        if self.proto is None:
            return

        raw_json = self.proto.get("raw_json", self.proto)

        path = os.path.join(dir_path, TIKTOKEN_CONFIG_FILENAME)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(raw_json, f, ensure_ascii=False)

    def load_assets(self, dir_path):
        """Load saved proto JSON and reinitialize tokenizer."""
        path = os.path.join(dir_path, TIKTOKEN_CONFIG_FILENAME)
        self.set_proto(path)

    def set_proto(self, proto):
        """Load normalized proto (dict OR tekken JSON path/bytes/dict)."""
        if proto is None:
            self.proto = None
            self._model = None
            return

        model_data = _load_json_like(proto)
        if _is_normalized_proto(model_data):
            normalized_proto = model_data
        else:
            normalized_proto = _normalize_tekken_proto(model_data)

        # Validate minimum requirements
        if (
            "pattern" not in normalized_proto
            or "mergeable_ranks" not in normalized_proto
        ):
            raise ValueError(
                "Normalized proto missing required fields "
                "('pattern', 'mergeable_ranks')."
            )

        pattern = normalized_proto["pattern"]
        mergeable_ranks = _normalize_mergeable_ranks(
            normalized_proto["mergeable_ranks"]
        )

        self._num_special_tokens = int(
            normalized_proto.get("num_special_tokens", 0)
        )
        self._vocab_size = normalized_proto.get("vocab_size")
        self._all_special_tokens = normalized_proto.get("special_tokens", [])
        self._special_tokens_reverse_vocab = normalized_proto.get(
            "special_lookup", {}
        )

        self.proto = normalized_proto

        self._model = tiktoken.Encoding(
            name=self.name,
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens={},
        )

        self._refresh_special_token_ids()

    def vocabulary_size(self):
        self._check_vocabulary()
        return int(self._model.n_vocab)

    def get_vocabulary(self):
        self._check_vocabulary()
        vocab = []
        for i in range(self._model.n_vocab):
            token_bytes = self._model.decode_single_token_bytes(i)
            text = token_bytes.decode("utf-8", errors="backslashreplace")
            vocab.append(text.replace("Ġ", "▁"))
        return vocab

    def id_to_token(self, tid):
        self._check_vocabulary()

        # Special tokens:
        if tid < self._num_special_tokens:
            if self._all_special_tokens:
                return self._all_special_tokens[tid]["token_str"]

        # Regular token (offset)
        tid = tid - self._num_special_tokens
        token_bytes = self._model.decode_single_token_bytes(tid)
        return token_bytes.decode("utf-8", errors="replace")

    def token_to_id(self, token):
        self._check_vocabulary()

        if token in self._special_tokens_reverse_vocab:
            return int(self._special_tokens_reverse_vocab[token])

        encoded = self._model.encode(token)
        if len(encoded) != 1:
            raise ValueError(
                f"Token '{token}' splits into multiple tokens: {encoded}. "
                "Use tokenize() for multi-token sequences."
            )

        return int(encoded[0]) + self._num_special_tokens

    @preprocessing_function
    def tokenize(self, inputs):
        self._check_vocabulary()

        inputs = tf.convert_to_tensor(inputs)
        unbatched = inputs.shape.rank == 0
        if unbatched:
            inputs = tf.expand_dims(inputs, 0)

        strings = tensor_to_list(inputs)
        encoded = _encode_strings(strings, self._model)

        if self._num_special_tokens > 0:
            encoded = [
                [int(t) + self._num_special_tokens for t in seq]
                for seq in encoded
            ]

        if self.add_bos:
            bos_id = self._get_special_id("start_token", "<s>")
            encoded = [[bos_id] + seq for seq in encoded]

        if self.add_eos:
            eos_id = self._get_special_id("end_token", "</s>")
            encoded = [seq + [eos_id] for seq in encoded]

        if self.compute_dtype == tf.string:
            encoded = [[str(t) for t in seq] for seq in encoded]
            rt = tf.ragged.constant(encoded, dtype=tf.string)
        else:
            rt = tf.ragged.constant(encoded, dtype=self.compute_dtype)

        if self.sequence_length:
            out_shape = rt.shape.as_list()
            out_shape[-1] = self.sequence_length
            rt = rt.to_tensor(shape=out_shape, default_value=0)

        if unbatched:
            rt = tf.squeeze(rt, 0)
            if self.sequence_length:
                tf.ensure_shape(rt, [self.sequence_length])

        return rt

    @preprocessing_function
    def detokenize(self, inputs):
        self._check_vocabulary()

        inputs, unbatched, _ = convert_to_ragged_batch(inputs)
        inputs = tf.cast(inputs, "int32")

        input_list = tensor_to_list(inputs)

        def strip_special(seq):
            return [
                int(t) - self._num_special_tokens
                for t in seq
                if int(t) >= self._num_special_tokens
            ]

        if isinstance(input_list, list):
            regular_lists = []
            for seq in input_list:
                if isinstance(seq, list):
                    regular_lists.append(strip_special(seq))
                else:
                    regular_lists.append(strip_special([seq]))
            decoded = _decode_token_ids(regular_lists, self._model)
        else:
            decoded = [
                _decode_token_ids(strip_special(input_list), self._model)
            ]

        out = tf.constant(decoded)
        if unbatched:
            out = tf.squeeze(out, 0)

        return out

    def _check_vocabulary(self):
        if self.proto is None:
            raise ValueError("Tokenizer proto not set.")
        if self._model is None:
            raise ValueError("Tokenizer model not initialized.")

    def compute_output_spec(self, input_spec):
        if self.sequence_length:
            return keras.KerasTensor(
                input_spec.shape + (self.sequence_length,),
                dtype=self.compute_dtype,
            )
        return keras.KerasTensor(
            input_spec.shape + (None,),
            dtype=self.compute_dtype,
        )
