"""Serialize a KerasHub BytePairTokenizer to a HuggingFace tokenizer.json.

Every ``BytePairTokenizer`` already wraps a live ``tokenizers.Tokenizer``
object (``self._tokenizer``, built in ``_set_vocabulary_and_merges_tokenizers``
from KerasHub's own vocab/merges and used for every ``encode_batch``/
``decode_batch``). This module calls ``.to_str()`` on that same object -- it
is *not* a format conversion and re-derives no token ids, so token identity
is byte-exact by construction. HF ``tokenizer.json`` is the target because
the LiteRT-LM runtime accepts only two on-device tokenizer formats
(SentencePiece and ``HF_Tokenizer_Zlib``); there is no KerasHub-native
on-device format to emit instead.

This module does not itself import ``tokenizers`` (it only serializes an
object built elsewhere); the library is used directly only in tests, to
validate the output.
"""

import os

from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


def materialize_hf_tokenizer_json(tokenizer, temp_dir):
    """Convert a KerasHub tokenizer and write ``tokenizer.json`` to disk.

    Relies on the private ``BytePairTokenizer._tokenizer`` attribute and its
    ``_maybe_initialized_tokenizers`` method.

    Args:
        tokenizer: A KerasHub ``BytePairTokenizer`` instance.
        temp_dir: str. Directory where ``tokenizer.json`` will be written.

    Returns:
        str: Path to the written ``tokenizer.json`` file.
    """
    if not isinstance(tokenizer, BytePairTokenizer):
        raise TypeError(
            "`materialize_hf_tokenizer_json` expects a BytePairTokenizer "
            f"instance. Received: {type(tokenizer).__name__}."
        )
    tokenizer._maybe_initialized_tokenizers()
    tokenizer_path = os.path.join(temp_dir, "tokenizer.json")
    with open(tokenizer_path, "w", encoding="utf-8") as f:
        f.write(tokenizer._tokenizer.to_str())
    return tokenizer_path
