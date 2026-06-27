"""Convert KerasHub BytePair tokenizers to HuggingFace tokenizer.json format.

The converter is intentionally dependency-free at runtime: it builds the JSON
structure that ``tokenizers.Tokenizer.from_file()`` expects without importing
the ``tokenizers`` library.  The ``tokenizers`` library is only used in tests
to validate the output.
"""

import json
import os

from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


def _validate_and_ensure_initialized(tokenizer, caller_name):
    """Validate ``tokenizer`` is a ``BytePairTokenizer`` and is initialized.

    This helper centralizes the isinstance guard and the private
    ``_maybe_initialized_tokenizers`` call used by the converter routines.
    It relies on the private ``BytePairTokenizer._tokenizer`` attribute and
    its ``_maybe_initialized_tokenizers`` method; renaming either in
    ``BytePairTokenizer`` will require updating this helper.
    """
    if not isinstance(tokenizer, BytePairTokenizer):
        raise TypeError(
            f"`{caller_name}` expects a BytePairTokenizer instance. "
            f"Received: {type(tokenizer).__name__}."
        )
    tokenizer._maybe_initialized_tokenizers()


def convert_byte_pair_to_hf(tokenizer):
    """Convert any KerasHub BytePairTokenizer subclass to HF tokenizer.json.

    Args:
        tokenizer: A
            ``keras_hub.src.tokenizers.byte_pair_tokenizer.BytePairTokenizer``
            instance.

    Returns:
        dict: A ``tokenizer.json``-compatible dictionary that can be written to
        disk and loaded with ``tokenizers.Tokenizer.from_file(path)``.
    """
    _validate_and_ensure_initialized(tokenizer, "convert_byte_pair_to_hf")
    return json.loads(tokenizer._tokenizer.to_str())


def materialize_hf_tokenizer_json(tokenizer, temp_dir):
    """Convert a KerasHub tokenizer and write ``tokenizer.json`` to disk.

    Args:
        tokenizer: A KerasHub ``BytePairTokenizer`` instance.
        temp_dir: str. Directory where ``tokenizer.json`` will be written.

    Returns:
        str: Path to the written ``tokenizer.json`` file.
    """
    _validate_and_ensure_initialized(tokenizer, "materialize_hf_tokenizer_json")
    tokenizer_path = os.path.join(temp_dir, "tokenizer.json")
    with open(tokenizer_path, "w", encoding="utf-8") as f:
        f.write(tokenizer._tokenizer.to_str())
    return tokenizer_path
