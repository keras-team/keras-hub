from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.llama.llama_tokenizer import LlamaTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.MoonshineTokenizer",
        "keras_hub.models.MoonshineTokenizer",
    ]
)
class MoonshineTokenizer(LlamaTokenizer):
    """
    Moonshine tokenizer layer based on SentencePiece and LlamaTokenizer.
    This tokenizer class tokenizes raw strings into integer sequences by
    extending the LlamaTokenizer class with Moonshine-specific special tokens.
    Source: https://arxiv.org/pdf/2410.15608.pdf (Section 3.1)

    Special tokens added:
        - Start token: "<s>"
        - End token: "</s>"
        - Unknown token: "<unk>"
        - Padding token: "<pad>"
        - Position embedding tokens: "<<ST_0>>" through "<<ST_767>>"
        (768 tokens)
        - Hex tokens: "<0x00>" through "<0xFF>" (256 tokens)
        - Empty token: "<>"

    Args:
        proto: Either a `string` path to a SentencePiece proto file, or a
        `bytes` object with a serialized SentencePiece proto. See the
        [SentencePiece repository](https://github.com/google/sentencepiece)
        for more details on the format.
        **kwargs: Additional keyword arguments passed to the parent
        LlamaTokenizer.

    Examples:
    ```python
    from keras_hub.src.models.moonshine.moonshine_tokenizer import (
        MoonshineTokenizer,
    )

    # Initialize tokenizer
    tokenizer = MoonshineTokenizer(
        "keras_hub/src/tests/test_data/moonshine_test_vocab.spm"
    )

    # Single input example
    single_input = "The quick brown fox jumped."
    single_tokens = tokenizer(single_input)
    print("\nSingle input tokenization:")
    print(f"Input text: {single_input}")
    print(f"Tokenized: {single_tokens}")

    # Batched input example
    batch_input = ["The quick brown fox jumped.", "The fox slept."]
    batch_tokens = tokenizer(batch_input)
    print("\nBatch input tokenization:")
    print(f"Input texts: {batch_input}")
    print(f"Tokenized: {batch_tokens}")

    # Detokenization example
    encoded = tokenizer(single_input)
    decoded = tokenizer.detokenize(encoded)
    print("\nDetokenization:")
    print(f"Original text: {single_input}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    ```
    """

    def __init__(self, proto, **kwargs):
        super().__init__(proto=proto, **kwargs)

        self._add_special_token("<s>", "start_token")
        self._add_special_token("</s>", "end_token")
        self._add_special_token("<unk>", "unk_token")
        self._add_special_token("<pad>", "pad_token")

        for i in range(768):
            self._add_special_token(f"<<ST_{i}>>", f"st_token_{i}")

        for i in range(256):
            self._add_special_token(f"<0x{i:02X}>", f"hex_token_{i}")

        self._add_special_token("<>", "empty_token")
