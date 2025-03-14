import base64

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

    This tokenizer class extends the `LlamaTokenizer` to tokenize raw strings
    to integer sequences while incorporating Moonshine-specific special tokens.

    **Special tokens added:**
    - **Start token:** "<s>"
    - **End token:** "</s>"
    - **Unknown token:** "<unk>"
    - **Padding token:** "<pad>"
    - **Position embedding tokens:** "<<ST_0>>" through "<<ST_767>>"
    - **Hex tokens:** "<0x00>" through "<0xFF>"
    - **Empty token:** "<>"

    Args:
        proto: `str` or `bytes`. Either a string path to a SentencePiece proto
            file or a bytes object containing a serialized SentencePiece proto.
            See the [SentencePiece repository](https://github.com/google/sentencepiece)
            for details on the format.
        **kwargs: Additional keyword arguments passed to the parent
            `LlamaTokenizer`.

    Examples:
    ```python
    from keras_hub.tokenizers import MoonshineTokenizer

    # Initialize tokenizer.
    tokenizer = MoonshineTokenizer(
        "keras_hub/src/tests/test_data/moonshine_test_vocab.spm"
    )

    # Single input example.
    single_input = "the quick brown fox"
    single_tokens = tokenizer(single_input)
    print("Single input tokenization:")
    print(f"Input text: {single_input}")
    print(f"Tokenized: {single_tokens}")

    # Batched input example.
    batch_input = ["the quick brown fox", "the earth is round"]
    batch_tokens = tokenizer(batch_input)
    print("Batch input tokenization:")
    print(f"Input texts: {batch_input}")
    print(f"Tokenized: {batch_tokens}")

    # Detokenization example.
    encoded = tokenizer(single_input)
    decoded = tokenizer.detokenize(encoded)
    print("Detokenization:")
    print(f"Original text: {single_input}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    ```
    """

    # References:
    # Defined in Section 3.1 of the Moonshine paper, "Moonshine: Speech
    # Recognition for Live Transcription and Voice Commands" (https://arxiv.org/pdf/2410.15608.pdf)

    def __init__(self, proto, **kwargs):
        super().__init__(proto=proto, **kwargs)
        self._add_special_token("<unk>", "unk_token")
        self._add_special_token("<pad>", "pad_token")

        for i in range(768):
            self._add_special_token(f"<<ST_{i}>>", f"st_token_{i}")

        for i in range(256):
            self._add_special_token(f"<0x{i:02X}>", f"hex_token_{i}")

        self._add_special_token("<>", "empty_token")

        self.bos_token_id = self.token_to_id("<s>")  # Beginning of sentence
        self.eos_token_id = self.token_to_id("</s>")  # End of sentence
        self.pad_token_id = self.token_to_id("<pad>")  # Padding token
        self.unk_token_id = self.token_to_id("<unk>")  # Unknown token

    def get_config(self):
        config = super().get_config()
        if isinstance(self.proto, bytes):
            config["proto"] = base64.b64encode(self.proto).decode("utf-8")
        else:
            config["proto"] = self.proto
        return config

    @classmethod
    def from_config(cls, config):
        if "proto" in config and isinstance(config["proto"], str):
            config["proto"] = base64.b64decode(config["proto"])
        return super().from_config(config)
