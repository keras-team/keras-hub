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
    Moonshine tokenizer layer based on `keras_hub.models.LlamaTokenizer`.

    This tokenizer class is an alias of `LlamaTokenizer` but for the Moonshine
    model. It uses a SentencePiece vocabulary to handle tokenization.

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
        "keras_hub/src/tests/test_data/llama_test_vocab.spm"
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

    # NOTE: The 768 future-use tokens defined in Section 3.1 of the Moonshine
    # paper, "Moonshine: Speech Recognition for Live Transcription and Voice
    # Commands" (https://arxiv.org/pdf/2410.15608.pdf) serve no purpose in the
    # tokenizer at the moment, and are hence not included in the vocabulary.
