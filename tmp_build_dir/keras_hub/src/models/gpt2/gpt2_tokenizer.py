from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.gpt2.gpt2_backbone import GPT2Backbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.GPT2Tokenizer",
        "keras_hub.models.GPT2Tokenizer",
    ]
)
class GPT2Tokenizer(BytePairTokenizer):
    """A GPT-2 tokenizer using Byte-Pair Encoding subword segmentation.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_hub.tokenizers.BytePairTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by GPT-2
    models and provides a `from_preset()` method to automatically download
    a matching vocabulary for a GPT-2 preset.

    If input is a batch of strings (rank > 0), the layer will output a
    `tf.RaggedTensor` where the last dimension of the output is ragged.

    If input is a scalar string (rank == 0), the layer will output a dense
    `tf.Tensor` with static shape `[None]`.

    Args:
        vocabulary: string or dict, maps token to integer ids. If it is a
            string, it should be the file path to a json file.
        merges: string or list, contains the merge rule. If it is a string,
            it should be the file path to merge rules. The merge rule file
            should have one merge rule per line. Every merge rule contains
            merge entities separated by a space.

    Examples:

    ```python
    # Unbatched input.
    tokenizer = keras_hub.models.GPT2Tokenizer.from_preset("gpt2_base_en")
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))

    # Custom vocabulary.
    vocab = {"<|endoftext|>": 0, "a": 4, "Ġquick": 5, "Ġfox": 6}
    merges = ["Ġ q", "u i", "c k", "ui ck", "Ġq uick"]
    merges += ["Ġ f", "o x", "Ġf ox"]
    tokenizer = keras_hub.models.GPT2Tokenizer(vocabulary=vocab, merges=merges)
    tokenizer("a quick fox.")
    ```
    """

    backbone_cls = GPT2Backbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        **kwargs,
    ):
        # GPT2 uses the same start as end token, i.e., "<|endoftext|>".
        self._add_special_token("<|endoftext|>", "end_token")
        self._add_special_token("<|endoftext|>", "start_token")
        self.pad_token_id = 0
        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )
