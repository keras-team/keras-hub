from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.opt.opt_backbone import OPTBackbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.OPTTokenizer",
        "keras_hub.models.OPTTokenizer",
    ]
)
class OPTTokenizer(BytePairTokenizer):
    """An OPT tokenizer using Byte-Pair Encoding subword segmentation.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_hub.tokenizers.BytePairTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by OPT
    models and provides a `from_preset()` method to automatically download
    a matching vocabulary for a OPT preset.

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
    tokenizer = keras_hub.models.OPTTokenizer.from_preset(
        "opt_125m_en",
    )
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))

    # Custom vocabulary.
    vocab = {"<pad>": 1, "</s>": 2, "Ġquick": 4, "Ġfox": 5}
    merges = ["Ġ q", "u i", "c k", "ui ck", "Ġq uick"]
    merges += ["Ġ f", "o x", "Ġf ox"]
    tokenizer = keras_hub.models.OPTTokenizer(vocabulary=vocab, merges=merges)
    tokenizer("The quick brown fox jumped.")
    ```
    """

    backbone_cls = OPTBackbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        **kwargs,
    ):
        self._add_special_token("</s>", "end_token")
        self._add_special_token("</s>", "start_token")
        self._add_special_token("<pad>", "pad_token")
        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )
