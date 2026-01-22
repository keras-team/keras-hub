from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.clip.clip_tokenizer import CLIPTokenizer
from keras_hub.src.models.sam3.sam3_pc_backbone import (
    SAM3PromptableConceptBackbone,
)


@keras_hub_export(
    [
        "keras_hub.tokenizers.SAM3Tokenizer",
        "keras_hub.models.SAM3Tokenizer",
    ]
)
class SAM3Tokenizer(CLIPTokenizer):
    """A SAM3 tokenizer using Byte-Pair Encoding subword segmentation.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_hub.tokenizers.BytePairTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by SAM3
    models and provides a `from_preset()` method to automatically download
    a matching vocabulary for a SAM3 preset.

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
    tokenizer = keras_hub.models.SAM3Tokenizer.from_preset("sam3_pcs")
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))
    ```
    """

    backbone_cls = SAM3PromptableConceptBackbone

    def __init__(self, vocabulary=None, merges=None, **kwargs):
        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            pad_with_end_token=True,
            **kwargs,
        )

    def get_config(self):
        config = super().get_config()
        del config["pad_with_end_token"]  # Always True for SAM3Tokenizer.
        return config
