from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.phi4.phi4_backbone import Phi4Backbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.Phi4Tokenizer",
        "keras_hub.models.Phi4Tokenizer",
    ]
)
class Phi4Tokenizer(BytePairTokenizer):
    """Phi4 tokenizer using Byte-Pair Encoding subword segmentation.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_hub.tokenizers.BytePairTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by
    Phi4 models and provides a `from_preset()` method to automatically
    download a matching vocabulary for a Phi4 preset.

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
        sequence_length: int. If set, the output will be
            padded or truncated to the `sequence_length`. Defaults to 100,352
            based on the [Phi-4 Technical Report](https://arxiv.org/pdf/2412.08905)

    Examples:
    ```python
    # Unbatched input.
    tokenizer = keras_hub.models.Phi4Tokenizer.from_preset(
        "phi4_mini_4k_instruct_en",
    )
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))
    ```

    # References

    - [Phi-4 tokenizer config](https://huggingface.co/microsoft/phi-4/raw/main/tokenizer.json)
    """

    backbone_cls = Phi4Backbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        sequence_length=100_352,
        **kwargs,
    ):
        self._add_special_token("<s>", "start_token")
        self._add_special_token("</s>", "end_token")
        self._add_special_token("<pad>", "pad_token")

        # FIM = Fill-in-the-middle, which uses special tokens to identify
        # the prefix/middle/suffix part of the input/output for coding tasks.
        self._add_special_token("<fim_prefix>", "fim_prefix")
        self._add_special_token("<fim_middle>", "fim_middle")
        self._add_special_token("<fim_suffix>", "fix_suffix")

        self._add_special_token("<im_start>", "input_message_start")
        self._add_special_token("<im_sep>", "input_message_separator")
        self._add_special_token("<im_end>", "input_message_end")

        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            sequence_length=sequence_length,
            **kwargs,
        )
