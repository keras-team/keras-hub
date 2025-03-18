from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.stablelm.stablelm_backbone import StableLMBackbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.StableLMTokenizer",
        "keras_hub.models.StableLMTokenizer",
    ]
)
class StableLMTokenizer(BytePairTokenizer):
    """A StableLM tokenizer using Byte-Pair Encoding subword segmentation.

    This tokenizer class tokenizes raw strings into integer sequences and is
    based on `keras_hub.tokenizers.BytePairTokenizer`. It mirrors the GPT-NeoX
    tokenizer, as specified in the StableLM official documentation, and checks
    for all special tokens required by StableLM models. It provides a
    `from_preset()` method to automatically download a matching vocabulary for
    a StableLM preset.

    If input is a batch of strings (rank > 0), the layer outputs a
    `tf.RaggedTensor` where the last dimension is ragged. If input is a scalar
    string (rank == 0), the layer outputs a dense `tf.Tensor` with static
    shape `[None]`.

    Args:
        vocabulary: string or dict, maps tokens to integer IDs. If a string, it
            should be the file path to a JSON file containing the vocabulary.
        merges: string or list, contains the merge rules. If a string, it should
            be the file path to a file with merge rules, where each line contains
            merge entities separated by a space.
    """

    backbone_cls = StableLMBackbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        **kwargs,
    ):
        # StableLM uses the GPT-NeoX tokenizer, which has "<|endoftext|>" as both
        # start and end token.
        self._add_special_token("<|endoftext|>", "end_token")
        self._add_special_token("<|endoftext|>", "start_token")
        self.pad_token_id = 0
        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )