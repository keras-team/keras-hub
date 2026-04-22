"""BLIP-2 tokenizer."""

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.blip2.blip2_backbone import Blip2Backbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.Blip2Tokenizer",
        "keras_hub.models.Blip2Tokenizer",
    ]
)
class Blip2Tokenizer(BytePairTokenizer):
    """BLIP-2 tokenizer using Byte-Pair Encoding.

    This tokenizer is based on `keras_hub.tokenizers.BytePairTokenizer` and
    is configured for the BLIP-2 OPT language decoder. It handles the
    special tokens required by the BLIP-2 pipeline and provides a
    `from_preset()` method for automatic vocabulary download.

    If input is a batch of strings (rank > 0), the layer will output a
    `tf.RaggedTensor` where the last dimension of the output is ragged.

    If input is a scalar string (rank == 0), the layer will output a dense
    `tf.Tensor` with static shape `[None]`.

    Args:
        vocabulary: string or dict. Path to a vocabulary file or a dict
            mapping token strings to integer IDs.
        merges: string or list. BPE merge rules, either as a file path or a
            list of merge strings.
        unsplittable_tokens: list of strings. Tokens that must never be split
            during BPE tokenisation. Defaults to
            ``["<pad>", "</s>", "<image>"]``.
        add_prefix_space: bool. Whether to add a leading space before
            tokenising (standard practice for OPT / GPT-style models).
            Defaults to `False`.

    Example:
    ```python
    # Unbatched input.
    tokenizer = keras_hub.models.Blip2Tokenizer.from_preset("blip2_opt_2_7b")
    tokenizer("Question: What is this? Answer:")

    # Batched input.
    tokenizer(["Question: What is in the image?", "Describe the scene."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("Question: What is this? Answer:"))
    ```

    References:
        - [Li et al., 2023](https://arxiv.org/abs/2301.12597)
    """

    backbone_cls = Blip2Backbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        unsplittable_tokens=None,
        add_prefix_space=False,
        **kwargs,
    ):
        # Register special tokens before super().__init__() so they are
        # available as soon as the tokenizer is built — same ordering as
        # Gemma3Tokenizer._add_special_token() calls.
        #
        # OPT uses `</s>` as BOS and overrides EOS to `\n` (Ċ / U+010A),
        # matching the official LAVIS BLIP-2 implementation.
        self._add_special_token("</s>", "start_token")
        self._add_special_token("\u010a", "end_token")
        self._add_special_token("<pad>", "pad_token")

        if unsplittable_tokens is None:
            unsplittable_tokens = ["<pad>", "</s>", "<image>"]

        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            unsplittable_tokens=unsplittable_tokens,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )
