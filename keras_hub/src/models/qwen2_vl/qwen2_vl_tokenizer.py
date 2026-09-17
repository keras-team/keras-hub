from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen2_vl.qwen2_vl_backbone import Qwen2VLBackbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export(
    [
        "keras_hub.models.Qwen2VLTokenizer",
        "keras_hub.tokenizers.Qwen2VLTokenizer",
    ]
)
class Qwen2VLTokenizer(BytePairTokenizer):
    """Qwen2-VL tokenizer layer.

    This tokenizer layer provides an implementation of a Qwen2-VL tokenizer
    using the BytePair (BPE) method. It includes vocabulary and merges data
    necessary for tokenizing Qwen2-VL model inputs.

    In addition to standard text tokenization, this tokenizer registers the
    vision-related special tokens used to construct multimodal input
    sequences:

    - ``image_token``: ``<|image_pad|>`` (HF ID 151655). One placeholder per
      merged vision patch is inserted by the preprocessor.
    - ``video_token``: ``<|video_pad|>`` (HF ID 151656).
    - ``vision_start_token``: ``<|vision_start|>`` (HF ID 151652). Marks the
      start of a vision token block.
    - ``vision_end_token``: ``<|vision_end|>`` (HF ID 151653). Marks the end
      of a vision token block.

    Note: ``<|image_pad|>`` and ``<|video_pad|>`` are defined in
    HuggingFace's ``tokenizer_config.json`` (``added_tokens_decoder``)
    but are absent from ``tokenizer.json``'s ``added_tokens`` list.
    The converter loads them from ``tokenizer_config.json`` so they
    are present in the vocabulary passed to this class.

    Args:
        vocabulary: string or dict, maps token to integer ids. If it is a
            string, it should be the file path to a json file.
        merges: string or list, contains the merge rule. If it is a string,
            it should be the file path to merge rules. The merge rule file
            should have one merge rule per line.

    Examples:

    ```python
    # Unbatched input.
    tokenizer = keras_hub.models.Qwen2VLTokenizer.from_preset(
        "qwen2_vl_2b_instruct",
    )
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize([[151643, 791, 4320, 14198]])
    ```
    """

    backbone_cls = Qwen2VLBackbone

    def __init__(self, vocabulary=None, merges=None, **kwargs):
        # `<|im_end|>` is the EOS for instruct checkpoints; `<|endoftext|>`
        # doubles as the pad token, matching the HF tokenizer config.
        self._add_special_token("<|im_end|>", "end_token")
        self._add_special_token("<|endoftext|>", "pad_token")
        self._add_special_token("<|im_start|>", "im_start_token")
        self._add_special_token("<|vision_start|>", "vision_start_token")
        self._add_special_token("<|vision_end|>", "vision_end_token")
        self._add_special_token("<|image_pad|>", "image_token")
        self._add_special_token("<|video_pad|>", "video_token")

        self.start_token_id = None
        self.start_token = None

        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )
