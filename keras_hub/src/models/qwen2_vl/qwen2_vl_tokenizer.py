from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen.qwen_tokenizer import QwenTokenizer
from keras_hub.src.models.qwen2_vl.qwen2_vl_backbone import Qwen2VLBackbone


@keras_hub_export(
    [
        "keras_hub.models.Qwen2VLTokenizer",
        "keras_hub.tokenizers.Qwen2VLTokenizer",
    ]
)
class Qwen2VLTokenizer(QwenTokenizer):
    """Qwen2-VL tokenizer layer.

    This tokenizer layer provides an implementation of a Qwen2-VL tokenizer
    using the BytePair (BPE) method. It includes vocabulary and merges data
    necessary for tokenizing Qwen2-VL model inputs.

    In addition to standard text tokenization, this tokenizer exposes
    vision-related token IDs used by the preprocessor to construct
    multimodal input sequences:

    - ``image_token_id``: resolved from ``<|image_pad|>`` (HF ID 151655).
      One placeholder per merged vision patch is inserted by the
      preprocessor.
    - ``video_token_id``: resolved from ``<|video_pad|>`` (HF ID 151656).
    - ``vision_start_token_id``: resolved from ``<|vision_start|>``
      (HF ID 151652). Marks the start of a vision token block.
    - ``vision_end_token_id``: resolved from ``<|vision_end|>``
      (HF ID 151653). Marks the end of a vision token block.

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_vision_token_ids()

    def set_vocabulary_and_merges(self, vocabulary, merges):
        """Override to re-resolve vision token IDs after vocabulary load.

        Keras preset deserialization is two-phase: ``__init__`` is called
        with ``vocabulary=None``, and then ``load_preset_assets``
        loads the real vocabulary from files and calls this method.
        By hooking here we ensure vision token IDs are set
        correctly after both phases.
        """
        super().set_vocabulary_and_merges(vocabulary, merges)
        self._init_vision_token_ids()

    def _init_vision_token_ids(self):
        """Resolve vision token IDs from the vocabulary."""
        if self.vocabulary is not None:
            self.image_token_id = self.token_to_id("<|image_pad|>")
            self.video_token_id = self.token_to_id("<|video_pad|>")
            self.vision_start_token_id = self.token_to_id("<|vision_start|>")
            self.vision_end_token_id = self.token_to_id("<|vision_end|>")
        else:
            self.image_token_id = None
            self.video_token_id = None
            self.vision_start_token_id = None
            self.vision_end_token_id = None
