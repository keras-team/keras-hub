"""Tokenizer for HRM-Text."""

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.hrm_text.hrm_text_backbone import HrmTextBackbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.HrmTextTokenizer",
        "keras_hub.models.HrmTextTokenizer",
    ]
)
class HrmTextTokenizer(BytePairTokenizer):
    """Byte-pair tokenizer used by HRM-Text.

    The official checkpoint uses the Qwen2 BPE vocabulary with HRM-Text's
    start, end, and padding special tokens. Instantiate it from a converted
    local preset, or construct it from the official tokenizer vocabulary and
    merge rules.

    Args:
        vocabulary: Mapping from token strings to token ids.
        merges: Ordered BPE merge rules.

    The active HRM-Text inference tokens are ``<|im_start|>``,
    ``<|im_end|>``, ``<|box_end|>``, ``<|endoftext|>``, and the four
    condition tokens ``<|object_ref_start|>``, ``<|object_ref_end|>``,
    ``<|quad_start|>``, and ``<|quad_end|>``. They are registered as special
    tokens so they are encoded atomically. The checkpoint retains additional
    Qwen-style vocabulary tokens, but those are not part of this tokenizer's
    active HRM-Text inference protocol.

    Examples:

    ```python
    tokenizer = keras_hub.models.HrmTextTokenizer.from_preset(
        "/path/to/hrm_text_1b"
    )
    token_ids = tokenizer(["HRM-Text uses two recurrent Transformer stacks."])
    ```
    """

    backbone_cls = HrmTextBackbone

    def __init__(self, vocabulary=None, merges=None, **kwargs):
        self._add_special_token("<|im_start|>", "start_token")
        self._add_special_token("<|im_end|>", "prefix_end_token")
        self._add_special_token("<|box_end|>", "end_token")
        self._add_special_token("<|endoftext|>", "pad_token")
        self._add_special_token(
            "<|object_ref_start|>", "direct_condition_token"
        )
        self._add_special_token("<|object_ref_end|>", "cot_condition_token")
        self._add_special_token("<|quad_start|>", "noisy_condition_token")
        self._add_special_token("<|quad_end|>", "synth_condition_token")
        super().__init__(vocabulary=vocabulary, merges=merges, **kwargs)
