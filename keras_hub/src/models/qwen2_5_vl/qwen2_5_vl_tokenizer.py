from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_backbone import (
    Qwen2_5VLBackbone,
)
from keras_hub.src.models.qwen2_vl.qwen2_vl_tokenizer import Qwen2VLTokenizer


@keras_hub_export(
    [
        "keras_hub.models.Qwen2_5VLTokenizer",
        "keras_hub.tokenizers.Qwen2_5VLTokenizer",
    ]
)
class Qwen2_5VLTokenizer(Qwen2VLTokenizer):
    """Qwen2.5-VL tokenizer layer.

    Qwen2.5-VL checkpoints ship the same BPE tokenizer as Qwen2-VL,
    including the same vision special tokens, so this is a name alias for
    preset resolution.

    Args:
        vocabulary: string or dict, maps token to integer ids.
        merges: string or list, contains the merge rules.
    """

    backbone_cls = Qwen2_5VLBackbone
