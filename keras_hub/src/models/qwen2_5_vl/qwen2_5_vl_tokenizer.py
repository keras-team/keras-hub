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
        vocabulary: string or dict, maps token to integer ids. If it is a
            string, it should be the file path to a json file. Defaults to
            ``None``.
        merges: string or list, contains the merge rules as "a b" strings.
            If it is a string, it should be the file path to merge rules.
            Checkpoints storing ``["a", "b"]`` pairs (e.g. 32B) are
            normalized to strings during conversion. Defaults to ``None``.

    Examples:

    ```python
    # Unbatched input.
    tokenizer = keras_hub.models.Qwen2_5VLTokenizer.from_preset(
        "qwen2_5_vl_3b_instruct",
    )
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize([[151643, 791, 4320, 14198]])
    ```
    """

    backbone_cls = Qwen2_5VLBackbone
