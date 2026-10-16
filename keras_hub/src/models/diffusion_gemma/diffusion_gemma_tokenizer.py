from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.diffusion_gemma.diffusion_gemma_backbone import (
    DiffusionGemmaBackbone,
)
from keras_hub.src.models.gemma4.gemma4_tokenizer import Gemma4Tokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.DiffusionGemmaTokenizer",
        "keras_hub.models.DiffusionGemmaTokenizer",
    ]
)
class DiffusionGemmaTokenizer(Gemma4Tokenizer):
    """DiffusionGemma tokenizer based on SentencePiece.

    The tokenizer converts strings to integer sequences. The tokenizer provides
    `from_preset()` to load a matching vocabulary.

    Args:
        proto: A path to a SentencePiece proto file or serialized proto bytes.

    Examples:

    ```python
    tokenizer = keras_hub.models.DiffusionGemmaTokenizer.from_preset(
        "diffusion_gemma_26b_a4b_it"
    )
    tokenizer("The quick brown fox jumped.")
    ```
    """

    backbone_cls = DiffusionGemmaBackbone
