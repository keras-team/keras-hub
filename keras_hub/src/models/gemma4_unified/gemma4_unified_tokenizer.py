from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.gemma4.gemma4_tokenizer import Gemma4Tokenizer
from keras_hub.src.models.gemma4_unified.gemma4_unified_backbone import (
    Gemma4UnifiedBackbone,
)


@keras_hub_export(
    [
        "keras_hub.tokenizers.Gemma4UnifiedTokenizer",
        "keras_hub.models.Gemma4UnifiedTokenizer",
    ]
)
class Gemma4UnifiedTokenizer(Gemma4Tokenizer):
    """Gemma4 Unified tokenizer. Identical to Gemma4Tokenizer but registered
    against Gemma4UnifiedBackbone so preset resolution works."""

    backbone_cls = Gemma4UnifiedBackbone
