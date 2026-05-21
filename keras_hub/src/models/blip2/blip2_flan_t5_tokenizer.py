"""BLIP-2 t5-flan Variants sentence piece tokenizer."""

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.blip2.blip2_backbone import BLIP2Backbone
from keras_hub.src.models.t5.t5_tokenizer import T5Tokenizer

@keras_hub_export([
    "keras_hub.tokenizers.BLIP2FlanT5Tokenizer",
    "keras_hub.models.BLIP2FlanT5Tokenizer",
])
class BLIP2FlanT5Tokenizer(T5Tokenizer):
    """BLIP-2 Flan-T5 tokenizer (SentencePiece).

    Thin wrapper around `T5Tokenizer` that associates this tokenizer
    with `BLIP2Backbone` for `from_preset()` support.

    Args:
        proto: Path to `spiece.model` or serialized SentencePiece proto bytes.
    """
    backbone_cls = BLIP2Backbone