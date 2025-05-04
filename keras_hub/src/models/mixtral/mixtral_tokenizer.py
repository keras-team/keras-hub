from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.mixtral.mixtral_backbone import MixtralBackbone
from keras_hub.src.tokenizers.sentence_piece_tokenizer import (
    SentencePieceTokenizer,
)


@keras_hub_export(
    [
        "keras_hub.tokenizers.MixtralTokenizer",
        "keras_hub.models.MixtralTokenizer",
    ]
)
class MixtralTokenizer(SentencePieceTokenizer):
    backbone_cls = MixtralBackbone

    def __init__(self, proto, **kwargs):
        self._add_special_token("<s>", "start_token")
        self._add_special_token("</s>", "end_token")
        self.pad_token_id = 0
        super().__init__(proto=proto, **kwargs)
