from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.modernbert.modernbert_backbone import (
    ModernBertBackbone,
)
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.ModernBertTokenizer",
        "keras_hub.models.ModernBertTokenizer",
    ]
)
class ModernBertTokenizer(BytePairTokenizer):
    backbone_cls = ModernBertBackbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        **kwargs,
    ):
        self._add_special_token("[CLS]", "cls_token")
        self._add_special_token("[SEP]", "sep_token")
        self._add_special_token("[PAD]", "pad_token")
        self._add_special_token("[UNK]", "unk_token")
        self._add_special_token("[MASK]", "mask_token")
        # Also add `tokenizer.start_token` and `tokenizer.end_token` for
        # compatibility with other tokenizers.
        self._add_special_token("[CLS]", "start_token")
        self._add_special_token("[SEP]", "end_token")
        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )
