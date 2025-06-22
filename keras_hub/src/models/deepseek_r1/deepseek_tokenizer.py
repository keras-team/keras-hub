from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.deepseek_r1.deepseek_backbone import (
    DeepSeekV3Backbone,
)
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.DeepSeekR1Tokenizer",
        "keras_hub.models.DeepSeekR1Tokenizer",
    ]
)
class DeepSeekR1Tokenizer(BytePairTokenizer):
    backbone_cls = DeepSeekV3Backbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        bos_token="<｜begin▁of▁sentence｜>",
        eos_token="<｜end▁of▁sentence｜>",
        misc_special_tokens={
            "<｜▁pad▁｜>",
        },
        **kwargs,
    ):
        # Note: all special tokens must also appear in "vocabulary"

        self._add_special_token(bos_token, "start_token")
        misc_special_tokens -= {bos_token}
        self._add_special_token(eos_token, "end_token")
        misc_special_tokens -= {eos_token}
        for i, token in enumerate(misc_special_tokens):
            self._add_special_token(token, f"special_token_{i:03d}")

        self.pad_token_id = 2
        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )
