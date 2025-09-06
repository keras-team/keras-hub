from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.gpt_oss.gpt_oss_backbone import GptOssBackbone
from keras_hub.src.tokenizers.sentence_piece_tokenizer import (
    SentencePieceTokenizer,
)


@keras_hub_export(
    [
        "keras_hub.tokenizers.GptOssTokenizer",
        "keras_hub.models.GptOssTokenizer",
    ]
)
class GptOssTokenizer(SentencePieceTokenizer):
    backbone_cls = GptOssBackbone

    def __init__(self, proto, **kwargs):
        # GPT-OSS, like Mixtral and Llama, typically uses <s> and </s> as special tokens
        # and 0 as the padding token ID.
        self._add_special_token("<s>", "start_token")
        self._add_special_token("</s>", "end_token")
        self.pad_token_id = 0
        super().__init__(proto=proto, **kwargs)
