from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.llama3.llama3_backbone import Llama3Backbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.Llama3Tokenizer",
        "keras_hub.models.Llama3Tokenizer",
    ]
)
class Llama3Tokenizer(BytePairTokenizer):
    backbone_cls = Llama3Backbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        **kwargs,
    ):
        self._add_special_token("<|begin_of_text|>", "start_token")
        self._add_special_token("<|end_of_text|>", "end_token")
        self._add_special_token("<|eot_id|>", "end_token2")
        self._add_special_token("<|start_header_id|>", "start_header_token")
        self._add_special_token("<|end_header_id|>", "end_header_token")
        self.pad_token_id = 0
        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )
