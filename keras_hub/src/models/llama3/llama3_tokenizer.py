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
        bos_token="<|begin_of_text|>",
        eos_token="<|end_of_text|>",
        misc_special_tokens={"<|start_header_id|>", "<|end_header_id|>"},
        **kwargs,
    ):
        # Note: all special tokens must also appear in "vocabulary"

        if bos_token is not None:
            self._add_special_token(bos_token, "start_token")
            if isinstance(misc_special_tokens, (set, list)):
                misc_special_tokens = set(misc_special_tokens) - {bos_token}
        else:
            self.start_token = None
            self.start_token_id = None

        if eos_token is not None:
            self._add_special_token(eos_token, "end_token")
            if isinstance(misc_special_tokens, (set, list)):
                misc_special_tokens = set(misc_special_tokens) - {eos_token}
        else:
            self.end_token = None
            self.end_token_id = None

        if misc_special_tokens:
            for i, token in enumerate(sorted(misc_special_tokens)):
                if vocabulary is None or token in vocabulary:
                    self._add_special_token(token, f"special_token_{i:03d}")

        if eos_token == "<|end_of_text|>" and (
            vocabulary is None or "<|eot_id|>" in vocabulary
        ):
            self._add_special_token("<|eot_id|>", "end_token2")

        self.pad_token_id = 0
        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )
