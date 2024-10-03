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

        self._add_special_token(bos_token, "start_token")
        misc_special_tokens -= {bos_token}
        self._add_special_token(eos_token, "end_token")
        misc_special_tokens -= {eos_token}
        for i, token in enumerate(misc_special_tokens):
            self._add_special_token(token, f"special_token_{i:03d}")

        # Hack:
        # Llama models use the <|end_of_text|> or the <|eot_id|> as the stop
        # token. This info can be read from config when loading a Hugging Face
        # checkpoint but no such config exists for Keras checkpoints.
        # Setting both probable end tokens when no config is availble will
        # make text generation work in all cases as it will stop
        # on both end tokens. However, the packer will always use
        # "<|end_of_text|>" , which will be the wrong eos_token for "instruct"
        # variants of Llama3.
        # TODO: load this correctly from a Keras tokenizer config.
        if eos_token == "<|end_of_text|>":
            self._add_special_token("<|eot_id|>", "end_token2")

        self.pad_token_id = 0
        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )
