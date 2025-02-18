from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen.qwen_backbone import Qwen2Backbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.Qwen2Tokenizer",
        "keras_hub.models.Qwen2Tokenizer",
    ]
)
class Qwen2Tokenizer(BytePairTokenizer):
    backbone_cls = Qwen2Backbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        bos_token=None,
        eos_token="<|endoftext|>",
        misc_special_tokens=set(),
        **kwargs,
    ):
        # Initialize special tokens set
        special_tokens = set()
        
        # Add BOS token if provided
        if bos_token is not None:
            self._add_special_token(bos_token, "start_token")
            special_tokens.add(bos_token)
            misc_special_tokens -= {bos_token}
            
        # Add EOS token
        self._add_special_token(eos_token, "end_token")
        special_tokens.add(eos_token)
        misc_special_tokens -= {eos_token}
        
        # Add misc special tokens
        for i, token in enumerate(misc_special_tokens):
            if token is not None:
                self._add_special_token(token, f"special_token_{i:03d}")
                special_tokens.add(token)

        # Add alternate EOS token if needed
        if eos_token == "<|end_of_text|>":
            self._add_special_token("<|eot_id|>", "end_token2")
            special_tokens.add("<|eot_id|>")

        self.pad_token_id = 0
        
        # Only pass non-None special tokens to parent class
        kwargs["unsplittable_tokens"] = list(special_tokens)
        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )
