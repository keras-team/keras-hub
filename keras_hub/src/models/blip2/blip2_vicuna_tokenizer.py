from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.blip2.blip2_backbone import BLIP2Backbone
from keras_hub.src.models.llama.llama_tokenizer import LlamaTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.BLIP2VicunaTokenizer",
        "keras_hub.models.BLIP2VicunaTokenizer",
    ]
)
class BLIP2VicunaTokenizer(LlamaTokenizer):
    """BLIP-2 Vicuna tokenizer (SentencePiece).

    Thin wrapper around `LlamaTokenizer` that associates the Vicuna (LLaMA)
    tokenizer with `BLIP2Backbone` for `from_preset()` support in the
    InstructBLIP-Vicuna pipeline. This tokenizes the prompt fed to the language
    model; the Q-Former instruction is tokenized separately by
    `keras_hub.models.BLIP2QFormerTokenizer`.

    Args:
        proto: Path to a `spiece.model` SentencePiece proto file, or a `bytes`
            object with a serialized SentencePiece proto.
    """

    backbone_cls = BLIP2Backbone

    def __init__(self, proto, **kwargs):
        super().__init__(proto=proto, **kwargs)
        self.padding_side = "right"
        # InstructBLIP-Vicuna's checkpoint sets bos == eos == "</s>" (id 2) with
        # `add_bos_token=False`, so HF prepends id 2 (not "<s>", id 1) to the
        # language-model prompt. Point the start token at "</s>" so the
        # preprocessor's packer reproduces HF's prompt exactly.
        self._add_special_token("</s>", "start_token")
