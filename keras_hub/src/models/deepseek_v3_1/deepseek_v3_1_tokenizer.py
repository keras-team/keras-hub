"""DeepSeek V3.1 Tokenizer."""

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.deepseek_v3_1.deepseek_v3_1_backbone import (
    DeepSeekV3_1Backbone,
)
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export("keras_hub.tokenizers.DeepSeekV3_1Tokenizer")
class DeepSeekV3_1Tokenizer(BytePairTokenizer):
    """Tokenizer for DeepSeek V3.1 models.

    Implements Byte-Pair Encoding (BPE) with the DeepSeek V3.1 vocabulary
    (128K tokens). Handles special tokens for sequence boundaries:
      - BOS: `<｜begin▁of▁sentence｜>` (id: 151646)
      - EOS: `<｜end▁of▁sentence｜>`   (id: 151643)
      - PAD: token id 0

    Args:
        vocabulary: dict or str. Token vocabulary as dict or path to file.
        merges: list or str. BPE merge rules as list or path to file.
        proto: str. Optional path to a SentencePiece model file. If provided,
            vocabulary and merges will be extracted from it.

    Example:
    ```python
    tokenizer = keras_hub.tokenizers.DeepSeekV3_1Tokenizer.from_preset(
        "deepseek_v3_1_base"
    )
    tokens = tokenizer.tokenize(["Hello, world!"])
    ```
    """

    backbone_cls = DeepSeekV3_1Backbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        proto=None,
        **kwargs,
    ):
        # Handle SentencePiece proto loading before any other setup
        if proto is not None:
            kwargs.pop("proto", None)
            try:
                import sentencepiece as spm

                sp = spm.SentencePieceProcessor()
                sp.Load(proto)
                vocabulary = {sp.IdToPiece(i): i for i in range(sp.GetPieceSize())}
                merges = []
            except Exception:
                pass

        # BytePairTokenizer's StaticHashTable cannot handle an empty merges
        # list (zero-length tensors). Inject a dummy merge to bootstrap the
        # table; it won't affect tokenization since "a b" only fires if the
        # vocabulary contains the merged token "ab".
        if isinstance(merges, list) and len(merges) == 0 and vocabulary is not None:
            merges = ["a b"]

        # FIX: Call super().__init__ BEFORE _add_special_token to ensure the
        # parent Layer is fully initialized before we mutate its state.
        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )

        # Special tokens for DeepSeek V3.1 (from official tokenizer config)
        bos_token = "<｜begin▁of▁sentence｜>"
        eos_token = "<｜end▁of▁sentence｜>"

        self._add_special_token(bos_token, "bos_token")
        self._add_special_token(eos_token, "eos_token")

        self.start_token = bos_token
        self.start_token_id = 151646
        self.end_token_id = 151643
        self.pad_token_id = 0
