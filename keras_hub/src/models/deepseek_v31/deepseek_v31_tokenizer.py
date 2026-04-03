"""DeepSeek V31 tokenizer."""

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.deepseek_v31.deepseek_v31_backbone import (
    DeepSeekV31Backbone,
)
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export("keras_hub.tokenizers.DeepSeekV31Tokenizer")
class DeepSeekV31Tokenizer(BytePairTokenizer):
    """Tokenizer for DeepSeek V31 models.

    This tokenizer uses Byte-Pair Encoding (BPE) with the DeepSeek V31
    vocabulary (~128K tokens). It adds special tokens for sequence boundary
    marking:

    - `bos_token` (`<｜begin▁of▁sentence｜>`, id 151646): prepended to every
      sequence during generation.
    - `eos_token` (`<｜end▁of▁sentence｜>`, id 151643): used as the generation
      stop token.
    - `pad_token_id` (0): used for padding batched inputs.

    Args:
        vocabulary: dict or str. Token-to-id mapping as a Python dict, or a
            path to a JSON vocabulary file. Mutually exclusive with `proto`.
        merges: list or str. BPE merge rules as a list of strings `"a b"`, or
            a path to a merges file. Mutually exclusive with `proto`.
        proto: str. Path to a SentencePiece `.model` file. When provided,
            `vocabulary` and `merges` will be extracted automatically.

    Example:

    ```python
    tokenizer = keras_hub.tokenizers.DeepSeekV31Tokenizer.from_preset(
        "deepseek_v31_base"
    )
    tokenizer.tokenize("Hello, world!")
    # [13225, 11, 1879, 0]
    tokenizer.detokenize([[13225, 11, 1879, 0]])
    # ["Hello, world!"]
    ```

    Reference:
     - [DeepSeek-AI et al., 2024](https://arxiv.org/abs/2412.19437)
    """

    backbone_cls = DeepSeekV31Backbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        proto=None,
        **kwargs,
    ):
        # Handle SentencePiece proto: extract vocab and merges before calling
        # super().__init__, since BytePairTokenizer needs them at construction.
        if proto is not None:
            from keras.src.saving import serialization_lib

            if isinstance(proto, str) and serialization_lib.in_safe_mode():
                raise ValueError(
                    "Requested the loading of a SentencePiece proto file "
                    "outside of the model archive. This carries a "
                    "potential risk of loading arbitrary and sensitive files"
                    " and thus it is disallowed "
                    "by default. If you trust the source of the artifact, you "
                    "can override this error by passing `safe_mode=False` to "
                    "the loading function, or calling "
                    "`keras.config.enable_unsafe_deserialization()`."
                )
            try:
                import sentencepiece as spm

                sp = spm.SentencePieceProcessor()
                sp.Load(proto)
                vocabulary = {
                    sp.IdToPiece(i): i for i in range(sp.GetPieceSize())
                }
                merges = []
            except ImportError:
                raise ImportError(
                    "Loading a SentencePiece proto requires the `sentencepiece`"
                    " package. Install it with `pip install sentencepiece`."
                )

        # BytePairTokenizer requires at least one merge rule to initialise its
        # internal StaticHashTable. Inject a harmless placeholder when the
        # merge list is empty (e.g. when loading from a SentencePiece proto).
        if (
            isinstance(merges, list)
            and len(merges) == 0
            and vocabulary is not None
        ):
            merges = ["a b"]

        super().__init__(vocabulary=vocabulary, merges=merges, **kwargs)

        self._add_special_token("<｜begin▁of▁sentence｜>", "bos_token")
        self._add_special_token("<｜end▁of▁sentence｜>", "eos_token")

        self.start_token = "<｜begin▁of▁sentence｜>"
        self.start_token_id = 151646
        self.end_token_id = 151643
        self.pad_token_id = 0
