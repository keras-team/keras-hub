import keras
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.modernbert.modernbert_backbone import (
    ModernBertBackbone,
)
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer

@keras.utils.register_keras_serializable(package="keras_hub")
@keras_hub_export(
    [
        "keras_hub.tokenizers.ModernBertTokenizer",
        "keras_hub.models.ModernBertTokenizer",
    ]
)
class ModernBertTokenizer(BytePairTokenizer):
    """ModernBERT tokenizer based on Byte-Pair Encoding (BPE).

    ModernBERT uses a byte-level BPE tokenizer. This class handles the 
    transformation of raw text into token IDs and manages special tokens 
    such as [PAD], [CLS], and [MASK].

    Args:
        vocabulary: dict or str. A dictionary mapping tokens to IDs, or a path
            to a JSON file containing the vocabulary.
        merges: list or str. A list of BPE merges, or a path to a merges file.
        **kwargs: Standard `BytePairTokenizer` arguments.
    """
    backbone_cls = ModernBertBackbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        **kwargs,
    ):
        # Initialize the base BytePairTokenizer
        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )
        self._add_special_token("<|endoftext|>", "cls_token")
        self._add_special_token("<|endoftext|>", "sep_token")
        self._add_special_token("<|padding|>", "pad_token")
        self._add_special_token("<|endoftext|>", "unk_token")
        self._add_special_token("<mask>", "mask_token")
        
        # Also add `tokenizer.start_token` and `tokenizer.end_token` for
        # compatibility with other tokenizers.
        self._add_special_token("<|endoftext|>", "start_token")
        self._add_special_token("<|endoftext|>", "end_token")

    @property
    def pad_token_id(self):
        """ID of the padding token."""
        return self.token_to_id("<|padding|>")

    @property
    def mask_token_id(self):
        """ID of the mask token."""
        return self.token_to_id("<mask>")

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary": self.vocabulary,
                "merges": self.merges,
            }
        )
        return config
