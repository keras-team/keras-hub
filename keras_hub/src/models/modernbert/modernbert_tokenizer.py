import keras
from keras_hub.src.api_export import keras_hub_export
# from keras_hub.src.models.modernbert.modernbert_backbone import(
#     ModernBertBackbone,
# )
from modernbert_backbone import(
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

    This tokenizer class is a specialized version of the 
    `keras_hub.tokenizers.BytePairTokenizer` tailored for ModernBERT. It 
    uses a byte-level BPE vocabulary and includes special tokens required 
    for ModernBERT's masking and sequence boundary logic.

    The tokenizer uses `<|endoftext|>` as the primary sequence boundary token 
    (serving as both start, end, and separator) and `<|padding|>` for padding.

    Args:
        vocabulary: dict or str. A dictionary mapping strings to token IDs, or 
            a path to a JSON file containing the vocabulary.
        merges: list or str. A list of merge rules, or a path to a text file 
            containing the merge rules.
        **kwargs: Standard `keras_hub.tokenizers.BytePairTokenizer` arguments.

    Examples:
    ```python
    # Instantiate from local files
    tokenizer = keras_hub.models.ModernBertTokenizer(
        vocabulary="vocab.json",
        merges="merges.txt",
    )
    ```
    """

    backbone_cls = ModernBertBackbone

    def __init__(self, vocabulary=None, merges=None, **kwargs):
        self.pad_token = "<|padding|>"
        self.mask_token = "[MASK]"
        self.end_token = "<|endoftext|>"

        # ModernBERT special tokens should not be split by the BPE process
        kwargs["unsplittable_tokens"] = [self.pad_token, self.mask_token, self.end_token]
        kwargs["add_prefix_space"] = kwargs.get("add_prefix_space", False)

        super().__init__(vocabulary=vocabulary, merges=merges, **kwargs)

    def _add_special_tokens(self):
        """Sets internal aliases for common special tokens."""
        self.cls_token = self.end_token
        self.sep_token = self.end_token
        self.start_token = self.end_token

    @property
    def pad_token_id(self):
        """Returns the ID of the padding token."""
        return self.token_to_id(self.pad_token)

    @property
    def mask_token_id(self):
        """Returns the ID of the mask token."""
        return self.token_to_id(self.mask_token)
    
    @property
    def vocabulary_size(self):
        """Returns the total number of tokens in the vocabulary."""
        return len(self.vocabulary)

    @property
    def end_token_id(self):
        """Returns the ID of the end-of-text token (<|endoftext|>)."""
        return self.token_to_id(self.end_token)

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocabulary": self.vocabulary, 
            "merges": self.merges
        })
        return config