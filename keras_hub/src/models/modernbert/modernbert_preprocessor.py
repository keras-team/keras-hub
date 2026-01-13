import keras
from keras import ops
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.layers.preprocessing.masked_lm_mask_generator import MaskedLMMaskGenerator
from keras_hub.src.layers.preprocessing.multi_segment_packer import MultiSegmentPacker

@keras_hub_export("keras_hub.models.ModernBertMaskedLMPreprocessor")
class ModernBertMaskedLMPreprocessor(Preprocessor):
    """ModernBERT Masked LM preprocessor.

    This class prepares raw strings for Masked Language Modeling (MLM) using 
    the ModernBERT architecture. It tokenizes the input, packs it with special 
    tokens, and generates masks for training.

    The output of this preprocessor is a tuple `(x, y, sw)`, where `x` is a 
    dictionary containing:
    - `"token_ids"`: The masked token IDs.
    - `"padding_mask"`: A mask for non-padding tokens.
    - `"mask_positions"`: The indices of the tokens that were masked.

    `y` contains the original token IDs for the masked positions, and `sw` 
    contains the sample weights for the loss function.

    Args:
        tokenizer: A `keras_hub.models.ModernBertTokenizer` instance.
        sequence_length: int. The length of the packed sequence.
        mask_selection_rate: float. The probability of masking a token.
        mask_selection_length: int. The maximum number of tokens to mask per 
            sequence.
        **kwargs: Standard `keras.layers.Layer` arguments.

    Examples:
    ```python
    # Load the preprocessor from a preset
    preprocessor = keras_hub.models.ModernBertMaskedLMPreprocessor.from_preset(
        "modernbert_base"
    )

    # Preprocess raw text
    x, y, sw = preprocessor(["The quick brown fox jumps over the dog."])
    ```
    """

    def __init__(
        self,
        tokenizer,
        sequence_length=512,
        mask_selection_rate=0.15,
        mask_selection_length=96,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer

        self.packer = MultiSegmentPacker(
            start_value=tokenizer.end_token_id,
            end_value=tokenizer.end_token_id,
            pad_value=tokenizer.pad_token_id,
            sequence_length=sequence_length,
        )

        self.masker = MaskedLMMaskGenerator(
            mask_selection_rate=mask_selection_rate,
            mask_selection_length=mask_selection_length,
            vocabulary_size=tokenizer.vocabulary_size,
            mask_token_id=tokenizer.mask_token_id,
            unselectable_token_ids=[
                tokenizer.pad_token_id,
                tokenizer.end_token_id,
            ],
        )
        self.sequence_length = sequence_length

    def call(self, x, y=None, sample_weight=None):
        """Transform raw strings into masked token sequences."""
        x = self.tokenizer(x)
        token_ids, padding_mask = self.packer(x)
        
        mask_data = self.masker(token_ids)

        x_masked = mask_data["token_ids"]
        y_labels = mask_data["mask_ids"]
        mask_positions = mask_data["mask_positions"]

        padding_mask = ops.cast(padding_mask, dtype="int32")
        
        return (
            {
                "token_ids": x_masked,
                "padding_mask": padding_mask,
                "mask_positions": mask_positions,
            },
            y_labels,
            mask_data["mask_weights"],
        )
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "mask_selection_rate": self.masker.mask_selection_rate,
            "mask_selection_length": self.masker.mask_selection_length,
        })
        return config
