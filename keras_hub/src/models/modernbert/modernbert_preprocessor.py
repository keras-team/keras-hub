from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.masked_lm_mask_generator import (
    MaskedLMMaskGenerator,
)
from keras_hub.src.layers.preprocessing.multi_segment_packer import (
    MultiSegmentPacker,
)
from keras_hub.src.models.preprocessor import Preprocessor


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
            Defaults to `512`.
        mask_selection_rate: float. The probability of masking a token.
            Defaults to `0.15`.
        mask_selection_length: int. The maximum number of tokens to mask per
            sequence. Defaults to `96`.
        mask_token_rate: float. The fraction of selected tokens replaced with
            the explicit mask token. Defaults to `0.8`.
        random_token_rate: float. The fraction of selected tokens replaced with
             a randomly chosen alternative token. Defaults to `0.1`.
        **kwargs: Additional keyword arguments passed to the parent
            `Preprocessor` layer.

    Examples:
    ```python
    import keras_hub

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
        mask_token_rate=0.8,
        random_token_rate=0.1,
        **kwargs,
    ):
        self.seed = kwargs.pop("seed", None)
        super().__init__(**kwargs)

        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.mask_selection_rate = mask_selection_rate
        self.mask_selection_length = mask_selection_length
        self.mask_token_rate = mask_token_rate
        self.random_token_rate = random_token_rate

        self.packer = MultiSegmentPacker(
            start_value=tokenizer.start_token_id,
            end_value=tokenizer.end_token_id,
            pad_value=tokenizer.pad_token_id,
            sequence_length=sequence_length,
        )

        self.masker = MaskedLMMaskGenerator(
            mask_selection_rate=mask_selection_rate,
            mask_selection_length=mask_selection_length,
            mask_token_rate=mask_token_rate,
            random_token_rate=random_token_rate,
            vocabulary_size=tokenizer.vocabulary_size,
            mask_token_id=tokenizer.mask_token_id,
            unselectable_token_ids=[
                tokenizer.start_token_id,
                tokenizer.end_token_id,
                tokenizer.pad_token_id,
            ],
        )

    def call(
        self,
        x,
        y=None,
        sample_weight=None,
    ):
        """Transform raw strings into masked token sequences."""
        token_ids = self.tokenizer(x)

        packed = self.packer(token_ids)

        if isinstance(packed, dict):
            token_ids = packed["token_ids"]
            segment_ids = packed["segment_ids"]
            padding_mask = packed["padding_mask"]
        else:
            token_ids, segment_ids = packed
            padding_mask = ops.cast(
                token_ids != self.tokenizer.pad_token_id,
                "int32",
            )

        mask_data = self.masker(token_ids)

        features = {
            "token_ids": mask_data["token_ids"],
            "segment_ids": segment_ids,
            "padding_mask": padding_mask,
            "mask_positions": mask_data["mask_positions"],
        }

        labels = mask_data["mask_ids"]
        sample_weights = mask_data["mask_weights"]

        return (
            features,
            labels,
            sample_weights,
        )

    def get_config(self):
        """Returns the serialization configuration of the preprocessor."""
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "mask_selection_rate": self.mask_selection_rate,
                "mask_selection_length": self.mask_selection_length,
                "mask_token_rate": self.mask_token_rate,
                "random_token_rate": self.random_token_rate,
                "seed": self.seed,
            }
        )
        return config
