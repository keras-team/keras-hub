import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.masked_lm_preprocessor import MaskedLMPreprocessor
from keras_hub.src.models.modernbert.modern_bert_backbone import (
    ModernBertBackbone,
)
from keras_hub.src.models.modernbert.modern_bert_tokenizer import (
    ModernBertTokenizer,
)


@keras_hub_export("keras_hub.models.ModernBertMaskedLMPreprocessor")
class ModernBertMaskedLMPreprocessor(MaskedLMPreprocessor):
    """ModernBERT Masked LM preprocessor.

    This preprocessor tokenizes and prepares inputs for masked language
    modeling with ModernBERT. The masking, packing, and serialization logic
    is inherited from `MaskedLMPreprocessor`.

    Args:
        tokenizer: A `keras_hub.models.ModernBertTokenizer` instance.
        sequence_length: int. The length of the packed sequence.
        mask_selection_rate: float. The probability of masking a token.
        mask_selection_length: int. The maximum number of tokens to mask
            per sequence.
        mask_token_rate: float. The fraction of selected tokens replaced
            with the mask token.
        random_token_rate: float. The fraction of selected tokens replaced
            with a randomly selected token.
        **kwargs: Additional keyword arguments passed to
            `MaskedLMPreprocessor`.
    """

    backbone_cls = ModernBertBackbone
    tokenizer_cls = ModernBertTokenizer

    def call(
        self,
        x,
        y=None,
        sample_weight=None,
    ):
        output = super().call(
            x,
            y=y,
            sample_weight=sample_weight,
        )

        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(output)

        # ModernBERT does not use segment IDs.
        del x["segment_ids"]

        return keras.utils.pack_x_y_sample_weight(
            x,
            y,
            sample_weight,
        )
