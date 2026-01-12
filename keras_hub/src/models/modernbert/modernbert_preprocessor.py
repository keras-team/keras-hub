import keras
from keras import ops
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.layers.preprocessing.masked_lm_mask_generator import MaskedLMMaskGenerator
from keras_hub.src.models.modernbert.modernbert_tokenizer import (
    ModernBertTokenizer,
)

@keras_hub_export("keras_hub.models.ModernBertMaskedLMPreprocessor")
class ModernBertMaskedLMPreprocessor(Preprocessor):
    """ModernBERT Masked LM preprocessor.

    This preprocessor tokenizes the input and generates masks for Masked Language 
    Modeling using the `MaskedLMMaskGenerator`.

    Args:
        tokenizer: A `ModernBertTokenizer` instance.
        sequence_length: int. The length of the packed contexts.
        mask_selection_rate: float. The probability of selecting a token for masking.
        mask_selection_length: int. The maximum number of tokens to mask.
        **kwargs: Standard `Preprocessor` arguments.
    """

    def __init__(
        self,
        tokenizer,
        sequence_length=512,
        mask_selection_rate=0.15,
        mask_selection_length=96,
        **kwargs,
    ):
        super().__init__(tokenizer=tokenizer, **kwargs)

        self.masker = MaskedLMMaskGenerator(
            mask_selection_rate=mask_selection_rate,
            mask_selection_length=mask_selection_length,
            vocabulary_size=tokenizer.vocabulary_size,
            mask_token_id=tokenizer.mask_token_id,
            unselectable_token_ids=[
                tokenizer.pad_token_id,
                tokenizer.cls_token_id,
                tokenizer.sep_token_id,
            ],
        )
        self.sequence_length = sequence_length

    def pack_inputs(self, inputs):
        """Pad and truncate to the target sequence length."""
        return ops.pad(
            inputs,
            axis=-1,
            constant_values=self.tokenizer.pad_token_id,
        )[:, :self.sequence_length]

    def call(self, x, y=None, sample_weight=None):
        if not isinstance(x, keras.KerasTensor):
            x = self.tokenizer(x)
        
        x = ops.convert_to_tensor(x)
        x = self.pack_inputs(x)
        
        mask_data = self.masker(x)

        x_masked = mask_data["token_ids"]
        y_labels = mask_data["mask_ids"]
        mask_positions = mask_data["mask_positions"]

        padding_mask = ops.not_equal(x_masked, self.tokenizer.pad_token_id)
        
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
        })
        return config