import keras
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.masked_lm import MaskedLM
from keras_hub.src.layers.modeling.masked_lm_head import (
    MaskedLMHead,
)
from keras_hub.src.models.modernbert.modernbert_backbone import (
    ModernBertBackbone,
)

@keras_hub_export("keras_hub.models.ModernBertMaskedLM")
class ModernBertMaskedLM(MaskedLM):
    """ModernBERT Masked Language Model Task.

    Args:
        backbone: A `ModernBertBackbone` instance.
        preprocessor: A `ModernBertMaskedLMPreprocessor` or None.
        **kwargs: Standard `MaskedLM` arguments.
    """

    def __init__(
        self,
        backbone,
        preprocessor=None,
        **kwargs,
    ):
        inputs = backbone.input
        sequence_output = backbone(inputs)


        self.mlm_head = MaskedLMHead(
            vocabulary_size=backbone.vocabulary_size,
            embedding_weights=backbone.token_embedding.embeddings,
            activation="gelu", 
            intermediate_activation="gelu",
            name="prediction_head",
        )
        
        # Note: MaskedLMHead in Keras Hub expects (sequence_output, mask_positions)
        # But for generic Task compatibility, we can call it on sequence_output directly
        prediction_logits = self.mlm_head(sequence_output)

        super().__init__(
            backbone=backbone,
            preprocessor=preprocessor,
            outputs=prediction_logits,
            **kwargs,
        )

    @classmethod
    def from_preset(cls, preset, **kwargs):
        return super().from_preset(preset, **kwargs)