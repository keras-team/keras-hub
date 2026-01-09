import keras
from keras import layers
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.masked_lm import MaskedLM
from keras_hub.src.models.modernbert.modernbert_backbone import ( 
    ModernBertBackbone,)
from keras_hub.src.models.modernbert.modernbert_preprocessor import (
    ModernBertPreprocessor,)

@keras_hub_export("keras_hub.models.ModernBertMaskedLM")
class ModernBertMaskedLM(MaskedLM):
    """ModernBERT Masked Language Model.

    A masked language model (MLM) built on top of the `ModernBertBackbone`.
    It uses a linear layer to project the transformer's output back to the
    vocabulary space for token prediction.

    Args:
        backbone: A `keras_hub.models.ModernBertBackbone` instance.
        preprocessor: A `keras_hub.models.ModernBertPreprocessor` or `None`.
        **kwargs: Standard `MaskedLM` arguments.

    Example:
        ```python
        backbone = keras_hub.models.ModernBertBackbone.from_preset("modern_bert_base")
        masked_lm = keras_hub.models.ModernBertMaskedLM(backbone=backbone)
        
        # Predict masked tokens
        input_data = {
            "token_ids": keras.ops.ones((2, 128), dtype="int32"),
            "padding_mask": keras.ops.ones((2, 128), dtype="int32"),
        }
        logits = masked_lm(input_data)
        ```
    """

    def __init__(self, backbone, preprocessor=None, **kwargs):
        super().__init__(backbone=backbone, preprocessor=preprocessor, **kwargs)

        self.prediction_head = layers.Dense(
            backbone.vocabulary_size,
            use_bias=False,
            name="prediction_head",
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
        )

    def call(self, inputs):
        """Forward pass for the MaskedLM.

        Args:
            inputs: A dict with keys "token_ids" and "padding_mask".
        Returns:
            Logits of shape (batch_size, sequence_length, vocabulary_size).
        """

        # Shape: (batch_size, sequence_length, hidden_dim)
        x = self.backbone(inputs)

        # Shape: (batch_size, sequence_length, vocabulary_size)
        return self.prediction_head(x)

    @property
    def backbone_class(self):
        return ModernBertBackbone

    @property
    def preprocessor_class(self):
        return ModernBertPreprocessor