import keras
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.masked_lm import MaskedLM
from keras_hub.src.models.modernbert.modernbert_backbone import (
    ModernBertBackbone,
)
from keras_hub.src.models.modernbert.modernbert_preprocessor import (
    ModernBertMaskedLMPreprocessor,
)


@keras_hub_export("keras_hub.models.ModernBertMaskedLM")
class ModernBertMaskedLM(MaskedLM):
    """ModernBERT Masked LM task model.

    The Masked LM model provides a prediction head for the Masked Language
    Modeling task. It is composed of a `keras_hub.models.ModernBertBackbone`
    and a prediction head which projects the backbone's hidden states back
    to the vocabulary space.

    This model can be used for pre-training or fine-tuning on a specific
    corpus.

    Args:
        backbone: A `keras_hub.models.ModernBertBackbone` instance.
        preprocessor: A `keras_hub.models.ModernBertMaskedLMPreprocessor`
        instance or `None`. If `None`, this model will not handle input
        preprocessing automatically during `fit()`, `predict()`, or
        `evaluate()`.
        dtype: string or `keras.DTypePolicy`. The precision policy used
        for the model's computations and weights.

    Examples:
    ```python
    import keras_hub
    import numpy as np

    # Pre-trained backbone and preprocessor
    tokenizer = keras_hub.models.ModernBertTokenizer(
        vocabulary="vocab.json",
        merges="merges.txt",
    )
    preprocessor = keras_hub.models.ModernBertMaskedLMPreprocessor(
        tokenizer=tokenizer,
        sequence_length=128,
    )
    backbone = keras_hub.models.ModernBertBackbone(
        vocabulary_size=50368,
        hidden_dim=768,
        intermediate_dim=1152,
        num_layers=22,
        num_heads=12,
    )

    # Instantiate the MaskedLM task model
    masked_lm = keras_hub.models.ModernBertMaskedLM(
        backbone=backbone,
        preprocessor=preprocessor,
    )

    # Predict on raw text strings
    raw_data = ["The quick brown fox [MASK] over the lazy dog."]
    predictions = masked_lm.predict(raw_data)
    ```
    """

    backbone_cls = ModernBertBackbone
    preprocessor_cls = ModernBertMaskedLMPreprocessor

    def __init__(
        self,
        backbone,
        preprocessor=None,
        **kwargs,
    ):
        # === Inputs ===
        inputs = backbone.input

        mask_positions = keras.Input(
            shape=(None,),
            dtype="int32",
            name="mask_positions",
        )

        # Output shape: (batch_size, sequence_length, hidden_dim)
        sequence_output = backbone(inputs)

        masked_sequence_output = ops.take_along_axis(
            sequence_output,
            mask_positions[:, :, None],
            axis=1,
        )

        masked_sequence_output = layers.RMSNormalization(
            epsilon=(backbone.layer_norm_epsilon),
            name="mlm_head_norm",
        )(masked_sequence_output)

        logits = backbone.token_embedding(
            masked_sequence_output,
            reverse=True,
        )

        # === Initialize the MaskedLM base class ===
        super().__init__(
            backbone=backbone,
            preprocessor=preprocessor,
            inputs={
                **inputs,
                "mask_positions": (mask_positions),
            },
            outputs=logits,
            **kwargs,
        )
