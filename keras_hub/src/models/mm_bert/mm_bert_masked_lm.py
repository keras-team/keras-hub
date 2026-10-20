from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.mm_bert.mm_bert_backbone import MMBertBackbone
from keras_hub.src.models.mm_bert.mm_bert_masked_lm_preprocessor import (
    MMBertMaskedLMPreprocessor,
)
from keras_hub.src.models.modern_bert.modern_bert_masked_lm import (
    ModernBertMaskedLM,
)


@keras_hub_export("keras_hub.models.MMBertMaskedLM")
class MMBertMaskedLM(ModernBertMaskedLM):
    """mmBERT Masked LM task model.

    The Masked LM model provides a prediction head for the Masked Language
    Modeling task. It is composed of a `keras_hub.models.MMBertBackbone` and a
    prediction head which projects the backbone's hidden states back to the
    vocabulary space. The head is the ModernBERT one, inherited unchanged: a
    `Dense` layer, a GELU activation, a Layer Normalization, and a decoder that
    reuses the backbone's token embedding weights.

    The `tokenizers`-style vocabulary of mmBERT changes only the size of that
    vocabulary (`256000`) and the mask token (`<mask>`, id `4`).

    This model can be used for pre-training or fine-tuning on a specific
    corpus.

    Args:
        backbone: A `keras_hub.models.MMBertBackbone` instance.
        preprocessor: A `keras_hub.models.MMBertMaskedLMPreprocessor`
            instance or `None`. If `None`, this model will not handle input
            preprocessing automatically during `fit()`, `predict()`, or
            `evaluate()`.
        dtype: string or `keras.DTypePolicy`. The precision policy used
            for the model's computations and weights. If `None`, defaults
            to the backbone's dtype policy.

    Examples:
    ```python
    import keras_hub

    # The checkpoint ships `pytorch_model.bin`, so convert it once with
    # `tools/checkpoint_conversion/convert_mm_bert_checkpoints.py` first.
    masked_lm = keras_hub.models.MMBertMaskedLM.from_preset(
        "./mm_bert_base_multi"
    )

    # Predict on raw text strings.
    masked_lm.predict(["The capital of France is <mask>."])
    ```
    """

    backbone_cls = MMBertBackbone
    preprocessor_cls = MMBertMaskedLMPreprocessor
