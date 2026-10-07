import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.masked_lm_head import MaskedLMHead
from keras_hub.src.models.distil_bert.distil_bert_backbone import (
    DistilBertBackbone,
)
from keras_hub.src.models.distil_bert.distil_bert_backbone import (
    distilbert_kernel_initializer,
)
from keras_hub.src.models.distil_bert.distil_bert_masked_lm_preprocessor import (  # noqa: E501
    DistilBertMaskedLMPreprocessor,
)
from keras_hub.src.models.masked_lm import MaskedLM


@keras_hub_export("keras_hub.models.DistilBertMaskedLM")
class DistilBertMaskedLM(MaskedLM):
    """An end-to-end DistilBERT model for the masked language modeling task.

    This model will train DistilBERT on a masked language modeling task.
    The model will predict labels for a number of masked tokens in the
    input data. For usage of this model with pre-trained weights, see the
    `from_preset()` constructor.

    This model can optionally be configured with a `preprocessor` layer, in
    which case inputs can be raw string features during `fit()`, `predict()`,
    and `evaluate()`. Inputs will be tokenized and dynamically masked during
    training and evaluation. This is done by default when creating the model
    with `from_preset()`.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/huggingface/transformers).

    Args:
        backbone: A `keras_hub.models.DistilBertBackbone` instance.
        preprocessor: A `keras_hub.models.DistilBertMaskedLMPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.

    Examples:

    Raw string data.
    ```python
    features = ["The quick brown fox jumped.", "I forgot my homework."]

    # Pretrained language model.
    masked_lm = keras_hub.models.DistilBertMaskedLM.from_preset(
        "distil_bert_base_en_uncased",
    )
    masked_lm.fit(x=features, batch_size=2)

    # Re-compile (e.g., with a new learning rate).
    masked_lm.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(5e-5),
        jit_compile=True,
    )
    # Access backbone programmatically (e.g., to change `trainable`).
    masked_lm.backbone.trainable = False
    # Fit again.
    masked_lm.fit(x=features, batch_size=2)
    ```

    Preprocessed integer data.
    ```python
    # Create preprocessed batch where 0 is the mask token.
    features = {
        "token_ids": np.array([[1, 2, 0, 4, 0, 6, 7, 8]] * 2),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1]] * 2),
        "mask_positions": np.array([[2, 4]] * 2)
    }
    # Labels are the original masked values.
    labels = [[3, 5]] * 2

    masked_lm = keras_hub.models.DistilBertMaskedLM.from_preset(
        "distil_bert_base_en_uncased",
        preprocessor=None,
    )
    masked_lm.fit(x=features, y=labels, batch_size=2)
    ```
    """

    backbone_cls = DistilBertBackbone
    preprocessor_cls = DistilBertMaskedLMPreprocessor

    def __init__(
        self,
        backbone,
        preprocessor=None,
        **kwargs,
    ):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor
        self.masked_lm_head = MaskedLMHead(
            vocabulary_size=backbone.vocabulary_size,
            token_embedding=backbone.token_embedding,
            intermediate_activation="gelu",
            kernel_initializer=distilbert_kernel_initializer(),
            dtype=backbone.dtype_policy,
            name="mlm_head",
        )

        # === Functional Model ===
        inputs = {
            **backbone.input,
            "mask_positions": keras.Input(
                shape=(None,), dtype="int32", name="mask_positions"
            ),
        }
        backbone_outputs = backbone(backbone.input)
        outputs = self.masked_lm_head(
            backbone_outputs, inputs["mask_positions"]
        )
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )
