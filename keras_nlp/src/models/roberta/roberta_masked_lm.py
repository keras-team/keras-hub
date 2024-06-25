# Copyright 2022 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import keras

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.layers.modeling.masked_lm_head import MaskedLMHead
from keras_nlp.src.models.masked_lm import MaskedLM
from keras_nlp.src.models.roberta.roberta_backbone import RobertaBackbone
from keras_nlp.src.models.roberta.roberta_backbone import (
    roberta_kernel_initializer,
)
from keras_nlp.src.models.roberta.roberta_masked_lm_preprocessor import (
    RobertaMaskedLMPreprocessor,
)


@keras_nlp_export("keras_nlp.models.RobertaMaskedLM")
class RobertaMaskedLM(MaskedLM):
    """An end-to-end RoBERTa model for the masked language modeling task.

    This model will train RoBERTa on a masked language modeling task.
    The model will predict labels for a number of masked tokens in the
    input data. For usage of this model with pre-trained weights, see the
    `from_preset()` method.

    This model can optionally be configured with a `preprocessor` layer, in
    which case inputs can be raw string features during `fit()`, `predict()`,
    and `evaluate()`. Inputs will be tokenized and dynamically masked during
    training and evaluation. This is done by default when creating the model
    with `from_preset()`.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/facebookresearch/fairseq).

    Args:
        backbone: A `keras_nlp.models.RobertaBackbone` instance.
        preprocessor: A `keras_nlp.models.RobertaMaskedLMPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.

    Examples:

    Raw string data.
    ```python
    features = ["The quick brown fox jumped.", "I forgot my homework."]

    # Pretrained language model.
    masked_lm = keras_nlp.models.RobertaMaskedLM.from_preset(
        "roberta_base_en",
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
    # Create a preprocessed dataset where 0 is the mask token.
    features = {
        "token_ids": np.array([[1, 2, 0, 4, 0, 6, 7, 8]] * 2),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1]] * 2),
        "mask_positions": np.array([[2, 4]] * 2)
    }
    # Labels are the original masked values.
    labels = [[3, 5]] * 2

    masked_lm = keras_nlp.models.RobertaMaskedLM.from_preset(
        "roberta_base_en",
        preprocessor=None,
    )

    masked_lm.fit(x=features, y=labels, batch_size=2)
    ```
    """

    backbone_cls = RobertaBackbone
    preprocessor_cls = RobertaMaskedLMPreprocessor

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
            kernel_initializer=roberta_kernel_initializer(),
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
