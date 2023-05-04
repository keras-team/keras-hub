# Copyright 2023 The KerasNLP Authors
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
"""BERT default optimizer."""

from tensorflow import keras

from keras_nlp.api_export import keras_nlp_export


@keras_nlp_export("keras_nlp.models.BertOptimizer")
class BertOptimizer(keras.optimizers.experimental.AdamW):
    """The default optimizer used by BERT modeling tasks.

    This class extends `keras.optimziers.AdamW` with defaults proposed in the
    original BERT release. In particular, gradients will be clipped to a global
    norm of 1.0, and bias and layernorm variables will not be included in the
    `AdamW` weight decay.

    The optimizer will default to a flat learning rate of 5e-5, but can be
    instead passed any learning rate schedule.

    Examples:

    Raw string data.
    ```python
    # Create classifier.
    classifier = keras_nlp.models.BertClassifier.from_preset(
        "bert_base_en_uncased",
    )

    # Customize the default optimizer with a LR schedule and different decay.
    schedule = keras.optimizers.schedules.CosineDecay(5e-5, decay_steps=10_000)
    classifier.compile(
        optimizer=keras_nlp.models.BertOptimizer(
            learning_rate=schedule,
            weight_decay=0.03,
        ),
    )
    ```
    """

    # Constants from https://github.com/google-research/bert/blob/master/optimization.py
    def __init__(
        self,
        learning_rate=5e-5,
        weight_decay=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        global_clipnorm=1.0,
        exclude_bias_from_decay=True,
        exclude_layernorm_from_decay=True,
        **kwargs
    ):
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            global_clipnorm=global_clipnorm,
            **kwargs,
        )
        if exclude_bias_from_decay:
            self.exclude_from_weight_decay(var_names=["bias"])
        if exclude_layernorm_from_decay:
            self.exclude_from_weight_decay(var_names=["gamma"])
            self.exclude_from_weight_decay(var_names=["beta"])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "exclude_bias_from_decay": self.exclude_bias_from_decay,
                "exclude_layernorm_from_decay": self.exclude_layernorm_from_decay,
            }
        )
        return config
