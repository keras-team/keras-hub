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
"""XLM-RoBERTa task specific models and heads."""

from tensorflow import keras

from keras_nlp.models.roberta import roberta_tasks


@keras.utils.register_keras_serializable(package="keras_nlp")
class XLMRobertaClassifier(roberta_tasks.RobertaClassifier):
    """XLM-RoBERTa encoder model with a classification head.

    Args:
        backbone: A `keras_nlp.models.XLMRoberta` instance.
        num_classes: int. Number of classes to predict.
        hidden_dim: int. The size of the pooler layer.

    Example usage:
    ```python
    # Randomly initialized XLM-RoBERTa encoder
    model = keras_nlp.models.XLMRoberta(
        vocabulary_size=50265,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=12
    )

    # Call classifier on the inputs.
    input_data = {
        "token_ids": tf.random.uniform(
            shape=(1, 12), dtype=tf.int64, maxval=model.vocabulary_size),
        "padding_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)),
    }
    classifier = keras_nlp.models.XLMRobertaClassifier(
        backbone=model,
        num_classes=4,
    )
    logits = classifier(input_data)
    ```
    """

    pass
