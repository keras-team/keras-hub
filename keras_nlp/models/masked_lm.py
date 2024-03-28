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
from keras_nlp.api_export import keras_nlp_export
from keras_nlp.models.task import Task


@keras_nlp_export("keras_nlp.models.MaskedLM")
class MaskedLM(Task):
    """Base class for masked language modeling tasks.

    `MaskedLM` tasks wrap a `keras_nlp.models.Backbone` and
    a `keras_nlp.models.Preprocessor` to create a model that can be used for
    unsupervised fine-tuning with a masked language modeling loss.

    When calling `fit()`, all input will be tokenized, and random tokens in
    the input sequence will be masked. These positions of these masked tokens
    will be fed as an additional model input, and the original value of the
    tokens predicted by the model outputs.

    All `MaskedLM` tasks include a `from_preset()` constructor which can be used
    to load a pre-trained config and weights.

    Example:
    ```python
    # Load a Bert MaskedLM with pre-trained weights.
    masked_lm = keras_nlp.models.MaskedLM.from_preset(
        "bert_base_en",
    )
    masked_lm.fit(train_ds)
    ```
    """
