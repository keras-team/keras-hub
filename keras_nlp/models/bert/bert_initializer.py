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
class BertInitializer(keras.initializers.TruncatedNormal):
    """The default intializer used by BERT models.

    This class provides the default initializer used by bert modeling components
    when instantiating new parameters. It is simply
    `keras.initializers.TruncatedNormal` with a stddev of 0.02.
    """

    def __init__(self, mean=0.0, stddev=0.02, seed=None, **kwargs):
        super().__init__(
            mean=mean,
            stddev=stddev,
            seed=seed,
            **kwargs,
        )
