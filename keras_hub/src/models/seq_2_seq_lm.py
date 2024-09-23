# Copyright 2024 The KerasHub Authors
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
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM


@keras_hub_export("keras_hub.models.Seq2SeqLM")
class Seq2SeqLM(CausalLM):
    """Base class for sequence to sequence language modeling tasks.

    `Seq2SeqLM` tasks wrap a `keras_hub.models.Backbone` and
    a `keras_hub.models.Preprocessor` to create a model that can be used for
    generation and generative fine-tuning, when generation is conditioned on
    additional input sequence in a sequence-to-sequence setting.

    `Seq2SeqLM` tasks provide an additional, high-level `generate()` function
    which can be used to auto-regressively sample an output sequence token by
    token. The `compile()` method of `Seq2SeqLM` classes contains an additional
    `sampler` argument, which can be used to pass a `keras_hub.samplers.Sampler`
    to control how the predicted distribution will be sampled.

    When calling `fit()`, each input should contain an input and output
    sequence. The model will be trained to predict the output sequence
    token-by-token using a causal mask, similar to a `keras_hub.models.CausalLM`
    task. Unlike the `CausalLM` task, an input sequence must be passed, and
    can be attended to in full by all tokens in the output sequence.

    All `Seq2SeqLM` tasks include a `from_preset()` constructor which can be
    used to load a pre-trained config and weights.

    Example:
    ```python
    # Load a Bart backbone with pre-trained weights.
    seq_2_seq_lm = keras_hub.models.Seq2SeqLM.from_preset(
        "bart_base_en",
    )
    seq_2_seq_lm.compile(sampler="top_k")
    # Generate conditioned on the `"The quick brown fox."` as an input sequence.
    seq_2_seq_lm.generate("The quick brown fox.", max_length=30)
    ```
    """

    # TODO: fill in during https://github.com/keras-team/keras-hub/pull/1425
