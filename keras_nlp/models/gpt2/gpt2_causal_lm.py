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
"""GPT2 Causal LM (Language Model)."""

import copy

import tensorflow as tf
from tensorflow import keras

import keras_nlp
from keras_nlp.models.gpt2.gpt2_backbone import GPT2Backbone
from keras_nlp.models.gpt2.gpt2_causal_lm_preprocessor import (
    GPT2CausalLMPreprocessor,
)
from keras_nlp.models.gpt2.gpt2_presets import backbone_presets
from keras_nlp.models.task import Task
from keras_nlp.utils.python_utils import classproperty


@keras.utils.register_keras_serializable(package="keras_nlp")
class ReverseEmbedding(keras.layers.Layer):
    """A layer multiplying model outputs by the token embedding.

    This layer is used to map model outputs to logits over all vocab tokens.
    It's used in `GPT2CausalLM` to calculate next token's probability.

    Args:
        embedding_layer: a `tf.keras.layers.Embedding` instance, the token
            embedding layer.
    """

    def __init__(self, embedding_layer, name="embedding_mapping", **kwargs):
        super().__init__(name=name, **kwargs)
        self.embedding_layer = embedding_layer

    def call(self, inputs):
        return tf.matmul(
            inputs,
            self.embedding_layer.embeddings,
            transpose_b=True,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_layer": keras.layers.serialize(self.embedding_layer),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["embedding_layer"] = keras.layers.deserialize(
            config["embedding_layer"],
        )
        return cls(**config)


@keras.utils.register_keras_serializable(package="keras_nlp")
class GPT2CausalLM(Task):
    """GPT2 Causal LM task model.

    Causal LM is predicting the next token based on previous tokens, which is
    the way GPT2 gets pretrained. Users can finetune `GPT2CausalLM` to generate
    text similar to the custom dataset. `GPT2CausalLM` also has a public method
    `generate()`, which generates text based on given prompt.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to raw inputs during
    `fit()`, `predict()`, and `evaluate()`. This is done by default when
    creating the model with `from_preset()`.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/openai/gpt-2).

    Args:
        backbone: A `keras_nlp.models.GPT2Backbone` instance.
        preprocessor: A `keras_nlp.models.GPT2CausalLMPreprocessor` or `None`.
            If `None`, this model will not apply preprocessing, and inputs
            should be preprocessed before calling the model.

    Examples:

    Use `generate()` method to do text generation.
    ```python
    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en")
    gpt2_lm.generate("I want to say", max_length=30)

    # Generate with batched prompts.
    gpt2_lm.generate(["This is a", "Where are you"], max_length=30)
    ```

    Use a custom sampler for text generation.
    ```python
    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en")

    # Use string identifier to set sampler.
    gpt2_lm.generate("I want to say", max_length=30, sampler="top_p")

    # Construct a sampler instance.
    sampler = keras_nlp.samplers.BeamSampler(num_beams=2)
    gpt2_lm.generate("I want to say", max_length=30, sampler=sampler)
    ```

    Load a pretrained `GPT2CausalLM` and get outputs on raw string inputs.
    ```python
    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en")
    gpt2_lm.predict(["You know this is just a test string"])
    ```

    Load a pretrained GPT2 and fit on a string dataset.
    ```python
    features = [
        "I don't listen to music while coding.",
        "But I watch youtube while coding!",
    ]
    ds = tf.data.Dataset.from_tensor_slices(features)

    # Create a `GPT2CausalLM` and fit your data.
    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
        "gpt2_base_en",
    )
    gpt2_lm.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    gpt2_lm.fit(x=features, batch_size=2)
    ```

    Load a pretrained `GPT2CausalLM` with custom preprocessor, and predict on
    string inputs.
    ```python
    # Use a shorter sequence length.
    preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
        "gpt2_base_en",
        sequence_length=128,
    )

    # Create a `GPT2CausalLM`, using pretrained GPT2 and custom preprocessor.
    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
        "gpt2_base_en",
        preprocessor=preprocessor,
    )
    gpt2_lm.predict(["You know this is still a test string"])
    ```

    Fit your preprocessed data with randomly initialized GPT2. This is useful
    when you want to do data preprocessing inside `tf.data` pipeline.
    ```python
    # Define preprocessed input.
    features = {
        "token_ids": tf.constant(
            [[1, 2, 3, 4, 0, 0]] * 2, shape=(2, 6)
        ),
        "padding_mask": tf.constant(
            [[1, 1, 1, 1, 0, 0]] * 2, shape=(2, 6)
        ),
    }
    labels = tf.constant(
        [[2, 3, 4, 0, 0, 0]] * 2, shape=(2, 6)
    )
    sample_weight = tf.constant(
        [[1, 1, 1, 0, 0, 0]] * 2, shape=(2, 6)
    )

    # Randomly initialize a GPT2 backbone.
    backbone = keras_nlp.models.GPT2Backbone(
        vocabulary_size=50257,
        num_layers=2,
        num_heads=2,
        hidden_dim=128,
        intermediate_dim=256,
        max_sequence_length=128,
    )
    # Create a `GPT2CausalLM` without preprocessor and fit the data.
    gpt2_lm = keras_nlp.models.GPT2CausalLM(backbone, preprocessor=None)
    gpt2_lm.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    gpt2_lm.fit(
        x=features,
        y=labels,
        sample_weight=sample_weight,
        batch_size=2,
    )
    ```

    """

    def __init__(self, backbone, preprocessor=None, **kwargs):
        inputs = backbone.input
        x = backbone(inputs)
        embedding_layer = backbone.get_layer("token_embedding")
        embedding_map_layer = ReverseEmbedding(embedding_layer)
        outputs = embedding_map_layer(x)

        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            include_preprocessing=preprocessor is not None,
            **kwargs,
        )

        self._backbone = backbone
        self._preprocessor = preprocessor

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)

    @classproperty
    def backbone_cls(cls):
        return GPT2Backbone

    @classproperty
    def preprocessor_cls(cls):
        return GPT2CausalLMPreprocessor

    def _get_token_probability(self, prompt, mask):
        model_inputs = {
            "token_ids": prompt,
            "padding_mask": mask,
        }
        return self(model_inputs)

    def generate(
        self,
        prompt,
        max_length,
        sampler="top_k",
    ):
        """Generate text.

        This method generates text based on given `prompt`. Generation will
        continue until `max_length` is met, and all tokens generated after
        `end_token` will be truncated.

        Args:
            prompt: a string, string Tensor or string RaggedTensor. The prompt
                text for generation.
            max_length: int. The max length of generated sequence.
            end_token: string, defaults to "<|endoftext|>", which is the default
                end token of GPT2. The token marking the end of the sequence,
                tokens generated after the end token will be truncated.
            sampler: a string or `keras_nlp.samplers.Sampler` instance. The
                sampler to be used for text generation.
        """
        end_token_id = self.preprocessor.tokenizer.end_token_id

        if isinstance(sampler, str):
            sampler = keras_nlp.samplers.get(sampler)
        if hasattr(self, "jit_compile"):
            sampler.jit_compile = self.jit_compile
        sampler.run_eagerly = self.run_eagerly
        generated = sampler(
            self.preprocessor.tokenizer(prompt),
            self._get_token_probability,
            max_length=max_length,
            end_token_id=end_token_id,
        )
        return self.preprocessor.tokenizer.detokenize(generated)
