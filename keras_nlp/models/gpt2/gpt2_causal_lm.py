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
from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice

import keras_nlp
from keras_nlp.models.gpt2.gpt2_backbone import GPT2Backbone
from keras_nlp.models.gpt2.gpt2_causal_lm_preprocessor import (
    GPT2CausalLMPreprocessor,
)
from keras_nlp.models.gpt2.gpt2_presets import backbone_presets
from keras_nlp.models.task import Task
from keras_nlp.utils.python_utils import classproperty


@keras.utils.register_keras_serializable(package="keras_nlp")
class GPT2CausalLM(Task):
    """An end-to-end GPT2 model for causal langauge modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens the next token based on previous tokens, which is the way GPT2 gets
    pretrained. You can finetune `GPT2CausalLM` to generate text similar to
    the custom dataset. `GPT2CausalLM` also has a method `generate()`, which
    generates text based on given prompt.

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

    Map raw string to languages model logit predictions.
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
    gpt2_lm.fit(ds, batch_size=2)
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

    def __init__(
        self,
        backbone,
        preprocessor=None,
        **kwargs,
    ):
        inputs = backbone.input
        x = backbone(inputs)
        # Use token embedding weights to project from the token representation
        # to vocabulary logits.
        outputs = tf.matmul(
            x,
            backbone.token_embedding.embeddings,
            transpose_b=True,
        )

        # Instantiate using Functional API Model constructor.
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            include_preprocessing=preprocessor is not None,
            **kwargs,
        )

        self.backbone = backbone
        self.preprocessor = preprocessor

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)

    @classproperty
    def backbone_cls(cls):
        return GPT2Backbone

    @classproperty
    def preprocessor_cls(cls):
        return GPT2CausalLMPreprocessor

    def call_with_cache(self, token_ids, padding_mask, cache, cache_index=None):
        """Forward pass of `GPT2CausalLM` with cache.

        `call_with_cache` adds an additional forward pass for the model for
        autoregressive inference. Unlike calling the model directly, this method
        allows caching previous key/value Tensors in multi-head attention layer,
        and avoids recomputing the outputs of seen tokens.

        Args:
            token_ids: a dense int Tensor, input token ids.
            padding_mask: a dense bool Tensor, input padding mask.
            cache: a dense float Tensor, the cache of key and value.
            cache_index: int, or int Tensor, defaults to None. If set, it
                represents the index of current inputs in the whole sequence.

        Returns:
            A (logits, cache) tuple. Where the first output is the language
            model logits for the input token_ids and the second output is the
            cache.
        """
        transformer_layers = []
        for i in range(self.backbone.num_layers):
            transformer_layers.append(
                self.backbone.get_layer(f"transformer_layer_{i}")
            )
        layer_norm = self.backbone.get_layer("layer_norm")
        token_embeddings = self.backbone.get_layer("token_embedding")
        position_embeddings = self.backbone.get_layer("position_embedding")
        embeddings_add = self.backbone.get_layer("embeddings_add")
        embeddings_dropout = self.backbone.get_layer("embeddings_dropout")

        token_embedding = token_embeddings(token_ids)
        if cache_index is None:
            position_embedding = position_embeddings(token_embedding)
        else:
            position_embedding = position_embeddings.position_embeddings[
                cache_index, :
            ]
        x = embeddings_add((token_embedding, position_embedding))
        x = embeddings_dropout(x)
        # Each `TransformerDecoder` layer has a cache, we update separately.
        caches = tf.unstack(cache, axis=1)
        for i, transformer_layer in enumerate(transformer_layers):
            current_cache = caches[i]
            x, current_cache = transformer_layer(
                x,
                decoder_padding_mask=padding_mask,
                cache=current_cache,
                cache_index=cache_index,
            )
            caches[i] = current_cache
        cache = tf.stack(caches, axis=1)
        x = layer_norm(x)
        x = tf.matmul(
            x,
            self.backbone.token_embedding.embeddings,
            transpose_b=True,
        )
        return x, cache

    def build_initial_cache(self, initial_inputs, max_length):
        """Build initial cache based on the prompt.

        This method should be called before the decoding loop to build the
        initial cache. The cache is of shape [batch_size, `self.num_layers`, 2
        max_length, `self.num_heads`, `self.hidden_dim // self.num_heads`].
        The first dim represents it's a key or value in multi-head attention.

        Args:
            initial_inputs: a dense Tensor, the initial inputs to the decoding
                loop.
            max_length: int, the max length of the generated sequence.

        Returns:
            A (logits, cache) tuple. The first output is the language
            model logits for the input token_ids and the second output is the
            cache.
        """
        token_ids = initial_inputs["token_ids"]
        padding_mask = initial_inputs["padding_mask"]

        if max_length < self.backbone.max_sequence_length:
            token_ids = token_ids[:, :max_length]
            padding_mask = padding_mask[:, :max_length]

        batch_size = tf.shape(token_ids)[0]
        outputs = tf.zeros(
            [batch_size, max_length, self.backbone.vocabulary_size]
        )
        cache = tf.zeros(
            [
                batch_size,
                self.backbone.num_layers,
                2,
                max_length,
                self.backbone.num_heads,
                self.backbone.hidden_dim // self.backbone.num_heads,
            ],
        )

        output, cache = self.call_with_cache(
            token_ids,
            padding_mask,
            cache=cache,
        )
        outputs = dynamic_update_slice(outputs, output, [0, 0, 0])
        return outputs, cache

    def _get_token_probability(
        self,
        prompt,
        mask,
        cache=None,
        cache_index=None,
    ):
        model_inputs = {
            "token_ids": prompt,
            "padding_mask": mask,
        }
        if cache_index is None and cache is None:
            return self(model_inputs)
        if cache_index is not None:
            batch_size = tf.shape(prompt)[0]
            prompt = tf.slice(prompt, [0, cache_index], [batch_size, 1])
            mask = mask[:, : cache_index + 1]
        output, cache = self.call_with_cache(
            prompt,
            mask,
            cache,
            cache_index,
        )
        return output, cache

    def generate(
        self,
        prompt,
        max_length,
        sampler="top_k",
    ):
        """Generate text.

        This method generates text based on given `prompt`. Generation will
        continue until `max_length` is met, and all tokens generated after
        `end_token` will be truncated. The sampling approach used can be
        controlled via the sampler argument.

        Args:
            prompt: a string, string Tensor or string RaggedTensor. The prompt
                text for generation.
            max_length: int. The max length of generated sequence.
            sampler: a string or `keras_nlp.samplers.Sampler` instance. The
                sampler to be used for text generation.
        """
        if self.preprocessor is None:
            raise ValueError(
                "`self.preprocessor` is None, please make sure `preprocessor` "
                "is set before calling `generate`."
            )
        end_token_id = self.preprocessor.tokenizer.end_token_id

        sampler = keras_nlp.samplers.get(sampler)
        if hasattr(self, "jit_compile"):
            # `jit_compile` is a public property as of tf 2.12. hasattr is for
            # backward compat.
            sampler.jit_compile = self.jit_compile
        sampler.run_eagerly = self.run_eagerly
        x, _, _ = self.preprocessor(prompt)
        if len(x["token_ids"].shape) == 1:
            x["token_ids"] = x["token_ids"][tf.newaxis, :]
            x["padding_mask"] = x["padding_mask"][tf.newaxis, :]
        _, cache = self.build_initial_cache(x, max_length)
        next_token_probability = self._get_token_probability
        generated = sampler(
            self.preprocessor.tokenizer(prompt),
            next_token_probability,
            max_length=max_length,
            end_token_id=end_token_id,
            cache=cache,
        )
        return self.preprocessor.tokenizer.detokenize(generated)
