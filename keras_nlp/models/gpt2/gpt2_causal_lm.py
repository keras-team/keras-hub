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

from keras_nlp import samplers
from keras_nlp.api_export import keras_nlp_export
from keras_nlp.models.gpt2.gpt2_backbone import GPT2Backbone
from keras_nlp.models.gpt2.gpt2_causal_lm_preprocessor import (
    GPT2CausalLMPreprocessor,
)
from keras_nlp.models.gpt2.gpt2_presets import backbone_presets
from keras_nlp.models.task import Task
from keras_nlp.utils.keras_utils import is_xla_compatible
from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.tf_utils import truncate_at_token


@keras_nlp_export("keras_nlp.models.GPT2CausalLM")
class GPT2CausalLM(Task):
    """An end-to-end GPT2 model for causal langauge modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens the next token based on previous tokens, which is the way GPT2 gets
    pretrained. You can finetune `GPT2CausalLM` to generate text similar to
    the custom dataset.

    This model has a `generate()` method, which generates text based on a
    prompt. The generation strategy used is controlled by an additional
    `sampler` argument on `compile()`. You can recompile the model with
    different `keras_nlp.samplers` objects to control the generation. By
    default, `"top_k"` sampling will be used.

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

    Compile the `generate()` function with custom samplers.
    ```python
    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en")
    gpt2_lm.compile(sampler="top_p")
    gpt2_lm.generate("I want to say", max_length=30)

    gpt2_lm.compile(sampler=keras_nlp.samplers.BeamSampler(num_beams=2))
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
    ds = tf.data.Dataset.from_tensor_slices(features).batch(2)

    # Create a `GPT2CausalLM` and fit your data.
    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
        "gpt2_base_en",
    )
    gpt2_lm.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    gpt2_lm.fit(ds)
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
        self.generate_function = None
        # Private sampler set by compile.
        self._sampler = samplers.get("top_k")

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)

    @classproperty
    def backbone_cls(cls):
        return GPT2Backbone

    @classproperty
    def preprocessor_cls(cls):
        return GPT2CausalLMPreprocessor

    def call_with_cache(self, token_ids, cache, cache_index):
        """Forward pass of `GPT2CausalLM` with cache.

        `call_with_cache` adds an additional forward pass for the model for
        autoregressive inference. Unlike calling the model directly, this method
        allows caching previous key/value Tensors in multi-head attention layer,
        and avoids recomputing the outputs of seen tokens.

        Args:
            token_ids: a dense int Tensor, input token ids.
            cache: a dense float Tensor, the cache of key and value.
            cache_index: int, or int Tensor. The index of current inputs in the
                whole sequence.

        Returns:
            A (logits, cache) tuple. Where the first output is the language
            model logits for the input token_ids and the second output is the
            cache.
        """
        token_embedding = self.backbone.get_layer("token_embedding")(token_ids)
        position_embedding = self.backbone.get_layer("position_embedding")(
            token_embedding, start_index=cache_index
        )
        x = self.backbone.get_layer("embeddings_add")(
            (token_embedding, position_embedding)
        )
        x = self.backbone.get_layer("embeddings_dropout")(x)
        # Each decoder layer has a cache; we update them separately.
        caches = tf.unstack(cache, axis=1)
        for i in range(self.backbone.num_layers):
            current_cache = caches[i]
            x, next_cache = self.backbone.get_layer(f"transformer_layer_{i}")(
                x,
                cache=current_cache,
                cache_index=cache_index,
            )
            caches[i] = next_cache
        cache = tf.stack(caches, axis=1)
        x = self.backbone.get_layer("layer_norm")(x)
        x = tf.matmul(
            x,
            self.backbone.get_layer("token_embedding").embeddings,
            transpose_b=True,
        )
        return x, cache

    def _build_cache(self, prompt):
        """Build an empty cache for use with `call_with_cache()`."""
        batch_size, max_length = tf.shape(prompt)[0], tf.shape(prompt)[1]
        num_layers = self.backbone.num_layers
        num_heads = self.backbone.num_heads
        head_dim = self.backbone.hidden_dim // self.backbone.num_heads
        shape = [batch_size, num_layers, 2, max_length, num_heads, head_dim]
        cache = tf.zeros(shape)
        # Seed the cache.
        _, cache = self.call_with_cache(prompt, cache, 0)
        return cache

    def compile(
        self,
        *args,
        run_eagerly=False,
        jit_compile=True,
        sampler="top_k",
        **kwargs,
    ):
        xla_compatible = is_xla_compatible(self)
        super().compile(
            *args,
            run_eagerly=run_eagerly,
            # Only `jit_compile` if not eager and in a compatible environment.
            jit_compile=jit_compile and xla_compatible and not run_eagerly,
            **kwargs,
        )
        self._sampler = samplers.get(sampler)
        # Clear the compiled generate function.
        self.generate_function = None

    def make_generate_function(self):
        """Create or return the compiled generation function."""
        if self.generate_function is not None:
            return self.generate_function

        def generate_function(prompt, input_mask, min_length):
            # Create and seed cache with a single forward pass.
            cache = self._build_cache(prompt)

            def next(prompt, state, index):
                # The cache index is the index of our previous token.
                cache_index = index - 1
                prompt = tf.slice(prompt, [0, cache_index], [-1, 1])
                logits, state = self.call_with_cache(prompt, state, cache_index)
                return tf.squeeze(logits, axis=1), state

            return self._sampler(
                next=next,
                prompt=prompt,
                state=cache,
                index=min_length,
                mask=input_mask,
                end_token_id=self.preprocessor.tokenizer.end_token_id,
            )

        if self.run_eagerly:
            self.generate_function = generate_function
        else:
            # `jit_compile` is a property of keras.Model after TF 2.12.
            # Use `getattr()` for backwards compatibility.
            jit_compile = getattr(self, "jit_compile", True)
            self.generate_function = tf.function(
                generate_function, jit_compile=jit_compile
            )
        return self.generate_function

    def generate(
        self,
        prompt,
        max_length,
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
                "`self.preprocessor` is `None`, please make sure "
                "`preprocessor` is set before calling `generate`."
            )

        # Tokenize.
        prompt = tf.convert_to_tensor(prompt)
        input_is_scalar = prompt.shape.rank == 0
        prompt = prompt[tf.newaxis] if input_is_scalar else prompt
        prompt = self.preprocessor.tokenizer(prompt)

        # Pad ragged to dense tensors.
        padded_shape = (None, max_length)
        min_length = tf.reduce_min(prompt.row_lengths())
        input_mask = tf.ones_like(prompt, tf.bool).to_tensor(shape=padded_shape)
        prompt = prompt.to_tensor(shape=padded_shape)

        # Run the (possibly compiled) generate function on dense inputs.
        generate_function = self.make_generate_function()
        output = generate_function(prompt, input_mask, min_length)

        # Truncate to ragged by removing tokens after the first end token.
        end_token_id = self.preprocessor.tokenizer.end_token_id
        output = truncate_at_token(output, end_token_id, input_mask)

        # Detokenize.
        output = self.preprocessor.tokenizer.detokenize(output)
        return tf.squeeze(output, 0) if input_is_scalar else output
