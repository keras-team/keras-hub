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

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.models.gpt2.gpt2_backbone import GPT2Backbone
from keras_nlp.models.gpt2.gpt2_causal_lm_preprocessor import (
    GPT2CausalLMPreprocessor,
)
from keras_nlp.models.gpt2.gpt2_presets import backbone_presets
from keras_nlp.models.task import Task
from keras_nlp.samplers.serialization import get as get_sampler
from keras_nlp.utils.keras_utils import is_xla_compatible
from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.tf_utils import tensor_to_string_list
from keras_nlp.utils.tf_utils import truncate_at_token


@keras_nlp_export("keras_nlp.models.GPT2CausalLM")
class GPT2CausalLM(Task):
    """An end-to-end GPT2 model for causal langauge modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens. This task setup can be used to train the model unsupervised on
    plain text input, or to autoregressively generate plain text similar to
    the data used for training. This task can be used for pre-training or
    fine-tuning a GPT-2 model, simply by calling `fit()`.

    This model has a `generate()` method, which generates text based on a
    prompt. The generation strategy used is controlled by an additional
    `sampler` argument on `compile()`. You can recompile the model with
    different `keras_nlp.samplers` objects to control the generation. By
    default, `"top_k"` sampling will be used.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to raw inputs during
    `fit()`, `predict()`, `evaluate()` and `generate()`. This is done by default
    when creating the model with `from_preset()`.

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

    Use `generate()` to do text generation.
    ```python
    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en")
    gpt2_lm.generate("I want to say", max_length=30)

    # Generate with batched prompts.
    gpt2_lm.generate(["This is a", "Where are you"], max_length=30)
    ```

    Compile the `generate()` function with a custom sampler.
    ```python
    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en")
    gpt2_lm.compile(sampler="greedy")
    gpt2_lm.generate("I want to say", max_length=30)

    gpt2_lm.compile(sampler=keras_nlp.samplers.BeamSampler(num_beams=2))
    gpt2_lm.generate("I want to say", max_length=30)
    ```

    Use `generate()` without preprocessing.
    ```python
    # Prompt the model with `5338, 318` (the token ids for `"Who is"`).
    # Use `"padding_mask"` to indicate values that should not be overridden.
    prompt = {
        "token_ids": tf.constant([[5338, 318, 0, 0, 0]] * 2),
        "padding_mask": tf.constant([[1, 1, 0, 0, 0]] * 2),
    }

    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
        "gpt2_base_en",
        preprocessor=None,
    )
    gpt2_lm.generate(prompt)
    ```

    Call `fit()` on a single batch.
    ```python
    features = ["The quick brown fox jumped.", "I forgot my homework."]
    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en")
    gpt2_lm.fit(x=features, batch_size=2)
    ```

    Call `fit()` without preprocessing.
    ```python
    x = {
        "token_ids": tf.constant([[50256, 1, 2, 3, 4]] * 2),
        "padding_mask": tf.constant([[1, 1, 1, 1, 1]] * 2),
    }
    y = tf.constant([[1, 2, 3, 4, 50256]] * 2)
    sw = tf.constant([[1, 1, 1, 1, 1]] * 2)

    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
        "gpt2_base_en",
        preprocessor=None,
    )
    gpt2_lm.fit(x=x, y=y, sample_weight=sw, batch_size=2)
    ```

    Custom backbone and vocabulary.
    ```python
    features = ["a quick fox.", "a fox quick."]
    vocab = {"<|endoftext|>": 0, "a": 4, "Ġquick": 5, "Ġfox": 6}
    merges = ["Ġ q", "u i", "c k", "ui ck", "Ġq uick"]
    merges += ["Ġ f", "o x", "Ġf ox"]

    tokenizer = keras_nlp.models.GPT2Tokenizer(
        vocabulary=vocab,
        merges=merges,
    )
    preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor(
        tokenizer=tokenizer,
        sequence_length=128,
    )
    backbone = keras_nlp.models.GPT2Backbone(
        vocabulary_size=30552,
        num_layers=4,
        num_heads=4,
        hidden_dim=256,
        intermediate_dim=512,
        max_sequence_length=128,
    )
    gpt2_lm = keras_nlp.models.GPT2CausalLM(
        backbone=backbone,
        preprocessor=preprocessor,
    )
    gpt2_lm.fit(x=features, batch_size=2)
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
        self._sampler = None

        # Default compilation
        self.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.Adam(2e-5),
            metrics=keras.metrics.SparseCategoricalAccuracy(),
            jit_compile=is_xla_compatible(self),
        )

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)

    @classproperty
    def backbone_cls(cls):
        return GPT2Backbone

    @classproperty
    def preprocessor_cls(cls):
        return GPT2CausalLMPreprocessor

    def call_with_cache(
        self,
        token_ids,
        cache,
        cache_index,
    ):
        """Forward pass of `GPT2CausalLM` with cache.

        `call_with_cache` adds an additional forward pass for the model for
        autoregressive inference. Unlike calling the model directly, this method
        allows caching previous key/value Tensors in multi-head attention layer,
        and avoids recomputing the outputs of seen tokens.

        Args:
            token_ids: a dense int Tensor with shape `(batch_size, max_length)`.
            cache: a dense float Tensor, the cache of key and value.
            cache_index: int, or int Tensor. The index of current inputs in the
                whole sequence.

        Returns:
            A (logits, hidden_states, cache) tuple. Where `logits` is the
            language model logits for the input token_ids, `hidden_states` is
            the final hidden representation of the input tokens, and `cache` is
            the decoding cache.
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
        hidden_states = x
        logits = tf.matmul(
            hidden_states,
            self.backbone.get_layer("token_embedding").embeddings,
            transpose_b=True,
        )
        return logits, hidden_states, cache

    def _build_cache(self, token_ids):
        """Build an empty cache for use with `call_with_cache()`."""
        batch_size, max_length = tf.shape(token_ids)[0], tf.shape(token_ids)[1]
        num_layers = self.backbone.num_layers
        num_heads = self.backbone.num_heads
        head_dim = self.backbone.hidden_dim // self.backbone.num_heads
        shape = [batch_size, num_layers, 2, max_length, num_heads, head_dim]
        cache = tf.zeros(shape, dtype=self.compute_dtype)
        # Seed the cache.
        _, hidden_states, cache = self.call_with_cache(token_ids, cache, 0)
        return hidden_states, cache

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
        self._sampler = get_sampler(sampler)
        # Clear the compiled generate function.
        self.generate_function = None

    def make_generate_function(self):
        """Create or return the compiled generation function."""
        if self.generate_function is not None:
            return self.generate_function

        if self.run_eagerly:
            self.generate_function = self.generate_step
        else:
            # `jit_compile` is a property of keras.Model after TF 2.12.
            # Use `getattr()` for backwards compatibility.
            jit_compile = getattr(self, "jit_compile", True)
            self.generate_function = tf.function(
                self.generate_step, jit_compile=jit_compile
            )
        return self.generate_function

    def generate_step(
        self,
        token_ids,
        padding_mask,
        end_token_id=None,
    ):
        """A compilable generation function for a single batch of inputs.

        This function represents the inner, XLA-compilable, generation function
        for a single batch of inputs. It takes in a dense `tf.Tensor` of token
        ids, and return a dense `tf.Tensor` of token ids, and includes no
        preprocessing. This function is wrapped by the `generate()` method.

        Args:
            token_ids: A dense int Tensor, with shape
                `(batch_size, max_length)`. The user provided token ids
                padded to `max_length`.
            padding_mask: A dense boolean Tensor, with the same shape as
                `token_ids`. Positions that are True in the `padding_mask`
                are assumed to be user input and never updated.
            end_token_id: The id of the end token to stop on. If all
                sequences have produced a new `end_token_id`, generation
                will stop.
        """
        # Create and seed cache with a single forward pass.
        hidden_states, cache = self._build_cache(token_ids)
        # Compute the lengths of all user inputted tokens ids.
        row_lengths = tf.math.reduce_sum(
            tf.cast(padding_mask, tf.int32), axis=-1
        )
        # Start at the first index that has no user inputted id.
        index = tf.math.reduce_min(row_lengths)

        def next(prompt, cache, index):
            # The cache index is the index of our previous token.
            cache_index = index - 1
            prompt = tf.slice(prompt, [0, cache_index], [-1, 1])
            logits, hidden_states, cache = self.call_with_cache(
                prompt,
                cache,
                cache_index,
            )
            return (
                tf.squeeze(logits, axis=1),
                tf.squeeze(hidden_states, axis=1),
                cache,
            )

        return self._sampler(
            next=next,
            prompt=token_ids,
            cache=cache,
            index=index,
            mask=padding_mask,
            end_token_id=end_token_id,
            hidden_states=hidden_states,
        )

    def generate(
        self,
        inputs,
        max_length=None,
    ):
        """Generate text given prompt `inputs`.

        This method generates text based on given `inputs`. The sampling method
        used for generation can be set in the `compile` method.

        If `inputs` are a `tf.data.Dataset`, outputs will be generated
        "batch-by-batch" and concatenated. Otherwise, all inputs will be handled
        as a single batch.

        If a `preprocessor` is attached to the model, `inputs` should be
        strings and returned sequences will be strings. Otherwise, inputs should
        be preprocessed before calling `generate()`, and returned sequences will
        be token ids.

        Args:
            inputs: a string `tf.Tensor`, a `tf.data.Dataset` of strings, a
                python string or a list of python strings. If no `preprocessor`
                is attached to the model, inputs should instead be a nested
                `tf.Tensor` or `tf.data.Dataset` with keys `"token_ids"` and
                `"padding_mask"`.
            max_length: Optional. int. The max length of the generated sequence.
                Will default to the max configured `sequence_length` of the
                `preprocessor`. If `preprocessor` is `None`, `inputs` should be
                should be padded to the desired maximum length and this argument
                will be ignored.

        Returns:
            A string or string list if `preprocessor` is set, and a integer
            tensor of token IDs if `preprocessor is None`.
        """
        input_is_scalar = False

        if self.preprocessor is not None:

            def preprocess(x):
                return self.preprocessor(
                    x,
                    sequence_length=max_length,
                    return_labels=False,
                    # We do not append an end token by default during
                    # generation, as generating directly in the same sequence is
                    # the most common workflow. If an end token directly after
                    # a prompt is desired, it can be added to the prompt string.
                    add_end_token=False,
                )

            if not isinstance(inputs, tf.data.Dataset):
                inputs = tf.convert_to_tensor(inputs)
                input_is_scalar = inputs.shape.rank == 0
                inputs = inputs[tf.newaxis] if input_is_scalar else inputs
                # Wrap a list to avoid the overhead of converting to dataset.
                inputs = [preprocess(inputs)]
            else:
                inputs = inputs.map(preprocess, tf.data.AUTOTUNE)
                inputs = inputs.prefetch(tf.data.AUTOTUNE)
        else:
            if not isinstance(inputs, tf.data.Dataset):
                # Wrap a list to avoid the overhead of converting to dataset.
                inputs = [inputs]

        generate_function = self.make_generate_function()
        outputs = []
        for batch in inputs:
            token_ids, padding_mask = batch["token_ids"], batch["padding_mask"]
            # If `preprocessor` is attached, we can stop after end_token_id.
            end_token_id = None
            if self.preprocessor is not None:
                end_token_id = self.preprocessor.tokenizer.end_token_id
            # Run the compiled generate function.
            output = generate_function(token_ids, padding_mask, end_token_id)

            if self.preprocessor is not None:
                # Truncate to ragged by removing tokens after the first
                # generated `end_token_id`.
                output = truncate_at_token(output, end_token_id, padding_mask)
                # Strip start token if added.
                if self.preprocessor.add_start_token:
                    output = output[:, 1:]
                # Detokenize.
                output = self.preprocessor.tokenizer.detokenize(output)
            outputs.append(output)

        outputs = tf.concat(outputs, axis=0)
        outputs = tf.squeeze(outputs, 0) if input_is_scalar else outputs
        # Convert outputs to a friendly pythonic type. For numerical outputs
        # that is numpy, for string outputs that is `list` and `str`.
        if outputs.dtype == tf.string:
            return tensor_to_string_list(outputs)
        return outputs.numpy()
