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

import itertools

import keras
from keras import ops
from keras import tree

from keras_nlp.src import samplers
from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.models.task import Task
from keras_nlp.src.utils.tensor_utils import tensor_to_list

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_nlp_export("keras_nlp.models.CausalLM")
class CausalLM(Task):
    """Base class for generative language modeling tasks.

    `CausalLM` tasks wrap a `keras_nlp.models.Backbone` and
    a `keras_nlp.models.Preprocessor` to create a model that can be used for
    generation and generative fine-tuning.

    `CausalLM` tasks provide an additional, high-level `generate()` function
    which can be used to auto-regressively sample a model token by token with a
    string in, string out signature. The `compile()` method of all `CausalLM`
    classes contains an additional `sampler` argument, which can be used to pass
    a `keras_nlp.samplers.Sampler` to control how the predicted distribution
    will be sampled.

    When calling `fit()`, the tokenized input will be predicted token-by-token
    with a causal mask applied, which gives both a pre-training and supervised
    fine-tuning setup for controlling inference-time generation.

    All `CausalLM` tasks include a `from_preset()` constructor which can be used
    to load a pre-trained config and weights.

    Example:
    ```python
    # Load a GPT2 backbone with pre-trained weights.
    causal_lm = keras_nlp.models.CausalLM.from_preset(
        "gpt2_base_en",
    )
    causal_lm.compile(sampler="top_k")
    causal_lm.generate("Keras is a", max_length=64)

    # Load a Mistral instruction tuned checkpoint at bfloat16 precision.
    causal_lm = keras_nlp.models.CausalLM.from_preset(
        "mistral_instruct_7b_en",
        dtype="bfloat16",
    )
    causal_lm.compile(sampler="greedy")
    causal_lm.generate("Keras is a", max_length=64)
    ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler = None
        self.generate_function = None
        # Default compilation.
        self.compile()

    def build_cache(self, batch_size, max_length):
        """Builds an empty cache for use with `call_with_cache`.

        Args:
            batch_size: int. The size of the batch for generation.
            max_length: int. The maximum sequence length for the cache.

        Returns:
            A cache Tensor, the exact shape will depend on the model.
        """
        raise NotImplementedError

    def call_with_cache(self, token_ids, cache, index):
        """Forward pass with cache for generation.

        `call_with_cache` adds an additional forward pass for the model for
        autoregressive inference. Unlike calling the model directly, this method
        allows caching previous key/value results in multi-head attention layer,
        and avoids recomputing the outputs of seen tokens.

        Args:
            token_ids: a dense int Tensor with shape `(batch_size, n)`, where
                `n` is some sequence length less than or equal to the max
                length of the cache. Usually `n` is either the full cache
                length, to "prefill" the prompt cache values, or `1`, to predict
                single token id.
            cache: a dense float Tensor. The cache of key and value projections
                used in the attention layers of the model. The exact shape will
                depend on the model.
            index: int, or int Tensor. The index of the first token of
                `token_ids` in the entire generated sequence.

        Returns:
            A `(logits, hidden_states, cache)` tuple. Where `logits` is the
            language model logits for the input token_ids, `hidden_states` is
            the final hidden representation of the input tokens, and `cache` is
            the updated decoding cache.
        """
        raise NotImplementedError

    def compile(
        self,
        optimizer="auto",
        loss="auto",
        *,
        weighted_metrics="auto",
        sampler="top_k",
        **kwargs,
    ):
        """Configures the `CausalLM` task for training and generation.

        The `CausalLM` task extends the default compilation signature of
        `keras.Model.compile` with defaults for `optimizer`, `loss`, and
        `weighted_metrics`. To override these defaults, pass any value
        to these arguments during compilation.

        The `CausalLM` task adds a new `sampler` to `compile`, which can be used
        to control the sampling strategy used with the `generate` function.

        Note that because training inputs include padded tokens which are
        excluded from the loss, it is almost always a good idea to compile with
        `weighted_metrics` and not `metrics`.

        Args:
            optimizer: `"auto"`, an optimizer name, or a `keras.Optimizer`
                instance. Defaults to `"auto"`, which uses the default optimizer
                for the given model and task. See `keras.Model.compile` and
                `keras.optimizers` for more info on possible `optimizer` values.
            loss: `"auto"`, a loss name, or a `keras.losses.Loss` instance.
                Defaults to `"auto"`, where a
                `keras.losses.SparseCategoricalCrossentropy` loss will be
                applied for the token classification `CausalLM` task. See
                `keras.Model.compile` and `keras.losses` for more info on
                possible `loss` values.
            weighted_metrics: `"auto"`, or a list of metrics to be evaluated by
                the model during training and testing. Defaults to `"auto"`,
                where a `keras.metrics.SparseCategoricalAccuracy` will be
                applied to track the accuracy of the model at guessing masked
                token values. See `keras.Model.compile` and `keras.metrics` for
                more info on possible `weighted_metrics` values.
            sampler: A sampler name, or a `keras_nlp.samplers.Sampler` instance.
                Configures the sampling method used during `generate()` calls.
                See `keras_nlp.samplers` for a full list of built-in sampling
                strategies.
            **kwargs: See `keras.Model.compile` for a full list of arguments
                supported by the compile method.
        """
        if optimizer == "auto":
            optimizer = keras.optimizers.Adam(2e-5)
        if loss == "auto":
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        if weighted_metrics == "auto":
            weighted_metrics = [keras.metrics.SparseCategoricalAccuracy()]
        super().compile(
            optimizer=optimizer,
            loss=loss,
            weighted_metrics=weighted_metrics,
            **kwargs,
        )
        self.sampler = samplers.serialization.get(sampler)
        # Clear the compiled generate function.
        self.generate_function = None

    def generate_step(
        self,
        inputs,
        end_token_id=None,
    ):
        """Run an entire generation loop on a single input batch."""
        data, index = self.prefill(inputs)

        def cond(data, index):
            return self.is_decoding(
                data=data,
                index=index,
                end_token_id=end_token_id,
            )

        def body(data, index):
            return self.decode(data, index)

        data, _ = ops.while_loop(
            cond,
            body,
            (data, index),
        )
        return self.finish_decoding(data)

    def stateless_generate_step(
        self,
        state,
        inputs,
        stop_token_ids=None,
    ):
        """Stateless version of `generate_step()` for use with Jax."""
        with self.generate_stateless_scope(state) as scope:
            data, index = self.prefill(inputs)
        state = self.update_generate_state(state, scope)

        def cond(state, data, index):
            return self.is_decoding(
                data=data,
                index=index,
                stop_token_ids=stop_token_ids,
            )

        def body(state, data, index):
            with self.generate_stateless_scope(state) as scope:
                data, index = self.decode(data, index)
            state = self.update_generate_state(state, scope)
            return state, data, index

        state, data, index = ops.while_loop(
            cond,
            body,
            (state, data, index),
        )
        # Only return sampler variables from generation. Weights do not change,
        # and returning them across the compilation boundary is slow.
        sampler_variables = state[0]
        return sampler_variables, self.finish_decoding(data)

    def prefill(self, data):
        """Run inference on the entire input sequence to seed generate data."""
        # Create an empty cache.
        batch_size, max_length = ops.shape(data["token_ids"])
        cache = self.build_cache(batch_size, max_length)
        # Run a forward pass with the full padded token id sequence.
        logits, hidden_states, cache = self.call_with_cache(
            token_ids=data["token_ids"],
            cache=cache,
            index=0,
        )
        # Update our data dict.
        data = {
            **data,
            "cache": cache,
            "hidden_states": hidden_states,
        }
        # Add sampling beams, other sampling state.
        data = self.sampler.start(data)
        # Compute the lengths of all user inputted tokens ids.
        row_lengths = ops.sum(data["padding_mask"], axis=-1)
        # Start at the last index that has all user inputted ids.
        index = ops.min(row_lengths) - 1
        # Generate one token from the logits we just computed.
        data = self.sampler.next(
            data=data,
            index=index,
            logits=logits[:, index, :],
        )
        return data, index + 1

    def is_decoding(self, data, index, stop_token_ids=None):
        """Returns true if decoding should continue."""
        return self.sampler.has_next(
            data=data,
            index=index,
            stop_token_ids=stop_token_ids,
        )

    def decode(self, data, index):
        """Sample a single token of output."""
        # Run a forward pass with a single token id, and full length cache.
        logits, hidden_states, cache = self.call_with_cache(
            token_ids=data["token_ids"][:, index][:, None],
            cache=data["cache"],
            index=index,
        )
        # Update our data dict.
        data = {
            **data,
            "cache": cache,
            "hidden_states": ops.slice_update(
                data["hidden_states"], [0, index, 0], hidden_states
            ),
        }
        # Generate the next token.
        data = self.sampler.next(
            data=data,
            index=index,
            logits=logits[:, 0, :],
        )
        return data, index + 1

    def finish_decoding(self, data):
        # Remove sampling beams, other sampling state.
        data = self.sampler.finish(data)
        return {
            "token_ids": data["token_ids"],
            "padding_mask": data["padding_mask"],
        }

    def get_generate_state(self):
        """Get a tuple of all model state used during generation."""
        return (
            self.sampler.variables,
            [v.value for v in self.trainable_variables],
            [v.value for v in self.non_trainable_variables],
        )

    def update_generate_state(self, state, scope):
        """Updates sampler variables given a `StatelessScope`."""
        # Update all sampler variables.
        sampler_variables = []
        for v in self.sampler.variables:
            new_v = scope.get_current_value(v)
            sampler_variables.append(new_v if new_v is not None else v)
        return (sampler_variables,) + state[1:]

    def generate_stateless_scope(self, state):
        """Get stateless scope for using model state without side effect."""
        (
            sampler_variables,
            trainable_variables,
            non_trainable_variables,
        ) = state
        mapping = itertools.chain(
            zip(self.sampler.variables, sampler_variables),
            zip(self.trainable_variables, trainable_variables),
            zip(self.non_trainable_variables, non_trainable_variables),
        )
        return keras.StatelessScope(state_mapping=mapping)

    def make_generate_function(self):
        """Create or return the compiled generation function."""
        if self.generate_function is not None:
            return self.generate_function

        self.generate_function = self.generate_step
        if keras.config.backend() == "torch":
            import torch

            def wrapped_generate_function(
                data,
                stop_token_ids=None,
            ):
                with torch.no_grad():
                    return self.generate_step(data, stop_token_ids)

            self.generate_function = wrapped_generate_function
        elif keras.config.backend() == "tensorflow" and not self.run_eagerly:
            self.generate_function = tf.function(
                self.generate_step, jit_compile=self.jit_compile
            )
        elif keras.config.backend() == "jax":
            import jax

            if self.run_eagerly:
                compiled_generate_step = self.stateless_generate_step
            else:
                compiled_generate_step = jax.jit(
                    self.stateless_generate_step,
                    static_argnames=["stop_token_ids"],
                )

            # Wrap the compiled function to do state passing.
            def wrapped_generate_step(
                data,
                stop_token_ids=None,
            ):
                if stop_token_ids is not None:
                    stop_token_ids = tuple(stop_token_ids)
                sample_variables, data = compiled_generate_step(
                    self.get_generate_state(),
                    data,
                    stop_token_ids=stop_token_ids,
                )
                for ref_v, v in zip(self.sampler.variables, sample_variables):
                    ref_v.assign(v)
                return data

            self.generate_function = wrapped_generate_step

        return self.generate_function

    def _normalize_generate_inputs(
        self,
        inputs,
    ):
        """Normalize user input to the generate function.

        This function coverts all inputs to tensors, adds a batch dimension if
        necessary, and returns a iterable "dataset like" object (either an
        actual `tf.data.Dataset` or a list with a single batch element).
        """
        input_is_scalar = False

        if isinstance(inputs, tf.data.Dataset):
            return inputs, input_is_scalar

        def normalize(x):
            x_is_scalar = False
            if isinstance(x, str) or isinstance(x, list):
                x = tf.convert_to_tensor(x)

            if isinstance(x, tf.Tensor) and x.shape.rank == 0:
                x_is_scalar = True
                x = x[tf.newaxis]

            return x, x_is_scalar

        if isinstance(inputs, dict):
            for key in inputs:
                inputs[key], input_is_scalar = normalize(inputs[key])
        else:
            inputs, input_is_scalar = normalize(inputs)

        # We avoid converting to a dataset purely for speed, for a single batch
        # of input, creating a dataset would add significant overhead.
        return [inputs], input_is_scalar

    def _normalize_generate_outputs(
        self,
        outputs,
        input_is_scalar,
    ):
        """Normalize user output from the generate function.

        This function converts all output to numpy (for integer output), or
        python strings (for string output). If a batch dimension was added to
        the input, it is removed from the output (so generate can be string in,
        string out).
        """

        def normalize(x):
            if isinstance(x[0], list):
                outputs = []
                for batch in x:
                    for e in batch:
                        outputs.append(e)
                return outputs[0] if input_is_scalar else outputs
            if isinstance(x[0], tf.Tensor) and x[0].dtype == tf.string:
                outputs = tf.concat(x, axis=0)
                outputs = tf.squeeze(outputs, 0) if input_is_scalar else outputs
                return tensor_to_list(outputs)
            outputs = ops.concatenate(x, axis=0)
            outputs = ops.squeeze(outputs, 0) if input_is_scalar else outputs
            return ops.convert_to_numpy(outputs)

        if isinstance(outputs[0], dict):
            normalized = {}
            for key in outputs[0]:
                normalized[key] = normalize([x[key] for x in outputs])
            return normalized
        return normalize([x for x in outputs])

    def generate(
        self,
        inputs,
        max_length=None,
        stop_token_ids="auto",
    ):
        """Generate text given prompt `inputs`.

        This method generates text based on given `inputs`. The sampling method
        used for generation can be set via the `compile()` method.

        If `inputs` are a `tf.data.Dataset`, outputs will be generated
        "batch-by-batch" and concatenated. Otherwise, all inputs will be handled
        as a single batch.

        If a `preprocessor` is attached to the model, `inputs` will be
        preprocessed inside the `generate()` function and should match the
        structure expected by the `preprocessor` layer (usually raw strings).
        If a `preprocessor` is not attached, inputs should match the structure
        expected by the `backbone`. See the example usage above for a
        demonstration of each.

        Args:
            inputs: python data, tensor data, or a `tf.data.Dataset`. If a
                `preprocessor` is attached to the model, `inputs` should match
                the structure expected by the `preprocessor` layer. If a
                `preprocessor` is not attached, `inputs` should match the
                structure expected the `backbone` model.
            max_length: Optional. int. The max length of the generated sequence.
                Will default to the max configured `sequence_length` of the
                `preprocessor`. If `preprocessor` is `None`, `inputs` should be
                should be padded to the desired maximum length and this argument
                will be ignored.
            stop_token_ids: Optional. `None`, "auto", or tuple of token ids. Defaults
                to "auto" which uses the `preprocessor.tokenizer.end_token_id`.
                Not specifying a processor will produce an error. None stops
                generation after generating `max_length` tokens. You may also
                specify a list of token id's the model should stop on. Note that
                sequences of tokens will each be interpreted as a stop token,
                multi-token stop sequences are not supported.
        """
        # Setup our three main passes.
        # 1. Optionally preprocessing strings to dense integer tensors.
        # 2. Generate new tokens via a compiled function on dense tensors.
        # 3. Optionally postprocess dense integer tensors back to string.
        generate_function = self.make_generate_function()

        if self.preprocessor is None and stop_token_ids == "auto":
            raise ValueError(
                'A `preprocessor` must be attached to the model if `stop_token_ids="auto"`. '
                "Currently `preprocessor=None`. To call `generate()` with preprocessing "
                "detached, either pass `stop_token_ids=None` to always generate until "
                "`max_length` or pass a tuple of token ids that should terminate generation "
                "as `stop_token_ids`."
            )
        elif stop_token_ids == "auto":
            stop_token_ids = [self.preprocessor.tokenizer.end_token_id]

        def preprocess(x):
            return self.preprocessor.generate_preprocess(
                x, sequence_length=max_length
            )

        def generate(x):
            x = tree.map_structure(ops.convert_to_tensor, x)
            return generate_function(x, stop_token_ids=stop_token_ids)

        def postprocess(x):
            return self.preprocessor.generate_postprocess(x)

        # Normalize inputs, apply our three passes, and normalize outputs.
        inputs, input_is_scalar = self._normalize_generate_inputs(inputs)

        if self.preprocessor is not None:
            if isinstance(inputs, tf.data.Dataset):
                inputs = inputs.map(preprocess, tf.data.AUTOTUNE)
                inputs = inputs.prefetch(tf.data.AUTOTUNE)
            else:
                # Fast path for non-dataset, single-batch input.
                inputs = [preprocess(data) for data in inputs]

        outputs = [generate(x) for x in inputs]

        if self.preprocessor is not None:
            outputs = [postprocess(data) for data in outputs]

        return self._normalize_generate_outputs(outputs, input_is_scalar)
