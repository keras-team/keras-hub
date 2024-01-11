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

import tensorflow as tf
import tree

from keras_nlp.backend import config
from keras_nlp.backend import keras
from keras_nlp.backend import ops
from keras_nlp.models.task import Task
from keras_nlp.samplers.serialization import get as get_sampler
from keras_nlp.utils.tensor_utils import tensor_to_list


@keras.saving.register_keras_serializable(package="keras_nlp")
class GenerativeTask(Task):
    """Base class for Generative Task models."""

    def compile(
        self,
        *args,
        run_eagerly=False,
        jit_compile=True,
        sampler="top_k",
        **kwargs,
    ):
        xla_compatible = True
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

    def generate_step(self):
        """Run generation on a single batch of input."""
        raise NotImplementedError

    def make_generate_function(self):
        """Create or return the compiled generation function."""
        if self.generate_function is not None:
            return self.generate_function

        self.generate_function = self.generate_step
        if config.backend() == "torch":
            import torch

            def wrapped_generate_function(
                inputs,
                end_token_id=None,
            ):
                with torch.no_grad():
                    return self.generate_step(inputs, end_token_id)

            self.generate_function = wrapped_generate_function
        elif config.backend() == "tensorflow" and not self.run_eagerly:
            # `jit_compile` is a property of keras.Model after TF 2.12.
            # Use `getattr()` for backwards compatibility.
            jit_compile = getattr(self, "jit_compile", True)
            self.generate_function = tf.function(
                self.generate_step, jit_compile=jit_compile
            )
        elif config.backend() == "jax" and not self.run_eagerly:
            import jax

            @jax.jit
            def compiled_generate_function(inputs, end_token_id, state):
                (
                    sampler_variables,
                    trainable_variables,
                    non_trainable_variables,
                ) = state
                mapping = itertools.chain(
                    zip(self._sampler.variables, sampler_variables),
                    zip(self.trainable_variables, trainable_variables),
                    zip(self.non_trainable_variables, non_trainable_variables),
                )

                with keras.StatelessScope(state_mapping=mapping) as scope:
                    outputs = self.generate_step(inputs, end_token_id)

                # Get updated sampler variables from the stateless scope.
                sampler_variables = []
                for v in self._sampler.variables:
                    new_v = scope.get_current_value(v)
                    sampler_variables.append(new_v if new_v is not None else v)
                return outputs, sampler_variables

            def wrapped_generate_function(
                inputs,
                end_token_id=None,
            ):
                # Create an explicit tuple of all variable state.
                state = (
                    self._sampler.variables,
                    # Use the explicit variable.value to preserve the
                    # sharding spec of distribution.
                    [v.value for v in self.trainable_variables],
                    [v.value for v in self.non_trainable_variables],
                )
                inputs = tree.map_structure(ops.convert_to_tensor, inputs)
                outputs, sampler_variables = compiled_generate_function(
                    inputs,
                    end_token_id,
                    state,
                )
                # Only assign the sampler variables (random seeds), as other
                # model variables should never be updated in generation.
                for ref_v, v in zip(self._sampler.variables, sampler_variables):
                    ref_v.assign(v)
                return outputs

            self.generate_function = wrapped_generate_function

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
        """
        # Setup our three main passes.
        # 1. Optionally preprocessing strings to dense integer tensors.
        # 2. Generate new tokens via a compiled function on dense tensors.
        # 3. Optionally postprocess dense integer tensors back to string.
        generate_function = self.make_generate_function()
        end_token_id = None
        if self.preprocessor is not None:
            end_token_id = self.preprocessor.tokenizer.end_token_id

        def preprocess(x):
            return self.preprocessor.generate_preprocess(
                x, sequence_length=max_length
            )

        def generate(x):
            return generate_function(x, end_token_id=end_token_id)

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
                inputs = [preprocess(x) for x in inputs]

        outputs = [generate(x) for x in inputs]

        if self.preprocessor is not None:
            outputs = [postprocess(x) for x in outputs]

        return self._normalize_generate_outputs(outputs, input_is_scalar)
