# Copyright 2024 The KerasNLP Authors
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
from functools import partial

import keras
from keras import ops
from keras import random

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.models.task import Task

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_nlp_export("keras_nlp.models.TextToImage")
class TextToImage(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Default compilation.
        self.compile()

    @property
    def latent_shape(self):
        return self.backbone.latent_shape

    def compile(
        self,
        optimizer="auto",
        loss="auto",
        *,
        metrics="auto",
        **kwargs,
    ):
        # TODO: Figure out how to compile.

        # Clear the compiled functions.
        self.generate_function = None

    def _make_function(self, function):
        generated_function = function
        if keras.config.backend() == "torch":
            import torch

            def wrapped_function(*args, **kwargs):
                with torch.no_grad():
                    return function(*args, **kwargs)

            generated_function = wrapped_function
        elif keras.config.backend() == "tensorflow" and not self.run_eagerly:
            generated_function = tf.function(
                function, jit_compile=self.jit_compile
            )
        elif keras.config.backend() == "jax" and not self.run_eagerly:
            import jax

            @partial(jax.jit)
            def compiled_function(state, *args, **kwargs):
                (
                    trainable_variables,
                    non_trainable_variables,
                ) = state
                mapping = itertools.chain(
                    zip(self.trainable_variables, trainable_variables),
                    zip(self.non_trainable_variables, non_trainable_variables),
                )

                with keras.StatelessScope(state_mapping=mapping):
                    outputs = function(*args, **kwargs)
                return outputs

            def wrapped_function(*args, **kwargs):
                # Create an explicit tuple of all variable state.
                state = (
                    # Use the explicit variable.value to preserve the
                    # sharding spec of distribution.
                    [v.value for v in self.trainable_variables],
                    [v.value for v in self.non_trainable_variables],
                )
                outputs = compiled_function(state, *args, **kwargs)
                return outputs

            generated_function = wrapped_function
        return generated_function

    def generate_step(self, *args, **kwargs):
        raise NotImplementedError

    def make_generate_function(self):
        if self.generate_function is not None:
            return self.generate_function
        self.generate_function = self._make_function(self.generate_step)
        return self.generate_function

    def _normalize_generate_inputs(self, inputs):
        """Normalize user input to the generate function.

        This function converts all inputs to tensors, adds a batch dimension if
        necessary, and returns a iterable "dataset like" object (either an
        actual `tf.data.Dataset` or a list with a single batch element).
        """
        if tf and isinstance(inputs, tf.data.Dataset):
            return inputs.as_numpy_iterator(), False

        def normalize(x):
            if x is None:
                x = ""
            if isinstance(x, str):
                return [x], True
            if tf and isinstance(x, tf.Tensor) and x.shape.rank == 0:
                return x[tf.newaxis], True
            return x, False

        if isinstance(inputs, dict):
            for key in inputs:
                inputs[key], input_is_scalar = normalize(inputs[key])
        else:
            inputs, input_is_scalar = normalize(inputs)

        return inputs, input_is_scalar

    def _normalize_generate_outputs(self, outputs, input_is_scalar):
        """Normalize user output from the generate function.

        This function converts all output to numpy with a value range of
        `[0, 255]`. If a batch dimension was added to the input, it is removed
        from the output.
        """

        def normalize(x):
            outputs = ops.clip(ops.divide(ops.add(x, 1.0), 2.0), 0.0, 1.0)
            outputs = ops.cast(ops.round(ops.multiply(outputs, 255.0)), "uint8")
            outputs = ops.convert_to_numpy(outputs)
            if input_is_scalar:
                outputs = outputs[0]
            return outputs

        if isinstance(outputs[0], dict):
            normalized = {}
            for key in outputs[0]:
                normalized[key] = normalize([x[key] for x in outputs])
            return normalized
        return normalize([x for x in outputs])

    def generate(
        self,
        inputs,
        negative_inputs,
        num_steps,
        classifier_free_guidance_scale,
        seed=None,
    ):
        num_steps = int(num_steps)
        classifier_free_guidance_scale = float(classifier_free_guidance_scale)
        # Setup our three main passes.
        # 1. Preprocessing strings to dense integer tensors.
        # 2. Generate outputs via a compiled function on dense tensors.
        # 3. Postprocess dense tensors to a value range of `[0, 255]`.
        generate_function = self.make_generate_function()

        def preprocess(x):
            return self.preprocessor.generate_preprocess(x)

        # Normalize and preprocess inputs.
        inputs, input_is_scalar = self._normalize_generate_inputs(inputs)
        negative_inputs, _ = self._normalize_generate_inputs(negative_inputs)
        token_ids = preprocess(inputs)
        negative_token_ids = preprocess(negative_inputs)

        # Initialize random latents.
        latent_shape = (len(inputs),) + tuple(self.latent_shape)[1:]
        latents = random.normal(latent_shape, dtype="float32", seed=seed)

        # Text-to-image.
        outputs = generate_function(
            latents,
            token_ids,
            negative_token_ids,
            ops.convert_to_tensor(num_steps),
            ops.convert_to_tensor(classifier_free_guidance_scale),
        )
        return self._normalize_generate_outputs(outputs, input_is_scalar)
