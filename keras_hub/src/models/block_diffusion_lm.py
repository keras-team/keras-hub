import itertools

import keras
from keras import ops
from keras import tree

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.task import Task
from keras_hub.src.samplers.serialization import get as get_sampler

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.models.BlockDiffusionLM")
class BlockDiffusionLM(Task):
    """Base class for discrete block-diffusion language modeling tasks.

    `BlockDiffusionLM` tasks wrap a `keras_hub.models.Backbone` and a
    `keras_hub.models.Preprocessor` to create a model that can be used for
    block-diffusion generation and generative fine-tuning.

    `BlockDiffusionLM` tasks provide an additional, high-level `generate()`
    function which iteratively denoises blocks of tokens in parallel. The
    `compile()` method of all `BlockDiffusionLM` classes contains an additional
    `sampler` argument, which can be used to pass a
    `keras_hub.samplers.Sampler` to control token commitment and re-noising
    during generation.

    When calling `fit()`, tokenized inputs are trained with shifted token
    labels. A task preprocessor may use sample weights to restrict the loss to
    response tokens for supervised fine-tuning.

    All `BlockDiffusionLM` tasks include a `from_preset()` constructor which
    can be used to load a pre-trained config and weights.

    Example:
    ```python
    # Load a DiffusionGemma model with pre-trained weights.
    diffusion_lm = keras_hub.models.BlockDiffusionLM.from_preset(
        "diffusion_gemma_26b_a4b_it",
    )
    diffusion_lm.compile(sampler="entropy_bound")
    diffusion_lm.generate("Keras is a")
    ```
    """

    def compile(
        self,
        optimizer="auto",
        loss="auto",
        *,
        weighted_metrics="auto",
        sampler="entropy_bound",
        **kwargs,
    ):
        """Configures the `BlockDiffusionLM` task for training and generation.

        The `BlockDiffusionLM` task extends the default compilation signature of
        `keras.Model.compile` with defaults for `optimizer`, `loss`, and
        `weighted_metrics`. To override these defaults, pass any value to
        these arguments during compilation.

        The `BlockDiffusionLM` task adds a `sampler` argument to `compile`,
        which controls token commitment and re-noising during `generate()`.

        Args:
            optimizer: `"auto"`, an optimizer name, or a `keras.Optimizer`
                instance. Defaults to `"auto"`.
            loss: `"auto"`, a loss name, or a `keras.losses.Loss` instance.
                Defaults to `"auto"`.
            weighted_metrics: `"auto"`, or a list of metrics. Defaults to
                `"auto"`.
            sampler: A sampler name or a `keras_hub.samplers.Sampler` instance.
                Defaults to `"entropy_bound"`.
            **kwargs: Additional arguments passed to `keras.Model.compile`.
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
        self.sampler = get_sampler(sampler)
        self.generate_function = None

    def make_generate_function(self):
        """Create or return the compiled generation function."""
        if self.generate_function is not None:
            return self.generate_function

        self.generate_function = self.generate_step
        if keras.config.backend() == "openvino":
            from keras_hub.src.utils.openvino_utils import ov_infer

            def wrapped_generate_function(
                inputs,
                max_length=None,
                stop_token_ids=None,
            ):
                inputs = tree.map_structure(ops.convert_to_numpy, inputs)
                return ov_infer(
                    self,
                    inputs,
                    stop_token_ids,
                    lambda x, stops: self.generate_step(
                        x,
                        max_length=max_length,
                        stop_token_ids=stops,
                    ),
                )

            self.generate_function = wrapped_generate_function

        if keras.config.backend() == "torch":
            import torch

            def wrapped_generate_function(
                inputs,
                max_length=None,
                stop_token_ids=None,
            ):
                with torch.no_grad():
                    return self.generate_step(
                        inputs,
                        max_length=max_length,
                        stop_token_ids=stop_token_ids,
                    )

            self.generate_function = wrapped_generate_function

        elif keras.config.backend() == "tensorflow" and not self.run_eagerly:
            jit_compile = getattr(self, "jit_compile", True)
            self.generate_function = tf.function(
                self.generate_step, jit_compile=jit_compile
            )

        elif keras.config.backend() == "jax" and not self.run_eagerly:
            import jax

            def compiled_generate_function(
                inputs, state, max_length, stop_token_ids
            ):
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

                with keras.StatelessScope(state_mapping=mapping) as scope:
                    outputs = self.generate_step(
                        inputs,
                        max_length=max_length,
                        stop_token_ids=stop_token_ids,
                    )

                sampler_variables = []
                for variable in self.sampler.variables:
                    new_value = scope.get_current_value(variable)
                    sampler_variables.append(
                        new_value if new_value is not None else variable
                    )
                return outputs, sampler_variables

            compiled_generate_function = jax.jit(
                compiled_generate_function,
                static_argnames=(
                    "max_length",
                    "stop_token_ids",
                ),
            )

            def wrapped_generate_function(
                inputs,
                max_length=None,
                stop_token_ids=None,
            ):
                if isinstance(stop_token_ids, list):
                    stop_token_ids = tuple(stop_token_ids)
                state = (
                    [v.value for v in self.sampler.variables],
                    [v.value for v in self.trainable_variables],
                    [v.value for v in self.non_trainable_variables],
                )
                inputs = tree.map_structure(ops.convert_to_tensor, inputs)
                outputs, sampler_variables = compiled_generate_function(
                    inputs,
                    state,
                    max_length,
                    stop_token_ids,
                )
                for reference, variable in zip(
                    self.sampler.variables, sampler_variables
                ):
                    reference.assign(variable)
                return outputs

            self.generate_function = wrapped_generate_function

        return self.generate_function

    def generate_step(self):
        """Run generation on a single batch of input."""
        raise NotImplementedError

    def _normalize_generate_inputs(self, inputs):
        """Normalize user input to the generate function.

        This function converts all inputs to tensors, adds a batch dimension if
        necessary, and returns a iterable "dataset like" object (either an
        actual `tf.data.Dataset` or a list with a single batch element).
        """
        if tf and isinstance(inputs, tf.data.Dataset):
            return inputs.as_numpy_iterator(), False

        if self.preprocessor is None:
            return [inputs], False

        def normalize(x):
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

        return [inputs], input_is_scalar

    def _normalize_generate_outputs(self, outputs, input_is_scalar):
        """Normalize user output from the generate function.

        Converts all output to numpy (for integer output) or Python strings
        (for string output). Removes the batch dimension added by
        `_normalize_generate_inputs` when the original input was scalar.
        """

        def normalize(x):
            if isinstance(x[0], list):
                result = []
                for batch in x:
                    for e in batch:
                        result.append(e)
                return result[0] if input_is_scalar else result
            result = ops.concatenate(x, axis=0)
            result = ops.squeeze(result, 0) if input_is_scalar else result
            return ops.convert_to_numpy(result)

        if isinstance(outputs[0], dict):
            normalized = {}
            for key in outputs[0]:
                normalized[key] = normalize([x[key] for x in outputs])
            return normalized
        return normalize(outputs)

    def generate(self, inputs, max_length=None, stop_token_ids="auto"):
        """Generate a denoised canvas given prompt inputs.

        Args:
            inputs: python data, tensor data, or a `tf.data.Dataset`. If a
                `preprocessor` is attached to the model, `inputs` should match
                the structure expected by the `preprocessor` layer. If a
                `preprocessor` is not attached, `inputs` should match the
                structure expected by the `backbone` model.
            max_length: Optional int. Maximum length of the generated sequence.
                Defaults to the concrete model's configured canvas length.
            stop_token_ids: Optional. `None`, `"auto"`, or tuple of token IDs.
                Defaults to `"auto"`, which uses stop IDs configured on the
                model or the preprocessor tokenizer's end token. `None`
                generates until `max_length`.

        Returns:
            Decoded string(s) or integer token arrays, depending on whether
            a `preprocessor` is attached.
        """
        if hasattr(self, "sampler") and hasattr(self.sampler, "reset"):
            self.sampler.reset()

        if max_length is not None and max_length <= 0:
            raise ValueError(
                "`max_length` must be a positive integer. "
                f"Received: max_length={max_length}."
            )

        if stop_token_ids == "auto":
            stop_token_ids = getattr(self, "stop_token_ids", None)
            if stop_token_ids is None:
                if self.preprocessor is None:
                    raise ValueError(
                        "A `preprocessor` or configured stop tokens are "
                        'required if `stop_token_ids="auto"`. Pass '
                        "`stop_token_ids=None` to generate until `max_length`."
                    )
                stop_token_ids = (self.preprocessor.tokenizer.end_token_id,)

        generate_function = self.make_generate_function()

        inputs, input_is_scalar = self._normalize_generate_inputs(inputs)

        if self.preprocessor is not None:
            inputs = [self.preprocessor.generate_preprocess(x) for x in inputs]

        outputs = [
            generate_function(
                x,
                max_length=max_length,
                stop_token_ids=stop_token_ids,
            )
            for x in inputs
        ]

        if self.preprocessor is not None:
            outputs = [
                self.preprocessor.generate_postprocess(x) for x in outputs
            ]

        return self._normalize_generate_outputs(outputs, input_is_scalar)

    def _post_quantize(self, mode, **kwargs):
        super()._post_quantize(mode, **kwargs)
        self.generate_function = None
