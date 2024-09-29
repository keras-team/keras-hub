import itertools
from functools import partial

import keras
from keras import ops
from keras import random

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.task import Task

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.models.TextToImage")
class TextToImage(Task):
    """Base class for text-to-image tasks.

    `TextToImage` tasks wrap a `keras_hub.models.Backbone` and
    a `keras_hub.models.Preprocessor` to create a model that can be used for
    generation and generative fine-tuning.

    `TextToImage` tasks provide an additional, high-level `generate()` function
    which can be used to generate image by token with a string in, image out
    signature.

    All `TextToImage` tasks include a `from_preset()` constructor which can be
    used to load a pre-trained config and weights.

    Example:

    ```python
    # Load a Stable Diffusion 3 backbone with pre-trained weights.
    text_to_image = keras_hub.models.TextToImage.from_preset(
        "stable_diffusion_3_medium",
    )
    text_to_image.generate(
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    )

    # Load a Stable Diffusion 3 backbone at bfloat16 precision.
    text_to_image = keras_hub.models.TextToImage.from_preset(
        "stable_diffusion_3_medium",
        dtype="bfloat16",
    )
    text_to_image.generate(
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    )
    ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Default compilation.
        self.compile()

    @property
    def latent_shape(self):
        return tuple(self.backbone.latent_shape)

    def compile(
        self,
        optimizer="auto",
        loss="auto",
        *,
        metrics="auto",
        **kwargs,
    ):
        """Configures the `TextToImage` task for training.

        The `TextToImage` task extends the default compilation signature of
        `keras.Model.compile` with defaults for `optimizer`, `loss`, and
        `metrics`. To override these defaults, pass any value
        to these arguments during compilation.

        Args:
            optimizer: `"auto"`, an optimizer name, or a `keras.Optimizer`
                instance. Defaults to `"auto"`, which uses the default optimizer
                for the given model and task. See `keras.Model.compile` and
                `keras.optimizers` for more info on possible `optimizer` values.
            loss: `"auto"`, a loss name, or a `keras.losses.Loss` instance.
                Defaults to `"auto"`, where a
                `keras.losses.MeanSquaredError` loss will be applied. See
                `keras.Model.compile` and `keras.losses` for more info on
                possible `loss` values.
            metrics: `"auto"`, or a list of metrics to be evaluated by
                the model during training and testing. Defaults to `"auto"`,
                where a `keras.metrics.MeanSquaredError` will be applied to
                track the loss of the model during training. See
                `keras.Model.compile` and `keras.metrics` for more info on
                possible `metrics` values.
            **kwargs: See `keras.Model.compile` for a full list of arguments
                supported by the compile method.
        """
        # Ref: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py#L410-L414
        if optimizer == "auto":
            optimizer = keras.optimizers.AdamW(
                1e-4, weight_decay=1e-2, epsilon=1e-8, clipnorm=1.0
            )
        if loss == "auto":
            loss = keras.losses.MeanSquaredError()
        if metrics == "auto":
            metrics = [keras.metrics.MeanSquaredError()]
        super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            **kwargs,
        )
        self.generate_function = None

    def generate_step(self, *args, **kwargs):
        """Run generation on batches of input."""
        raise NotImplementedError

    def make_generate_function(self):
        """Create or return the compiled generation function."""
        if self.generate_function is not None:
            return self.generate_function

        self.generate_function = self.generate_step
        if keras.config.backend() == "torch":
            import torch

            def wrapped_function(*args, **kwargs):
                with torch.no_grad():
                    return self.generate_step(*args, **kwargs)

            self.generate_function = wrapped_function
        elif keras.config.backend() == "tensorflow" and not self.run_eagerly:
            self.generate_function = tf.function(
                self.generate_step, jit_compile=self.jit_compile
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
                    outputs = self.generate_step(*args, **kwargs)
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

            self.generate_function = wrapped_function
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
        guidance_scale,
        seed=None,
    ):
        """Generate image based on the provided `inputs` and `negative_inputs`.

        If `inputs` are a `tf.data.Dataset`, outputs will be generated
        "batch-by-batch" and concatenated. Otherwise, all inputs will be
        processed as batches.

        Args:
            inputs: python data, tensor data, or a `tf.data.Dataset`.
            negative_inputs: python data, tensor data, or a `tf.data.Dataset`.
                Unlike `inputs`, these are used as negative inputs to guide the
                generation. If not provided, it defaults to `""` for each input
                in `inputs`.
            num_steps: int. The number of diffusion steps to take.
            guidance_scale: float. The classifier free guidance scale defined in
                [Classifier-Free Diffusion Guidance](
                https://arxiv.org/abs/2207.12598). A higher scale encourages
                generating images more closely related to the prompts, typically
                at the cost of lower image quality.
            seed: optional int. Used as a random seed.
        """
        num_steps = ops.convert_to_tensor(num_steps, "int32")
        guidance_scale = ops.convert_to_tensor(guidance_scale)

        # Setup our three main passes.
        # 1. Preprocessing strings to dense integer tensors.
        # 2. Generate outputs via a compiled function on dense tensors.
        # 3. Postprocess dense tensors to a value range of `[0, 255]`.
        generate_function = self.make_generate_function()

        def preprocess(x):
            return self.preprocessor.generate_preprocess(x)

        # Normalize and preprocess inputs.
        inputs, input_is_scalar = self._normalize_generate_inputs(inputs)
        if negative_inputs is None:
            negative_inputs = [""] * len(inputs)
        negative_inputs, _ = self._normalize_generate_inputs(negative_inputs)

        if self.preprocessor is not None:
            inputs = preprocess(inputs)
            negative_inputs = preprocess(negative_inputs)
        if isinstance(inputs, dict):
            batch_size = len(inputs[list(inputs.keys())[0]])
        else:
            batch_size = len(inputs)

        # Initialize random latents.
        latent_shape = (batch_size,) + self.latent_shape[1:]
        latents = random.normal(latent_shape, dtype="float32", seed=seed)

        # Text-to-image.
        outputs = generate_function(
            latents,
            inputs,
            negative_inputs,
            num_steps,
            guidance_scale,
        )
        return self._normalize_generate_outputs(outputs, input_is_scalar)
