import itertools
from functools import partial

import keras
from keras import ops
from keras import random

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.task import Task
from keras_hub.src.utils.keras_utils import standardize_data_format

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.models.ImageToImage")
class ImageToImage(Task):
    """Base class for image-to-image tasks.

    `ImageToImage` tasks wrap a `keras_hub.models.Backbone` and
    a `keras_hub.models.Preprocessor` to create a model that can be used for
    generation and generative fine-tuning.

    `ImageToImage` tasks provide an additional, high-level `generate()` function
    which can be used to generate image by token with a (image, string) in,
    image out signature.

    All `ImageToImage` tasks include a `from_preset()` constructor which can be
    used to load a pre-trained config and weights.

    Example:

    ```python
    # Load a Stable Diffusion 3 backbone with pre-trained weights.
    reference_image = np.ones((1024, 1024, 3), dtype="float32")
    image_to_image = keras_hub.models.ImageToImage.from_preset(
        "stable_diffusion_3_medium",
    )
    image_to_image.generate(
        reference_image,
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    )

    # Load a Stable Diffusion 3 backbone at bfloat16 precision.
    image_to_image = keras_hub.models.ImageToImage.from_preset(
        "stable_diffusion_3_medium",
        dtype="bfloat16",
    )
    image_to_image.generate(
        reference_image,
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    )
    ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Default compilation.
        self.compile()

    @property
    def support_negative_prompts(self):
        """Whether the model supports `negative_prompts` key in `generate()`."""
        return bool(True)

    @property
    def image_shape(self):
        return tuple(self.backbone.image_shape)

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
        """Configures the `ImageToImage` task for training.

        The `ImageToImage` task extends the default compilation signature of
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

        The input format must be one of the following:
        - A dict with "images", "prompts" and/or "negative_prompts" keys
        - A tf.data.Dataset with "images", "prompts" and/or "negative_prompts"
            keys

        The output will be a dict with "images", "prompts" and/or
        "negative_prompts" keys.
        """
        if tf and isinstance(inputs, tf.data.Dataset):
            _inputs = {
                "images": inputs.map(lambda x: x["images"]).as_numpy_iterator(),
                "prompts": inputs.map(
                    lambda x: x["prompts"]
                ).as_numpy_iterator(),
            }
            if self.support_negative_prompts:
                _inputs["negative_prompts"] = inputs.map(
                    lambda x: x["negative_prompts"]
                ).as_numpy_iterator()
            return _inputs, False

        if (
            not isinstance(inputs, dict)
            or "images" not in inputs
            or "prompts" not in inputs
        ):
            raise ValueError(
                '`inputs` must be a dict with "images" and "prompts" keys or a'
                f"tf.data.Dataset. Received: inputs={inputs}"
            )

        def normalize(x):
            if isinstance(x, str):
                return [x], True
            if tf and isinstance(x, tf.Tensor) and x.shape.rank == 0:
                return x[tf.newaxis], True
            return x, False

        def normalize_images(x):
            data_format = getattr(
                self.backbone, "data_format", standardize_data_format(None)
            )
            input_is_scalar = False
            x = ops.convert_to_tensor(x)
            if len(ops.shape(x)) < 4:
                x = ops.expand_dims(x, axis=0)
                input_is_scalar = True
            x = ops.image.resize(
                x,
                (self.backbone.image_shape[0], self.backbone.image_shape[1]),
                interpolation="nearest",
                data_format=data_format,
            )
            return x, input_is_scalar

        def get_dummy_prompts(x):
            dummy_prompts = [""] * len(x)
            if tf and isinstance(x, tf.Tensor):
                return tf.convert_to_tensor(dummy_prompts)
            else:
                return dummy_prompts

        for key in inputs:
            if key == "images":
                inputs[key], input_is_scalar = normalize_images(inputs[key])
            else:
                inputs[key], input_is_scalar = normalize(inputs[key])

        if self.support_negative_prompts and "negative_prompts" not in inputs:
            inputs["negative_prompts"] = get_dummy_prompts(inputs["prompts"])

        return [inputs], input_is_scalar

    def _normalize_generate_outputs(self, outputs, input_is_scalar):
        """Normalize user output from the generate function.

        This function converts all output to numpy with a value range of
        `[0, 255]`. If a batch dimension was added to the input, it is removed
        from the output.
        """

        def normalize(x):
            outputs = ops.concatenate(x, axis=0)
            outputs = ops.clip(ops.divide(ops.add(outputs, 1.0), 2.0), 0.0, 1.0)
            outputs = ops.cast(ops.round(ops.multiply(outputs, 255.0)), "uint8")
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
        num_steps,
        strength,
        guidance_scale=None,
        seed=None,
    ):
        """Generate image based on the provided `inputs`.

        Typically, `inputs` is a dict with `"images"` and `"prompts"` keys.
        `"images"` are reference images within a value range of
        `[-1.0, 1.0]`, which will be resized to `self.backbone.height` and
        `self.backbone.width`, then encoded into latent space by the VAE
        encoder. `"prompts"` are strings that will be tokenized and encoded by
        the text encoder.

        Some models support a `"negative_prompts"` key, which helps steer the
        model away from generating certain styles and elements. To enable this,
        add `"negative_prompts"` to the input dict.

        If `inputs` are a `tf.data.Dataset`, outputs will be generated
        "batch-by-batch" and concatenated. Otherwise, all inputs will be
        processed as batches.

        Args:
            inputs: python data, tensor data, or a `tf.data.Dataset`. The format
                must be one of the following:
                - A dict with `"images"`, `"prompts"` and/or
                    `"negative_prompts"` keys.
                - A `tf.data.Dataset` with `"images"`, `"prompts"` and/or
                    `"negative_prompts"` keys.
            num_steps: int. The number of diffusion steps to take.
            strength: float. Indicates the extent to which the reference
                `images` are transformed. Must be between `0.0` and `1.0`. When
                `strength=1.0`, `images` is essentially ignore and added noise
                is maximum and the denoising process runs for the full number of
                iterations specified in `num_steps`.
            guidance_scale: Optional float. The classifier free guidance scale
                defined in [Classifier-Free Diffusion Guidance](
                https://arxiv.org/abs/2207.12598). A higher scale encourages
                generating images more closely related to the prompts, typically
                at the cost of lower image quality. Note that some models don't
                utilize classifier-free guidance.
            seed: optional int. Used as a random seed.
        """
        num_steps = int(num_steps)
        strength = float(strength)
        guidance_scale = (
            float(guidance_scale) if guidance_scale is not None else None
        )
        if strength < 0.0 or strength > 1.0:
            raise ValueError(
                "`strength` must be between `0.0` and `1.0`. "
                f"Received strength={strength}."
            )
        if guidance_scale is not None and guidance_scale > 1.0:
            guidance_scale = ops.convert_to_tensor(float(guidance_scale))
        else:
            guidance_scale = None
        starting_step = int(num_steps * (1.0 - strength))
        starting_step = ops.convert_to_tensor(starting_step, "int32")
        num_steps = ops.convert_to_tensor(int(num_steps), "int32")

        # Check `inputs` format.
        required_keys = ["images", "prompts"]
        if tf and isinstance(inputs, tf.data.Dataset):
            spec = inputs.element_spec
            if not all(key in spec for key in required_keys):
                raise ValueError(
                    "Expected a `tf.data.Dataset` with the following keys:"
                    f"{required_keys}. Received: inputs.element_spec={spec}"
                )
        else:
            if not isinstance(inputs, dict):
                raise ValueError(
                    "Expected a `dict` or `tf.data.Dataset`. "
                    f"Received: inputs={inputs} of type {type(inputs)}."
                )
            if not all(key in inputs for key in required_keys):
                raise ValueError(
                    "Expected a `dict` with the following keys:"
                    f"{required_keys}. "
                    f"Received: inputs.keys={list(inputs.keys())}"
                )

        # Setup our three main passes.
        # 1. Preprocessing strings to dense integer tensors.
        # 2. Generate outputs via a compiled function on dense tensors.
        # 3. Postprocess dense tensors to a value range of `[0, 255]`.
        generate_function = self.make_generate_function()

        def preprocess(x):
            if self.preprocessor is not None:
                return self.preprocessor.generate_preprocess(x)
            else:
                return x

        def generate(images, x):
            token_ids = x[0] if self.support_negative_prompts else x

            # Initialize noises.
            if isinstance(token_ids, dict):
                arbitrary_key = list(token_ids.keys())[0]
                batch_size = ops.shape(token_ids[arbitrary_key])[0]
            else:
                batch_size = ops.shape(token_ids)[0]
            noise_shape = (batch_size,) + self.latent_shape[1:]
            noises = random.normal(noise_shape, dtype="float32", seed=seed)

            return generate_function(
                images, noises, x, starting_step, num_steps, guidance_scale
            )

        # Normalize and preprocess inputs.
        inputs, input_is_scalar = self._normalize_generate_inputs(inputs)
        if self.support_negative_prompts:
            images = [x["images"] for x in inputs]
            token_ids = [preprocess(x["prompts"]) for x in inputs]
            negative_token_ids = [
                preprocess(x["negative_prompts"]) for x in inputs
            ]
            # Tuple format: (images, (token_ids, negative_token_ids)).
            inputs = [
                x for x in zip(images, zip(token_ids, negative_token_ids))
            ]
        else:
            images = [x["images"] for x in inputs]
            token_ids = [preprocess(x["prompts"]) for x in inputs]
            # Tuple format: (images, token_ids).
            inputs = [x for x in zip(images, token_ids)]

        # Image-to-image.
        outputs = [generate(*x) for x in inputs]
        return self._normalize_generate_outputs(outputs, input_is_scalar)
