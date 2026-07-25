import itertools
from functools import partial

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
    """Abstract base class for discrete block-diffusion language models.

    `DiffusionLM` tasks wrap a backbone and preprocessor to implement the full
    outer denoising loop used in discrete block-diffusion generation.  Rather
    than predicting one token at a time, the model iteratively denoises an
    entire canvas of tokens in parallel.

    Subclasses must implement four hook methods:
    - `_encode_prompt`: encode prompt tokens, return (encoder_cache, N).
    - `_prepare_canvas_embeds`: embed current canvas tokens, optionally
      applying self-conditioning from previous step logits.
    - `_decode_canvas_step`: run one decoder forward pass over the canvas.
    - `_canvas_logits`: project decoder hidden states to vocabulary logits.

    The generation loop lives in `generate_step`, which is JIT-compiled via
    `make_generate_function` following the same backend-dispatch pattern used
    by `CausalLM`.

    Args:
        canvas_length: int. Number of tokens in the denoising canvas.
            Defaults to `256`.
        max_denoising_steps: int. Maximum number of denoising iterations per
            canvas block. Defaults to `48`.
        t_min: float. Minimum temperature (applied at the last step).
            Defaults to `0.4`.
        t_max: float. Maximum temperature (applied at the first step).
            Defaults to `0.8`.
    """

    def __init__(
        self,
        *args,
        canvas_length=256,
        max_denoising_steps=48,
        t_min=0.4,
        t_max=0.8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.canvas_length = canvas_length
        self.max_denoising_steps = max_denoising_steps
        self.t_min = t_min
        self.t_max = t_max

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
        """Create or return the compiled generation function.

        The transformer-heavy `_encode_prompt` and `_forward_step` are
        JIT-compiled for each backend.  The outer denoising loop and the
        sampler stay in eager Python so that `EntropyBoundSampler`'s adaptive
        stopping (which converts tensors to Python bools) works correctly.
        """
        if self.generate_function is not None:
            return self.generate_function

        if keras.config.backend() == "openvino":
            from keras_hub.src.utils.openvino_utils import ov_infer

            def wrapped_generate_function(inputs):
                inputs = tree.map_structure(ops.convert_to_numpy, inputs)
                return ov_infer(self, inputs, None, self.generate_step)

            self.generate_function = wrapped_generate_function

        if keras.config.backend() == "torch":
            import torch

            def wrapped_generate_function(inputs):
                with torch.no_grad():
                    return self.generate_step(inputs)

            self.generate_function = wrapped_generate_function

        elif keras.config.backend() == "tensorflow" and not self.run_eagerly:
            jit_compile = getattr(self, "jit_compile", True)
            _encode_fn = tf.function(
                self._encode_prompt, jit_compile=jit_compile
            )
            # tf.function creates separate traces for prev_logits=None (step 0)
            # and prev_logits=tensor (steps 1+), which is standard TF behaviour.
            _step_fn = tf.function(self._forward_step, jit_compile=jit_compile)

            # Precompute temperatures as tf.constant tensors to prevent
            # tf.function from retracing for each unique Python float value.
            _temperatures = [
                tf.constant(
                    self.t_max
                    - (self.t_max - self.t_min)
                    * step
                    / self.max_denoising_steps,
                    dtype=tf.float32,
                )
                for step in range(self.max_denoising_steps)
            ]

            def wrapped_generate_function(inputs):
                encoder_cache, prompt_length = _encode_fn(inputs)
                prompt_padding_mask = inputs.get("padding_mask", None)
                batch_size = ops.shape(inputs["token_ids"])[0]
                canvas = self._init_canvas(batch_size)
                prev_logits = None
                for step in range(self.max_denoising_steps):
                    logits = _step_fn(
                        canvas,
                        encoder_cache,
                        prompt_length,
                        prev_logits,
                        _temperatures[step],
                        prompt_padding_mask,
                    )
                    prev_logits = logits
                    canvas, stop, argmax_canvas = self.sampler(
                        canvas, logits, step
                    )
                    if bool(ops.convert_to_numpy(ops.all(stop))):
                        break
                return ops.cast(argmax_canvas, "int32")

            self.generate_function = wrapped_generate_function

        elif keras.config.backend() == "jax" and not self.run_eagerly:
            import jax

            # Two JIT functions handle the prev_logits=None vs tensor split:
            # JAX cannot trace None as an abstract array, so the first step
            # (no self-conditioning) uses a dedicated function that hardcodes
            # prev_logits=None, and subsequent steps use a second function that
            # accepts prev_logits as a concrete tensor argument.

            def _make_scope_mapping(state):
                sampler_vars, trainable_vars, non_trainable_vars = state
                return itertools.chain(
                    zip(self.sampler.variables, sampler_vars),
                    zip(self.trainable_variables, trainable_vars),
                    zip(self.non_trainable_variables, non_trainable_vars),
                )

            @jax.jit
            def jit_encode(inputs, state):
                with keras.StatelessScope(
                    state_mapping=_make_scope_mapping(state)
                ):
                    return self._encode_prompt(inputs)

            @partial(jax.jit, static_argnums=(2,))
            def jit_step_no_sc(
                canvas,
                enc_cache,
                prompt_len,
                temperature,
                prompt_padding_mask,
                state,
            ):
                with keras.StatelessScope(
                    state_mapping=_make_scope_mapping(state)
                ):
                    return self._forward_step(
                        canvas,
                        enc_cache,
                        prompt_len,
                        None,
                        temperature,
                        prompt_padding_mask=prompt_padding_mask,
                    )

            @partial(jax.jit, static_argnums=(2,))
            def jit_step(
                canvas,
                enc_cache,
                prompt_len,
                prev_logits,
                temperature,
                prompt_padding_mask,
                state,
            ):
                with keras.StatelessScope(
                    state_mapping=_make_scope_mapping(state)
                ):
                    return self._forward_step(
                        canvas,
                        enc_cache,
                        prompt_len,
                        prev_logits,
                        temperature,
                        prompt_padding_mask=prompt_padding_mask,
                    )

            def wrapped_generate_function(inputs):
                state = (
                    [v.value for v in self.sampler.variables],
                    [v.value for v in self.trainable_variables],
                    [v.value for v in self.non_trainable_variables],
                )
                inputs = tree.map_structure(ops.convert_to_tensor, inputs)
                encoder_cache, prompt_length = jit_encode(inputs, state)
                # Convert to a Python int so it can be used as a static
                # argument in jit_step_no_sc/jit_step, avoiding
                # TracerBoolConversionError in _decode_canvas_step.
                prompt_length = int(ops.convert_to_numpy(prompt_length))
                batch_size = ops.shape(inputs["token_ids"])[0]
                canvas = self._init_canvas(batch_size)

                prompt_padding_mask = inputs.get("padding_mask", None)

                # Step 0: no self-conditioning.
                logits = jit_step_no_sc(
                    canvas,
                    encoder_cache,
                    prompt_length,
                    self.t_max,
                    prompt_padding_mask,
                    state,
                )
                canvas, stop, argmax_canvas = self.sampler(canvas, logits, 0)
                prev_logits = logits

                for step in range(1, self.max_denoising_steps):
                    if bool(ops.convert_to_numpy(ops.all(stop))):
                        break
                    temperature = self.t_max - (
                        (self.t_max - self.t_min)
                        * step
                        / self.max_denoising_steps
                    )
                    logits = jit_step(
                        canvas,
                        encoder_cache,
                        prompt_length,
                        prev_logits,
                        temperature,
                        prompt_padding_mask,
                        state,
                    )
                    canvas, stop, argmax_canvas = self.sampler(
                        canvas, logits, step
                    )
                    prev_logits = logits

                return ops.cast(argmax_canvas, "int32")

            self.generate_function = wrapped_generate_function

        else:
            self.generate_function = self.generate_step

        return self.generate_function

    def _init_canvas(self, batch_size):
        """Create the initial random-token canvas `(B, canvas_length)`."""
        vocab_size = self.backbone.vocabulary_size
        canvas = keras.random.randint(
            shape=(batch_size, self.canvas_length),
            minval=0,
            maxval=vocab_size,
            seed=self.sampler.seed_generator,
            dtype="int32",
        )
        return canvas

    def _forward_step(
        self,
        canvas,
        encoder_cache,
        prompt_length,
        prev_logits,
        temperature,
        prompt_padding_mask=None,
    ):
        """Single denoising forward pass — JIT-compilable.

        Does not call the sampler or perform any Python bool conversion, so it
        is safe to wrap with `tf.function` / `jax.jit`.

        Args:
            canvas: int tensor of shape `(B, canvas_length)`.
            encoder_cache: encoder KV cache from `_encode_prompt`.
            prompt_length: int scalar, number of real prompt tokens.
            prev_logits: float tensor `(B, canvas_length, vocab_size)` from the
                previous step, or `None` on the first step.
            temperature: float scalar for logit scaling.
            prompt_padding_mask: bool tensor `(B, prompt_length)` indicating
                which prompt positions are real (True) vs padding (False).
                Used to prevent canvas queries from attending to padding keys.

        Returns:
            Float tensor of shape `(B, canvas_length, vocab_size)`.
        """
        canvas_embeds = self._prepare_canvas_embeds(canvas, prev_logits)
        hidden = self._decode_canvas_step(
            canvas_embeds,
            encoder_cache,
            prompt_length,
            prompt_padding_mask=prompt_padding_mask,
        )
        logits = self._canvas_logits(hidden)
        return ops.cast(logits, "float32") / temperature

    def generate_step(self, inputs):
        """Run one full denoising sequence for a single batch.

        Args:
            inputs: dict. Pre-processed inputs containing at minimum
                `"token_ids"` and `"padding_mask"`.

        Returns:
            A `(B, canvas_length)` int tensor of the final denoised tokens.
        """
        encoder_cache, prompt_length = self._encode_prompt(inputs)
        prompt_padding_mask = inputs.get("padding_mask", None)

        batch_size = ops.shape(inputs["token_ids"])[0]
        canvas = self._init_canvas(batch_size)
        prev_logits = None

        for step in range(self.max_denoising_steps):
            temperature = self.t_max - (
                (self.t_max - self.t_min) * step / self.max_denoising_steps
            )
            logits = self._forward_step(
                canvas,
                encoder_cache,
                prompt_length,
                prev_logits,
                temperature,
                prompt_padding_mask=prompt_padding_mask,
            )
            prev_logits = logits
            canvas, stop, argmax_canvas = self.sampler(canvas, logits, step)
            if bool(ops.convert_to_numpy(ops.all(stop))):
                break

        return ops.cast(argmax_canvas, "int32")

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

    def generate(self, inputs, max_length=None):
        """Generate a denoised canvas given prompt inputs.

        Args:
            inputs: python data, tensor data, or a `tf.data.Dataset`. If a
                `preprocessor` is attached to the model, `inputs` should match
                the structure expected by the `preprocessor` layer. If a
                `preprocessor` is not attached, `inputs` should match the
                structure expected by the `backbone` model.
            max_length: Optional. Not used for diffusion models (canvas length
                is fixed at compile time via `canvas_length`).  Accepted for
                API compatibility with `CausalLM.generate`.

        Returns:
            Decoded string(s) or integer token arrays, depending on whether
            a `preprocessor` is attached.
        """
        if hasattr(self, "sampler") and hasattr(self.sampler, "reset"):
            self.sampler.reset()

        generate_function = self.make_generate_function()

        inputs, input_is_scalar = self._normalize_generate_inputs(inputs)

        if self.preprocessor is not None:
            inputs = [
                self.preprocessor.generate_preprocess(
                    x, sequence_length=max_length
                )
                for x in inputs
            ]

        outputs = [generate_function(x) for x in inputs]

        if self.preprocessor is not None:
            outputs = [
                self.preprocessor.generate_postprocess(x) for x in outputs
            ]

        return self._normalize_generate_outputs(outputs, input_is_scalar)

    def _encode_prompt(self, inputs):
        """Encode the prompt."""
        raise NotImplementedError

    def _prepare_canvas_embeds(self, canvas, prev_logits):
        """Embed the current canvas tokens, applying self-conditioning."""
        raise NotImplementedError

    def _decode_canvas_step(self, canvas_embeds, encoder_cache, prompt_length):
        """Run one decoder forward pass over the canvas."""
        raise NotImplementedError

    def _canvas_logits(self, hidden):
        """Project hidden states to logits."""
        raise NotImplementedError

    def _post_quantize(self, mode, **kwargs):
        super()._post_quantize(mode, **kwargs)
        self.generate_function = None

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "canvas_length": self.canvas_length,
                "max_denoising_steps": self.max_denoising_steps,
                "t_min": self.t_min,
                "t_max": self.t_max,
            }
        )
        return config
