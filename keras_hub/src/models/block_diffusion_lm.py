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
        canvas_length=256,
        max_denoising_steps=48,
        t_min=0.4,
        t_max=0.8,
        *args,
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
        """Configures the `DiffusionLM` task for training and generation.

        The `DiffusionLM` task extends the default compilation signature of
        `keras.Model.compile` with defaults for `optimizer`, `loss`, and
        `weighted_metrics`. To override these defaults, pass any value to
        these arguments during compilation.

        The `DiffusionLM` task adds a `sampler` argument to `compile`, which
        controls token commitment and re-noising during `generate()`.

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
        if sampler == "entropy_bound":
            from keras_hub.src.samplers.entropy_bound_sampler import (
                EntropyBoundSampler,
            )

            sampler = EntropyBoundSampler(
                vocabulary_size=self.backbone.vocabulary_size
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
                    / max(self.max_denoising_steps - 1, 1),
                    dtype=tf.float32,
                )
                for step in range(self.max_denoising_steps)
            ]

            def wrapped_generate_function(inputs):
                encoder_cache, prompt_length = _encode_fn(inputs)
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
                    )
                    prev_logits = logits
                    canvas, stop = self.sampler(canvas, logits, step)
                    if stop:
                        break
                return ops.cast(ops.argmax(prev_logits, axis=-1), "int32")

            self.generate_function = wrapped_generate_function

        elif keras.config.backend() == "jax" and not self.run_eagerly:
            import jax

            # Two JIT functions handle the prev_logits=None vs tensor split:
            # JAX cannot trace None as an abstract array, so the first step
            # (no self-conditioning) uses a dedicated function that hardcodes
            # prev_logits=None, and subsequent steps use a second function that
            # accepts prev_logits as a concrete tensor argument.

            def _make_scope_mapping(self, state):
                trainable_vars, non_trainable_vars = state
                return list(
                    itertools.chain(
                        zip(self.trainable_variables, trainable_vars),
                        zip(self.non_trainable_variables, non_trainable_vars),
                    )
                )

            @jax.jit
            def jit_encode(inputs, state):
                with keras.StatelessScope(
                    state_mapping=_make_scope_mapping(self, state)
                ):
                    return self._encode_prompt(inputs)

            @jax.jit
            def jit_step_no_sc(
                canvas, enc_cache, prompt_len, temperature, state
            ):
                with keras.StatelessScope(
                    state_mapping=_make_scope_mapping(self, state)
                ):
                    return self._forward_step(
                        canvas, enc_cache, prompt_len, None, temperature
                    )

            @jax.jit
            def jit_step(
                canvas, enc_cache, prompt_len, prev_logits, temperature, state
            ):
                with keras.StatelessScope(
                    state_mapping=_make_scope_mapping(self, state)
                ):
                    return self._forward_step(
                        canvas, enc_cache, prompt_len, prev_logits, temperature
                    )

            def wrapped_generate_function(inputs):
                state = (
                    [v.value for v in self.trainable_variables],
                    [v.value for v in self.non_trainable_variables],
                )
                inputs = tree.map_structure(ops.convert_to_tensor, inputs)
                encoder_cache, prompt_length = jit_encode(inputs, state)
                batch_size = ops.shape(inputs["token_ids"])[0]
                canvas = self._init_canvas(batch_size)

                # Step 0: no self-conditioning.
                logits = jit_step_no_sc(
                    canvas, encoder_cache, prompt_length, self.t_max, state
                )
                canvas, stop = self.sampler(canvas, logits, 0)
                prev_logits = logits

                for step in range(1, self.max_denoising_steps):
                    if stop:
                        break
                    temperature = self.t_max - (
                        (self.t_max - self.t_min)
                        * step
                        / max(self.max_denoising_steps - 1, 1)
                    )
                    logits = jit_step(
                        canvas,
                        encoder_cache,
                        prompt_length,
                        prev_logits,
                        temperature,
                        state,
                    )
                    canvas, stop = self.sampler(canvas, logits, step)
                    prev_logits = logits

                return ops.cast(ops.argmax(prev_logits, axis=-1), "int32")

            self.generate_function = wrapped_generate_function

        else:
            self.generate_function = self.generate_step

        return self.generate_function

    def _init_canvas(self, batch_size):
        """Create the initial random-token canvas of shape (B, canvas_length).

        Tokens are sampled uniformly from the vocabulary.  The vocabulary size
        is obtained from the backbone's `vocabulary_size` attribute.
        """
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
        self, canvas, encoder_cache, prompt_length, prev_logits, temperature
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

        Returns:
            Float tensor of shape `(B, canvas_length, vocab_size)`.
        """
        canvas_embeds = self._prepare_canvas_embeds(canvas, prev_logits)
        hidden = self._decode_canvas_step(
            canvas_embeds, encoder_cache, prompt_length
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

        batch_size = ops.shape(inputs["token_ids"])[0]
        canvas = self._init_canvas(batch_size)
        prev_logits = None

        for step in range(self.max_denoising_steps):
            temperature = self.t_max - (
                (self.t_max - self.t_min)
                * step
                / max(self.max_denoising_steps - 1, 1)
            )
            logits = self._forward_step(
                canvas, encoder_cache, prompt_length, prev_logits, temperature
            )
            prev_logits = logits
            canvas, stop = self.sampler(canvas, logits, step)
            if stop:
                break

        return ops.cast(ops.argmax(prev_logits, axis=-1), "int32")

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
        generate_function = self.make_generate_function()

        def normalize(x):
            if isinstance(x, str):
                return [x], True
            if tf and isinstance(x, tf.Tensor) and x.shape.rank == 0:
                return x[tf.newaxis], True
            return x, False

        if tf and isinstance(inputs, tf.data.Dataset):
            batches = list(inputs.as_numpy_iterator())
            input_is_scalar = False
        elif self.preprocessor is None:
            batches = [inputs]
            input_is_scalar = False
        elif isinstance(inputs, dict):
            inputs["prompts"], input_is_scalar = normalize(inputs["prompts"])
            batches = [inputs]
        else:
            inputs, input_is_scalar = normalize(inputs)
            batches = [inputs]

        if self.preprocessor is not None:
            batches = [
                self.preprocessor.generate_preprocess(
                    x, sequence_length=max_length
                )
                for x in batches
            ]

        outputs = [generate_function(x) for x in batches]

        if self.preprocessor is not None:
            outputs = [
                self.preprocessor.generate_postprocess(x) for x in outputs
            ]

        def _normalize_outputs(outs):
            if isinstance(outs[0], list):
                # generate_postprocess returns a Python list of strings
                # (convert_preprocessing_outputs converts tf.string tensors
                # to lists via tensor_to_list).
                result = [e for batch in outs for e in batch]
                return result[0] if input_is_scalar else result
            result = ops.concatenate(outs, axis=0)
            if input_is_scalar:
                result = ops.squeeze(result, axis=0)
            return ops.convert_to_numpy(result)

        return _normalize_outputs(outputs)

    def _encode_prompt(self, inputs):
        """Encode the prompt and return (encoder_cache, prompt_length).

        Subclasses must implement this method.

        Args:
            inputs: dict of pre-processed inputs.

        Returns:
            A tuple `(encoder_cache, prompt_length)` where `encoder_cache` is
            a backend tensor or structure holding the frozen KV cache for all
            transformer layers, and `prompt_length` is an int scalar giving the
            number of real (non-padded) prompt tokens.
        """
        raise NotImplementedError(
            f"`{self.__class__.__name__}` must implement `_encode_prompt()`."
        )

    def _prepare_canvas_embeds(self, canvas, prev_logits):
        """Embed the current canvas tokens, applying self-conditioning.

        Subclasses must implement this method.

        Args:
            canvas: int tensor of shape `(B, canvas_length)`.
            prev_logits: float tensor of shape `(B, canvas_length, vocab_size)`
                from the previous denoising step, or `None` on the first step.

        Returns:
            Float tensor of shape `(B, canvas_length, hidden_dim)`.
        """
        raise NotImplementedError(
            f"`{self.__class__.__name__}` must implement "
            "`_prepare_canvas_embeds()`."
        )

    def _decode_canvas_step(self, canvas_embeds, encoder_cache, prompt_length):
        """Run one decoder forward pass over the canvas.

        Subclasses must implement this method.

        Args:
            canvas_embeds:
            float tensor of shape `(B, canvas_length, hidden_dim)`.
            encoder_cache: encoder KV cache from `_encode_prompt`.
            prompt_length: int scalar, number of real prompt tokens.

        Returns:
            Float tensor of hidden states with shape
            `(B, canvas_length, hidden_dim)`.
        """
        raise NotImplementedError(
            f"`{self.__class__.__name__}` must implement "
            "`_decode_canvas_step()`."
        )

    def _canvas_logits(self, hidden):
        """Project decoder hidden states to vocabulary logits.

        Subclasses must implement this method.

        Args:
            hidden: float tensor of shape `(B, canvas_length, hidden_dim)`.

        Returns:
            Float tensor of shape `(B, canvas_length, vocab_size)`.
        """
        raise NotImplementedError(
            f"`{self.__class__.__name__}` must implement `_canvas_logits()`."
        )

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

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)
