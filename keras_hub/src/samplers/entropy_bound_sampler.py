import keras
from keras import ops
from keras import random

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.samplers.sampler import Sampler


@keras_hub_export("keras_hub.samplers.EntropyBoundSampler")
class EntropyBoundSampler(Sampler):
    """Entropy-bound sampler for discrete block-diffusion generation.

    This sampler implements an entropy-bound decoding algorithm for use with
    `DiffusionLM` models. Each step, token positions are committed greedily
    from lowest to highest entropy until the cumulative entropy exceeds
    `entropy_bound`; uncommitted positions are re-noised with random tokens.

    Args:
        entropy_bound: float. Maximum cumulative entropy budget.  Positions are
            committed greedily from lowest to highest entropy until the next
            position would push the cumulative sum above this bound.  Smaller
            values commit fewer tokens per step (more denoising steps needed);
            larger values commit more. Defaults to `0.1`.
        confidence_threshold: float. Mean per-token entropy below which the
            model is considered confident enough to stop.  Defaults to `0.005`.
        stability_threshold: int. Number of consecutive steps for which the
            argmax assignment must be unchanged before stopping is allowed.
            Defaults to `1`.
        vocabulary_size: int. Vocabulary size used for uniform re-noising.
            Required.
        seed: int or `None`. Random seed for the re-noising step.
            Defaults to `None`.

    Call arguments:
        canvas: int tensor of shape `(B, canvas_length)`. Current canvas
            token assignments (may contain noise from the previous step).
        logits: float tensor of shape `(B, canvas_length, vocab_size)`.
            Raw (temperature-scaled) logits from the decoder.
        step: int scalar. Current denoising step index (0-based).

    Returns:
        A tuple `(new_canvas, stop)` where `new_canvas` is an int tensor
        of shape `(B, canvas_length)` and `stop` is a bool tensor of shape
        `(B,)` indicating per-row whether the adaptive stopping criterion
        is met.

    Examples:
    ```python
    diffusion_lm = keras_hub.models.Gemma4BlockDiffusionLM.from_preset(
        "gemma4_diffusion_2b_en"
    )

    # Pass by object.
    sampler = keras_hub.samplers.EntropyBoundSampler(
        vocabulary_size=256000,
        entropy_bound=0.1,
    )
    diffusion_lm.compile(sampler=sampler)
    diffusion_lm.generate(["Keras is"])
    ```
    """

    def __init__(
        self,
        entropy_bound=0.1,
        confidence_threshold=0.005,
        stability_threshold=1,
        vocabulary_size=None,
        seed=None,
        **kwargs,
    ):
        if vocabulary_size is None:
            raise ValueError(
                "`vocabulary_size` is required for `EntropyBoundSampler`."
            )
        super().__init__(**kwargs)
        self.entropy_bound = entropy_bound
        self.confidence_threshold = confidence_threshold
        self.stability_threshold = stability_threshold
        self.vocabulary_size = vocabulary_size
        self.seed = seed
        self.seed_generator = random.SeedGenerator(seed)
        # Both are set lazily on first call to _ensure_prev_argmax.
        self._prev_argmax = None
        self._stable_steps = None  # list of B ints, one counter per row

    def _ensure_prev_argmax(self, shape, batch_size):
        """Lazily initialise prev_argmax and per-row stable counters."""
        if self._prev_argmax is None:
            self._prev_argmax = keras.Variable(
                initializer=ops.zeros(shape, dtype="int32"),
                shape=shape,
                dtype="int32",
                trainable=False,
                name="prev_argmax",
            )
            self._stable_steps = [0] * batch_size

    def __call__(self, canvas, logits, step):
        logits = ops.cast(logits, "float32")

        # Per-token entropy: H[i] = -sum(softmax(l) * log_softmax(l))
        log_probs = ops.log_softmax(logits, axis=-1)
        probs = ops.exp(log_probs)
        # H shape: (B, canvas_length)
        H = -ops.sum(probs * log_probs, axis=-1)

        sorted_H = ops.sort(H, axis=-1)
        sort_idx = ops.argsort(H, axis=-1)
        cumsum_H = ops.cumsum(sorted_H, axis=-1)

        # Accept position i iff cumsum_H[i] - sorted_H[i] <= entropy_bound.
        accept_sorted = (cumsum_H - sorted_H) <= self.entropy_bound

        unsort_idx = ops.argsort(sort_idx, axis=-1)
        accept_mask = ops.take_along_axis(accept_sorted, unsort_idx, axis=-1)

        # Commit: accepted positions get a multinomial sample from the logits.
        canvas_shape = ops.shape(canvas)
        flat_logits = ops.reshape(logits, [-1, ops.shape(logits)[-1]])
        sampled_tokens = keras.random.categorical(
            flat_logits, num_samples=1, seed=self.seed_generator
        )
        sampled_canvas = ops.cast(
            ops.reshape(sampled_tokens[..., 0], canvas_shape), canvas.dtype
        )
        accepted_canvas = ops.where(accept_mask, sampled_canvas, canvas)

        # Re-noise: uncommitted positions get uniformly random new tokens so
        # the model cannot carry forward uncertain predictions across steps.
        random_canvas = keras.random.randint(
            shape=ops.shape(canvas),
            minval=0,
            maxval=self.vocabulary_size,
            seed=self.seed_generator,
            dtype=canvas.dtype,
        )
        new_canvas = ops.where(accept_mask, accepted_canvas, random_canvas)

        # --- Adaptive stopping (per-row) ---
        cur_argmax = ops.cast(ops.argmax(logits, axis=-1), "int32")
        # Per-row mean entropy and confidence, shape (B,)
        mean_H = ops.mean(H, axis=-1)
        confidence_met = mean_H < self.confidence_threshold

        batch_size_int = int(ops.convert_to_numpy(ops.shape(canvas)[0]))
        self._ensure_prev_argmax(ops.shape(cur_argmax), batch_size_int)

        # Stability: per-row argmax must be unchanged for `stability_threshold`
        # consecutive steps.  On step 0 there is no prior state; reset counters.
        if step > 0:
            row_unchanged = ops.convert_to_numpy(
                ops.all(ops.equal(cur_argmax, self._prev_argmax), axis=-1)
            )  # numpy (B,) bool
            self._stable_steps = [
                (sc + 1 if bool(u) else 0)
                for sc, u in zip(self._stable_steps, row_unchanged)
            ]
            stability_met = ops.convert_to_tensor(
                [sc >= self.stability_threshold for sc in self._stable_steps],
                dtype="bool",
            )
        else:
            self._stable_steps = [0] * batch_size_int
            stability_met = ops.zeros((batch_size_int,), dtype="bool")

        # Per-row stop: shape (B,) bool
        stop = confidence_met & stability_met

        self._prev_argmax.assign(cur_argmax)
        return new_canvas, stop

    def reset(self):
        """Reset per-call state between independent generate() calls."""
        self._prev_argmax = None
        self._stable_steps = None

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "entropy_bound": self.entropy_bound,
                "confidence_threshold": self.confidence_threshold,
                "stability_threshold": self.stability_threshold,
                "vocabulary_size": self.vocabulary_size,
                "seed": self.seed,
            }
        )
        return config
