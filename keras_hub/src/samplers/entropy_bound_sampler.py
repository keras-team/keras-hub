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
        seed: int or `None`. Random seed for the re-noising step.
            Defaults to `None`.

    Call arguments:
        next: Callable accepting `(canvas, prev_logits, step)` and returning
            logits for the current denoising step.
        canvas: int tensor of shape `(B, canvas_length)` containing the initial
            random token assignments.
        max_steps: int. Maximum number of denoising steps.
        model: Optional Keras model used by JAX stateless scopes.

    Returns:
        An int tensor of shape `(B, canvas_length)` containing the final greedy
        token assignment.

    Examples:
    ```python
    diffusion_lm = keras_hub.models.DiffusionGemmaBlockDiffusionLM.from_preset(
        "diffusion_gemma_26b_a4b_it"
    )

    # Pass by object.
    sampler = keras_hub.samplers.EntropyBoundSampler(
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
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.entropy_bound = entropy_bound
        self.confidence_threshold = confidence_threshold
        self.stability_threshold = stability_threshold
        self.seed = seed
        self.seed_generator = random.SeedGenerator(seed)

    def initialize_state(self, canvas):
        """Create tensor state for adaptive stopping."""
        previous_argmax = ops.zeros_like(canvas, dtype="int32")
        stable_steps = ops.zeros_like(canvas[..., 0], dtype="int32")
        has_previous = ops.convert_to_tensor(False, dtype="bool")
        return previous_argmax, stable_steps, has_previous

    def _sample_step(self, canvas, logits, step, state):
        vocabulary_size = logits.shape[-1]
        if vocabulary_size is None:
            raise ValueError(
                "The logits vocabulary dimension must be statically known."
            )
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

        # Commit: accepted positions get a multinomial sample from the logits,
        # rejected positions get a uniform random token.
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
            maxval=vocabulary_size,
            seed=self.seed_generator,
            dtype=canvas.dtype,
        )
        new_canvas = ops.where(accept_mask, accepted_canvas, random_canvas)

        # --- Adaptive stopping (per-row) ---
        cur_argmax = ops.cast(ops.argmax(logits, axis=-1), "int32")
        # Per-row mean entropy and confidence, shape (B,)
        mean_H = ops.mean(H, axis=-1)
        confidence_met = mean_H < self.confidence_threshold

        previous_argmax, stable_steps, has_previous = state
        has_previous = ops.logical_and(has_previous, ops.not_equal(step, 0))
        row_unchanged = ops.all(ops.equal(cur_argmax, previous_argmax), axis=-1)
        stable_steps = ops.where(
            ops.logical_and(has_previous, row_unchanged),
            stable_steps + 1,
            ops.zeros_like(stable_steps),
        )
        stability_met = ops.logical_and(
            has_previous, stable_steps >= self.stability_threshold
        )

        # Per-row stop: shape (B,) bool
        stop = ops.logical_and(confidence_met, stability_met)

        state = (
            cur_argmax,
            stable_steps,
            ops.convert_to_tensor(True, dtype="bool"),
        )
        return new_canvas, stop, cur_argmax, state

    def __call__(self, next, canvas, max_steps, model=None):
        state = self.initialize_state(canvas)
        logits = next(canvas, None, ops.convert_to_tensor(0, dtype="int32"))
        canvas, stop, argmax_canvas, state = self._sample_step(
            canvas, logits, 0, state
        )

        def cond(
            step,
            canvas,
            prev_logits,
            stop,
            argmax_canvas,
            previous_argmax,
            stable_steps,
            has_previous,
        ):
            return ops.logical_and(
                step < max_steps,
                ops.logical_not(ops.all(stop)),
            )

        def body(
            step,
            canvas,
            prev_logits,
            stop,
            argmax_canvas,
            previous_argmax,
            stable_steps,
            has_previous,
        ):
            finished_denoising = stop
            logits = next(canvas, prev_logits, step)
            state = previous_argmax, stable_steps, has_previous
            next_canvas, next_stop, next_argmax_canvas, next_state = (
                self._sample_step(canvas, logits, step, state)
            )
            finished_rows = ops.expand_dims(finished_denoising, axis=-1)
            canvas = ops.where(finished_rows, canvas, next_canvas)
            argmax_canvas = ops.where(
                finished_rows, argmax_canvas, next_argmax_canvas
            )
            logits = ops.where(
                ops.expand_dims(finished_rows, axis=-1),
                prev_logits,
                logits,
            )

            next_previous_argmax, next_stable_steps, next_has_previous = (
                next_state
            )
            previous_argmax = ops.where(
                finished_rows, previous_argmax, next_previous_argmax
            )
            stable_steps = ops.where(
                finished_denoising, stable_steps, next_stable_steps
            )
            has_previous = ops.logical_or(has_previous, next_has_previous)
            stop = ops.logical_or(finished_denoising, next_stop)
            return (
                step + 1,
                canvas,
                logits,
                stop,
                argmax_canvas,
                previous_argmax,
                stable_steps,
                has_previous,
            )

        loop_vars = (
            ops.convert_to_tensor(1, dtype="int32"),
            canvas,
            logits,
            stop,
            argmax_canvas,
            *state,
        )
        _, _, _, _, argmax_canvas, _, _, _ = self.run_loop(
            cond=cond,
            body=body,
            loop_vars=loop_vars,
            maximum_iterations=max_steps - 1,
            model=model,
        )
        return argmax_canvas

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "entropy_bound": self.entropy_bound,
                "confidence_threshold": self.confidence_threshold,
                "stability_threshold": self.stability_threshold,
                "seed": self.seed,
            }
        )
        return config
