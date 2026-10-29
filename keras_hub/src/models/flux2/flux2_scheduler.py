import keras
from keras import ops


def compute_empirical_mu(image_seq_len, num_steps):
    """The FLUX.2 timestep shift `mu` for a resolution and step count.

    FLUX.2 replaces FLUX.1's linear `base_shift`/`max_shift` interpolation
    with an empirical fit in both the image sequence length and the number of
    sampling steps. The constants are from the reference implementation
    (`compute_empirical_mu` in `black-forest-labs/flux2` and in diffusers'
    `Flux2Pipeline` / `Flux2KleinPipeline`).

    Args:
        image_seq_len: int. The number of packed latent tokens, i.e.
            `(height // 16) * (width // 16)`.
        num_steps: int or scalar tensor. The number of sampling steps.

    Returns:
        A float, or a float32 scalar tensor if `num_steps` is a tensor.
    """
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    image_seq_len = float(image_seq_len)
    if image_seq_len > 4300:
        return a2 * image_seq_len + b2

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    if isinstance(num_steps, (int, float)):
        return a * num_steps + b
    return a * ops.cast(num_steps, "float32") + b


class Flux2FlowMatchScheduler(keras.layers.Layer):
    """Flow-matching Euler schedule used by FLUX.2.

    Step `i` of `n` uses `t_i = 1 - i / n`, shifted with the exponential time
    shift `exp(mu) / (exp(mu) + (1 / t - 1))`, where `mu` comes from
    `compute_empirical_mu`. Step `n` is the terminal sigma, `0`. This matches
    diffusers' `FlowMatchEulerDiscreteScheduler` (with
    `use_dynamic_shifting=True, time_shift_type="exponential"`) as driven by
    `Flux2KleinPipeline`, and `get_schedule` in the reference
    implementation.

    Sigmas are computed on the fly from `(step, num_steps)` rather than
    precomputed, so `num_steps` can stay a traced tensor inside a compiled
    generate function without baking a stale schedule into the trace.

    Args:
        image_seq_len: int. The number of packed latent tokens the schedule
            is shifted for.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.

    Call arguments:
        step: int or int scalar tensor. The current step, in `[0, n]`.
        num_steps: int or int scalar tensor. The total number of steps `n`.

    Returns:
        A float32 tensor of shape `(1,)`: the sigma (the noise level, which is
        also the flow-matching time passed to the transformer) of `step`.
    """

    def __init__(self, image_seq_len, **kwargs):
        super().__init__(**kwargs)
        self.image_seq_len = int(image_seq_len)

    def call(self, step, num_steps):
        step = ops.cast(step, "float32")
        n = ops.cast(num_steps, "float32")
        mu = compute_empirical_mu(self.image_seq_len, n)
        # `np.linspace(1, 1 / n, n)[i] == 1 - i / n`. Clamped so the terminal
        # step does not divide by zero; it is replaced by `0` below.
        t = ops.maximum(1.0 - step / n, 1e-12)
        exp_mu = ops.exp(mu)
        sigma = exp_mu / (exp_mu + (1.0 / t - 1.0))
        sigma = ops.where(step < n, sigma, ops.zeros_like(sigma))
        return ops.reshape(sigma, (1,))

    def compute_output_shape(self, step_shape, num_steps_shape):
        return (1,)

    def get_config(self):
        config = super().get_config()
        config.update({"image_seq_len": self.image_seq_len})
        return config
