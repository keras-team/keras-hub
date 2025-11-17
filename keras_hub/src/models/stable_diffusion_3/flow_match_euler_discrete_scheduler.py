from keras import layers
from keras import ops


class FlowMatchEulerDiscreteScheduler(layers.Layer):
    """Flow-matching sampling euler scheduler.

    This layer is used to compute the discrete sigmas for the diffusion chain.
    Typically, the sigma refers to the amount of noise added during the
    diffusion process.

    Args:
        num_train_timesteps: int. The number of diffusion steps to train the
            model.
        shift: float. The shift value for the timestep schedule.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.

    Call arguments:
        inputs: The current step of the diffusion process.
        num_steps: The total number of steps in the diffusion process.

    References:
    - [Common Diffusion Noise Schedules and Sample Steps are Flawed](
    https://arxiv.org/abs/2305.08891).
    - [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](
    https://arxiv.org/abs/2403.03206).
    """

    def __init__(self, num_train_timesteps=1000, shift=3.0, **kwargs):
        super().__init__(**kwargs)
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)

        timesteps = ops.linspace(
            1, num_train_timesteps, num_train_timesteps, dtype="float32"
        )
        timesteps = ops.flip(timesteps, axis=0)
        sigmas = self._timestep_to_sigma(timesteps)

        self.sigma_min = sigmas[-1]
        self.sigma_max = sigmas[0]

    def _sigma_to_timestep(self, sigma):
        return sigma * self.num_train_timesteps

    def _timestep_to_sigma(self, timestep):
        sigma = ops.divide(timestep, self.num_train_timesteps)
        if self.shift != 1.0:
            sigma = ops.divide(
                ops.multiply(self.shift, sigma),
                ops.add(1, ops.multiply(self.shift - 1.0, sigma)),
            )
        return sigma

    def set_sigmas(self, num_steps):
        timesteps = ops.linspace(
            self._sigma_to_timestep(self.sigma_max),
            self._sigma_to_timestep(self.sigma_min),
            num_steps,
        )
        sigmas = self._timestep_to_sigma(timesteps)
        sigmas = ops.concatenate([sigmas, ops.zeros((1,), dtype=sigmas.dtype)])
        self.sigmas = sigmas

    def call(self, inputs, num_steps):
        if not hasattr(self, "sigmas"):
            self.set_sigmas(num_steps)

        step = ops.expand_dims(
            ops.convert_to_tensor(inputs, dtype="int32"), axis=0
        )
        sigma = ops.take(self.sigmas, step)
        timestep = self._sigma_to_timestep(sigma)
        return sigma, timestep

    def add_noise(self, inputs, noises, step, num_steps):
        sigma, _ = self(step, num_steps)
        return ops.add(
            ops.multiply(sigma, noises),
            ops.multiply(ops.subtract(1.0, sigma), inputs),
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_train_timesteps": self.num_train_timesteps,
                "shift": self.shift,
            }
        )
        return config

    def compute_output_shape(self):
        # Returns a tuple of (sigma, timestep).
        return (None,), (None,)
