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
from keras import ops


class FlowMatchEulerDiscreteScheduler:
    """Flow-matching sampling euler scheduler.

    This scheduler is introduced in [
    Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](
    https://arxiv.org/abs/2403.03206).

    Args:
        num_train_timesteps: int. The number of diffusion steps to train the
            model.
        shift: float. The shift value for the timestep schedule.
    """

    def __init__(self, num_train_timesteps=1000, shift=1.0):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)

        timesteps = ops.linspace(
            1, num_train_timesteps, num_train_timesteps, dtype="float32"
        )
        timesteps = ops.flip(timesteps, axis=0)
        sigmas = self._timestep_to_sigma(timesteps)

        self.timesteps = ops.multiply(sigmas, num_train_timesteps)
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

    def get_sigma(self, step, num_steps):
        """Get the discrete sigmas used for the diffusion chain.

        Typically, the sigma means the amount of noise added during the
        diffusion process.

        Args:
            step: int. The current step of the diffusion process.
            num_steps: int. The total number of steps in the diffusion process.
        """
        start = self._sigma_to_timestep(self.sigma_max)
        end = self._sigma_to_timestep(self.sigma_min)
        step_size = ops.divide(
            ops.subtract(end, start), ops.subtract(num_steps, 1)
        )
        result_timestep = ops.add(start, ops.multiply(step, step_size))
        result_sigma = self._timestep_to_sigma(result_timestep)
        return ops.maximum(result_sigma, 0.0)

    def step(self, latents, noise_residual, sigma, sigma_next):
        """Predict the sample from the previous timestep by reversing the SDE.

        This function propagates the diffusion process from the learned model
        outputs (most often the predicted noise).

        Args:
            latents: A current instance of a sample created by the diffusion
                process.
            noise_residual: The direct output from the diffusion model.
            sigma: The amount of noise added at the current timestep.
            sigma_next: The amount of noise added at the next timestep.
        """
        return latents + (sigma_next - sigma) * noise_residual
