from keras import ops

from keras_hub.src.models.flux2.flux2_scheduler import Flux2FlowMatchScheduler
from keras_hub.src.models.flux2.flux2_scheduler import compute_empirical_mu
from keras_hub.src.tests.test_case import TestCase


class Flux2FlowMatchSchedulerTest(TestCase):
    def test_compute_empirical_mu(self):
        # Reference values from diffusers' `compute_empirical_mu`.
        self.assertAllClose(compute_empirical_mu(4096, 4), 2.291179894115571)
        self.assertAllClose(compute_empirical_mu(64, 4), 1.9492865247731874)
        # Above 4300 tokens, `mu` does not depend on the number of steps.
        self.assertAllClose(compute_empirical_mu(6400, 2), 1.53999466)
        self.assertAllClose(compute_empirical_mu(6400, 50), 1.53999466)
        # A tensor step count gives the same value.
        self.assertAllClose(
            compute_empirical_mu(4096, ops.convert_to_tensor(4)),
            2.291179894115571,
        )

    def test_sigmas(self):
        # Reference values from diffusers' `FlowMatchEulerDiscreteScheduler`
        # with the FLUX.2 [klein] scheduler config, set up like
        # `Flux2KleinPipeline` (custom sigmas and `mu`). The last value is
        # the terminal sigma.
        cases = {
            (4096, 4): [1.0, 0.967384, 0.908144, 0.7672, 0.0],
            (64, 4): [1.0, 0.954692, 0.875369, 0.700709, 0.0],
            (6400, 2): [1.0, 0.823464, 0.0],
        }
        for (image_seq_len, num_steps), expected in cases.items():
            scheduler = Flux2FlowMatchScheduler(image_seq_len)
            num_steps_tensor = ops.convert_to_tensor(num_steps, "int32")
            sigmas = []
            for step in range(num_steps + 1):
                sigma = scheduler(
                    ops.convert_to_tensor(step, "int32"), num_steps_tensor
                )
                self.assertEqual(tuple(sigma.shape), (1,))
                sigmas.append(ops.convert_to_numpy(sigma)[0])
            self.assertAllClose(sigmas, expected, atol=1e-6, rtol=0)

    def test_serialization(self):
        self.run_serialization_test(Flux2FlowMatchScheduler(4096))
