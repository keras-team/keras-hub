from keras import ops

from keras_hub.src.models.vae.vae_layers import (
    DiagonalGaussianDistributionSampler,
)
from keras_hub.src.tests.test_case import TestCase


class DiagonalGaussianDistributionSamplerTest(TestCase):
    def test_layer_basics(self):
        self.run_layer_test(
            cls=DiagonalGaussianDistributionSampler,
            init_kwargs={"method": "mode"},
            input_data=ops.ones((2, 4, 4, 8)),
            expected_output_shape=(2, 4, 4, 4),
            expected_num_non_trainable_variables=1,
        )

    def test_method_is_serialized(self):
        layer = DiagonalGaussianDistributionSampler(method="sample")
        revived = DiagonalGaussianDistributionSampler.from_config(
            layer.get_config()
        )
        self.assertEqual(revived.method, "sample")
