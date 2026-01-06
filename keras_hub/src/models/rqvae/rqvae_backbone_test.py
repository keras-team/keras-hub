import pytest
from keras import ops

from keras_hub.src.models.rqvae.rqvae_backbone import RQVAEBackbone
from keras_hub.src.tests.test_case import TestCase


class RQVAEBackboneTest(TestCase):
    def setUp(self):
        self.input_dim = 16
        self.encoder_layer_dims = [32, 16]
        self.output_dim = 8
        self.decoder_layer_dims = [16, 32]
        self.num_embeddings = 4
        self.num_quantizers = 2
        self.batch_size = 2

        self.init_kwargs = {
            "input_dim": self.input_dim,
            "encoder_layer_dims": self.encoder_layer_dims,
            "output_dim": self.output_dim,
            "decoder_layer_dims": self.decoder_layer_dims,
            "num_embeddings": self.num_embeddings,
            "num_quantizers": self.num_quantizers,
            "decay": 0.99,
            "data_variance": 1.0,
            "commitment_cost": 0.25,
        }
        self.input_data = ops.ones((self.batch_size, self.input_dim))

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=RQVAEBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "reconstructions": (self.batch_size, self.input_dim),
                # Encodings stacked on axis 0: (num_quantizers, batch_size,
                # num_embeddings)
                "encodings": (
                    self.num_quantizers,
                    self.batch_size,
                    self.num_embeddings,
                ),
                # Usage ratios stacked on axis 0: (num_quantizers,)
                "usage_ratios": (self.num_quantizers,),
                "quantization_loss": (1,),  # scalar (reshaped for predict)
            },
            run_mixed_precision_check=False,
            run_quantization_check=False,  # Skip for now
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=RQVAEBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
