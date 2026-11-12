import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.flux2.flux2_vae import Flux2LatentNormalization
from keras_hub.src.models.flux2.flux2_vae import Flux2VAE
from keras_hub.src.models.flux2.flux2_vae import patchify_latents
from keras_hub.src.models.flux2.flux2_vae import unpatchify_latents
from keras_hub.src.tests.test_case import TestCase


def reference_patchify(latents):
    """`Flux2Pipeline._patchify_latents`, in NumPy, on NCHW latents."""
    batch, channels, height, width = latents.shape
    x = latents.reshape(batch, channels, height // 2, 2, width // 2, 2)
    x = x.transpose(0, 1, 3, 5, 2, 4)
    return x.reshape(batch, channels * 4, height // 2, width // 2)


class Flux2VAETest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "encoder_num_filters": [32, 32],
            "encoder_num_blocks": [1, 1],
            "decoder_num_filters": [32, 32],
            "decoder_num_blocks": [1, 1],
            "latent_channels": 4,
        }
        self.input_data = ops.ones((2, 32, 32, 3))

    def test_patchify_matches_reference_layout(self):
        # The channel order matters: the latent BatchNorm statistics are
        # stored per patchified channel.
        rng = np.random.default_rng(0)
        latents = rng.standard_normal((2, 4, 6, 8)).astype("float32")
        expected = reference_patchify(latents).transpose(0, 2, 3, 1)
        channels_last = latents.transpose(0, 2, 3, 1)
        patches = patchify_latents(channels_last)
        self.assertAllClose(patches, expected)
        self.assertAllClose(unpatchify_latents(patches), channels_last)

    def test_latent_normalization(self):
        rng = np.random.default_rng(0)
        layer = Flux2LatentNormalization(latent_channels=4)
        mean = rng.standard_normal(16).astype("float32")
        variance = rng.uniform(0.5, 2.0, 16).astype("float32")
        layer.moving_mean.assign(mean)
        layer.moving_variance.assign(variance)
        latents = rng.standard_normal((2, 6, 8, 4)).astype("float32")

        normalized = layer(latents)
        expected = (
            ops.convert_to_numpy(patchify_latents(latents)) - mean
        ) / np.sqrt(variance + 1e-4)
        self.assertAllClose(normalized, expected, atol=1e-5, rtol=1e-5)
        restored = layer(normalized, inverse=True)
        self.assertAllClose(restored, latents, atol=1e-5, rtol=1e-5)

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=Flux2VAE,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 32, 32, 3),
        )

    def test_encode_decode_shapes(self):
        vae = Flux2VAE(**self.init_kwargs)
        # 2x VAE downsampling, then 2x2 patches of 4 latent channels.
        latents = vae.encode(self.input_data)
        self.assertEqual(tuple(latents.shape), (2, 8, 8, 16))
        images = vae.decode(latents)
        self.assertEqual(tuple(images.shape), (2, 32, 32, 3))

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Flux2VAE,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
