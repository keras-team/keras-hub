import keras
import numpy as np
from keras import ops

from keras_hub.src.models.qwen3_omni.qwen3_omni_audio_encoder import (
    Qwen3OmniAudioEncoder,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_audio_encoder import (
    Qwen3OmniAudioEncoderLayer,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3OmniAudioEncoderTest(TestCase):
    def test_encoder_layer_output_shape(self):
        layer = Qwen3OmniAudioEncoderLayer(
            embed_dim=32,
            num_heads=4,
            ffn_dim=64,
            dtype="float32",
        )
        hidden_states = np.random.rand(1, 10, 32).astype("float32")
        output = layer(hidden_states)
        self.assertEqual(output.shape, (1, 10, 32))

    def test_encoder_output_shape(self):
        encoder = Qwen3OmniAudioEncoder(
            num_mel_bins=80,
            d_model=32,
            encoder_layers=2,
            encoder_attention_heads=4,
            encoder_ffn_dim=64,
            output_dim=16,
            max_source_positions=100,
            scale_embedding=False,
            dtype="float32",
        )
        input_features = np.random.rand(1, 160, 80).astype("float32")
        output = encoder({"input_features": input_features})
        self.assertEqual(output.shape[-1], 16)

    def test_encoder_config_roundtrip(self):
        encoder = Qwen3OmniAudioEncoder(
            num_mel_bins=80,
            d_model=32,
            encoder_layers=2,
            encoder_attention_heads=4,
            encoder_ffn_dim=64,
            output_dim=16,
            max_source_positions=100,
            scale_embedding=False,
            dtype="float32",
        )
        config = encoder.get_config()
        restored = Qwen3OmniAudioEncoder.from_config(config)
        self.assertEqual(restored.d_model, 32)
        self.assertEqual(restored.output_dim, 16)
        self.assertEqual(restored.encoder_layers, 2)

    def test_window_aftercnn_uses_hf_defaults(self):
        # HF defaults: chunk_length=n_window*2=200 -> ceil(200/8)=25 after
        # a single-pass (non-chunked) Keras CNN of three stride-2
        # padding=same convs. ratio = n_window_infer // chunk_length =
        # 400 // 200 = 2. So window_aftercnn = 25 * 2 = 50.
        encoder = Qwen3OmniAudioEncoder(
            num_mel_bins=80,
            d_model=32,
            encoder_layers=1,
            encoder_attention_heads=4,
            encoder_ffn_dim=64,
            output_dim=16,
            max_source_positions=100,
            scale_embedding=False,
            dtype="float32",
        )
        self.assertEqual(encoder.n_window, 100)
        self.assertEqual(encoder.n_window_infer, 400)
        self.assertEqual(encoder.conv_chunksize, 500)
        self.assertEqual(encoder.window_aftercnn, 50)

    def test_window_mask_block_diagonal(self):
        # Small toy window so we can assert the block structure without
        # running a 200-frame mel through the full encoder.
        encoder = Qwen3OmniAudioEncoder(
            num_mel_bins=80,
            d_model=32,
            encoder_layers=1,
            encoder_attention_heads=4,
            encoder_ffn_dim=64,
            output_dim=16,
            max_source_positions=100,
            scale_embedding=False,
            n_window=2,
            n_window_infer=4,
            conv_chunksize=500,
            dtype="float32",
        )
        self.assertEqual(encoder.window_aftercnn, 1)
        mask = ops.convert_to_numpy(encoder._build_window_attention_mask(5))
        # Expect the identity matrix (each position only attends to self).
        expected = np.eye(5, dtype=bool)[None, ...]
        np.testing.assert_array_equal(mask, expected)

    def test_window_mask_multi_block(self):
        encoder = Qwen3OmniAudioEncoder(
            num_mel_bins=80,
            d_model=32,
            encoder_layers=1,
            encoder_attention_heads=4,
            encoder_ffn_dim=64,
            output_dim=16,
            max_source_positions=100,
            scale_embedding=False,
            n_window=2,
            n_window_infer=8,
            conv_chunksize=500,
            dtype="float32",
        )
        # chunk_length=4 -> after CNN = 1, ratio = 8//4 = 2,
        # window_aftercnn = 1 * 2 = 2.
        self.assertEqual(encoder.window_aftercnn, 2)
        mask = ops.convert_to_numpy(encoder._build_window_attention_mask(5))
        # Blocks: [0,1] [2,3] [4]. Build expected manually.
        expected = np.zeros((5, 5), dtype=bool)
        for i in range(5):
            for j in range(5):
                if i // 2 == j // 2:
                    expected[i, j] = True
        np.testing.assert_array_equal(mask[0], expected)

    def test_conv_stack_matches_torch_padding1(self):
        # HF convs use padding=1 (symmetric); Keras "same" pads (0,1)
        # on even axes, shifting receptive fields. Drive the encoder's
        # conv stack on an even-length input and compare against a
        # manual symmetric-pad reference.
        encoder = Qwen3OmniAudioEncoder(
            num_mel_bins=16,
            d_model=32,
            encoder_layers=1,
            encoder_attention_heads=4,
            encoder_ffn_dim=64,
            output_dim=16,
            downsample_hidden_size=8,
            max_source_positions=100,
            scale_embedding=False,
            n_window=4,
            dtype="float32",
        )
        rng = np.random.RandomState(0)
        x = rng.randn(1, 8, 16, 1).astype("float32")  # one 8-frame chunk

        def ref_conv(x, conv):
            # torch.nn.Conv2d(k=3, s=2, padding=1) semantics
            xp = np.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)])
            t_out = (x.shape[1] + 2 - 3) // 2 + 1
            f_out = (x.shape[2] + 2 - 3) // 2 + 1
            w = conv.kernel.numpy()
            b = conv.bias.numpy()
            acc = np.zeros((1, t_out, f_out, w.shape[-1]), dtype=np.float32)
            for i in range(3):
                for j in range(3):
                    acc += np.einsum(
                        "btfc,cg->btfg",
                        xp[:, i : i + 2 * t_out : 2, j : j + 2 * f_out : 2],
                        w[i, j],
                    )
            return acc + b.reshape(1, 1, 1, -1)

        # Convs must be "valid" — the symmetric pad is applied manually.
        for conv in (encoder.conv2d1, encoder.conv2d2, encoder.conv2d3):
            self.assertEqual(conv.padding, "valid")

        # Full stack vs reference chain (pad1-conv + gelu each stage).
        got = ops.convert_to_numpy(encoder._conv_downsample(x))
        ref_full = x
        for conv in (encoder.conv2d1, encoder.conv2d2, encoder.conv2d3):
            ref_full = ops.convert_to_numpy(
                ops.gelu(ops.convert_to_tensor(ref_conv(ref_full, conv)))
            )
        np.testing.assert_allclose(got, ref_full, atol=1e-5)

        # Guard the actual regression: keras "same" on even-length axes
        # pads asymmetrically and produces a different result.
        same_conv = keras.layers.Conv2D(
            encoder.conv2d1.filters, 3, strides=2, padding="same"
        )
        _ = same_conv(x)  # build weights
        same_conv.kernel.assign(encoder.conv2d1.kernel)
        same_conv.bias.assign(encoder.conv2d1.bias)
        same_out = ops.convert_to_numpy(same_conv(x))
        ref1 = ops.convert_to_numpy(
            ops.gelu(ops.convert_to_tensor(ref_conv(x, encoder.conv2d1)))
        )
        self.assertFalse(np.allclose(same_out, ref1, atol=1e-5))

    def test_short_audio_mask_is_full(self):
        encoder = Qwen3OmniAudioEncoder(
            num_mel_bins=80,
            d_model=32,
            encoder_layers=1,
            encoder_attention_heads=4,
            encoder_ffn_dim=64,
            output_dim=16,
            max_source_positions=100,
            scale_embedding=False,
            dtype="float32",
        )
        mask = ops.convert_to_numpy(encoder._build_window_attention_mask(20))
        self.assertTrue(bool(mask.all()))
