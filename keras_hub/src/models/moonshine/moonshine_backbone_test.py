import numpy as np
import pytest
from keras import backend
from keras.src import ops

from keras_hub.src.models.moonshine.moonshine_backbone import MoonshineBackbone
from keras_hub.src.models.moonshine.moonshine_custom_attention import (
    MHACausalWithRope,
)
from keras_hub.src.models.moonshine.moonshine_custom_attention import (
    MHAPrecomputedKV,
)
from keras_hub.src.models.moonshine.moonshine_custom_feedforward import (
    FFLinearGelu,
)
from keras_hub.src.models.moonshine.moonshine_preprocessor import (
    AudioPreprocessor,
)
from keras_hub.src.models.moonshine.moonshine_utils import RotaryEmbedding
from keras_hub.src.tests.test_case import TestCase


def compute_expected_time_length(time_steps, kernel_sizes, strides):
    t = time_steps
    for k, s in zip(kernel_sizes, strides):
        t = (t - k) // s + 1
    return t


class MoonshineBackboneTest(TestCase):
    def setUp(self):
        super(MoonshineBackboneTest, self).setUp()
        # Model parameters.
        self.dim = 64
        self.inner_dim = 512  # Modified: Use a higher inner_dim so that
        # inner_dim//n_head=64, and then rotary embedding dim
        # = max(64//2, 32) = 32.
        self.n_head = 8
        self.enc_n_layers = 2
        self.enc_ff_mult = 4
        self.enc_ff_swiglu = False
        self.batch_size = 2
        self.seq_length = 16
        self.n_heads = self.n_head  # Ensure consistency for attention tests.
        # For testing, simulate 1 second of audio at 16kHz.
        self.time_steps = 16000
        # Create a dummy audio input of shape (batch_size, time_steps, channels)
        self.audio_input = np.random.rand(
            self.batch_size, self.time_steps, 1
        ).astype(np.float32)

        # Compute expected time dimension after preprocessor:
        # Conv1D layers: kernel_sizes=[127, 7, 3], strides=[64, 3, 2]
        self.expected_time = compute_expected_time_length(
            self.time_steps, kernel_sizes=[127, 7, 3], strides=[64, 3, 2]
        )

    def test_forward_pass(self):
        # Instantiate the backbone using only the required parameters.
        backbone = MoonshineBackbone(
            dim=self.dim,
            inner_dim=self.inner_dim,
            n_head=self.n_head,
            enc_n_layers=self.enc_n_layers,
            enc_ff_mult=self.enc_ff_mult,
            enc_ff_swiglu=self.enc_ff_swiglu,
        )
        # Call the backbone with the audio input.
        outputs = backbone(self.audio_input)
        # Our call method returns a tensor (or, for compatibility with test.
        # expectations, a dict)
        # Here we assume that outputs is the encoder feature tensor.
        expected_encoder_shape = (self.batch_size, self.expected_time, self.dim)

        self.assertEqual(outputs.shape, expected_encoder_shape)

    def test_serialization(self):
        backbone = MoonshineBackbone(
            dim=self.dim,
            inner_dim=self.inner_dim,
            n_head=self.n_head,
            enc_n_layers=self.enc_n_layers,
            enc_ff_mult=self.enc_ff_mult,
            enc_ff_swiglu=self.enc_ff_swiglu,
        )
        config = backbone.get_config()
        new_backbone = MoonshineBackbone.from_config(config)
        outputs = new_backbone(self.audio_input)
        # Check that the output tensor has the expected shape.
        expected_encoder_shape = (self.batch_size, self.expected_time, self.dim)
        self.assertEqual(outputs.shape, expected_encoder_shape)

    def test_function(self):
        backbone = MoonshineBackbone(
            dim=self.dim,
            inner_dim=self.inner_dim,
            n_head=self.n_head,
            enc_n_layers=self.enc_n_layers,
            enc_ff_mult=self.enc_ff_mult,
            enc_ff_swiglu=self.enc_ff_swiglu,
        )

        def run_model(audio):
            return backbone(audio)

        outputs = run_model(self.audio_input)

        expected_encoder_shape = (self.batch_size, self.expected_time, self.dim)
        self.assertEqual(outputs.shape, expected_encoder_shape)

    def test_audio_preprocessor(self):
        preprocessor = AudioPreprocessor(dim=self.dim)

        # Create sample audio input.
        audio_length = 16000  # 1 second at 16kHz.
        audio_input = ops.random.uniform((self.batch_size, audio_length, 1))

        output = preprocessor(audio_input)
        self.assertEqual(output.shape[-1], self.dim)  # Check output dimension.

    def test_rotary_embedding(self):
        rot_emb = RotaryEmbedding(dim=self.dim)
        position_ids = ops.arange(self.seq_length, dtype="float32")
        output = rot_emb(position_ids)
        self.assertEqual(output.shape, (self.seq_length, self.dim))

    def test_feedforward_network_linear_gelu(self):
        ff_network = FFLinearGelu(dim=self.dim, ff_mult=self.enc_ff_mult)

        inputs = ops.random.uniform(
            (self.batch_size, self.seq_length, self.dim)
        )
        output = ff_network(inputs)

        self.assertEqual(output.shape, inputs.shape)

    def test_causal_attention_mask(self):
        attention = MHACausalWithRope(
            num_heads=self.n_heads, key_dim=self.inner_dim // self.n_heads
        )

        query = ops.random.uniform((self.batch_size, self.seq_length, self.dim))
        mask = attention._compute_causal_mask(query, query)

        # Verify mask is lower triangular
        self.assertEqual(mask.shape, (1, self.seq_length, self.seq_length))
        mask_np = ops.convert_to_numpy(mask[0])
        self.assertTrue(np.allclose(mask_np, np.tril(np.ones_like(mask_np))))

    def test_precomputed_kv_attention(self):
        attention = MHAPrecomputedKV(
            num_heads=self.n_heads, key_dim=self.inner_dim // self.n_heads
        )

        query = ops.random.uniform((self.batch_size, self.seq_length, self.dim))
        key = ops.random.uniform((self.batch_size, self.seq_length, self.dim))
        value = ops.random.uniform((self.batch_size, self.seq_length, self.dim))

        # Test without cache
        output, cache_k, cache_v = attention(query=query, key=key, value=value)
        self.assertEqual(
            output.shape, (self.batch_size, self.seq_length, self.dim)
        )

        # Test with cache
        output_cached = attention(
            query=query,
            key=key,
            value=value,
            key_cache=cache_k,
            value_cache=cache_v,
        )
        self.assertEqual(
            output_cached.shape, (self.batch_size, self.seq_length, self.dim)
        )

    def test_swiglu_feedforward(self):
        backbone = MoonshineBackbone(
            dim=self.dim,
            inner_dim=self.inner_dim,
            n_head=self.n_head,
            enc_n_layers=self.enc_n_layers,
            enc_ff_mult=self.enc_ff_mult,
            enc_ff_swiglu=True,
        )
        outputs = backbone(self.audio_input)
        expected_time = compute_expected_time_length(
            self.time_steps, kernel_sizes=[127, 7, 3], strides=[64, 3, 2]
        )
        expected_shape = (self.batch_size, expected_time, self.dim)
        self.assertEqual(outputs.shape, expected_shape)

    def test_different_input_lengths(self):
        backbone = MoonshineBackbone(
            dim=self.dim,
            inner_dim=self.inner_dim,
            n_head=self.n_head,
            enc_n_layers=self.enc_n_layers,
            enc_ff_mult=self.enc_ff_mult,
        )
        short_input = np.random.rand(self.batch_size, 8000, 1).astype(
            np.float32
        )
        short_output = backbone(short_input)
        expected_time_short = compute_expected_time_length(
            8000, kernel_sizes=[127, 7, 3], strides=[64, 3, 2]
        )
        self.assertEqual(
            short_output.shape, (self.batch_size, expected_time_short, self.dim)
        )
        long_input = np.random.rand(self.batch_size, 32000, 1).astype(
            np.float32
        )
        long_output = backbone(long_input)
        expected_time_long = compute_expected_time_length(
            32000, kernel_sizes=[127, 7, 3], strides=[64, 3, 2]
        )
        self.assertEqual(
            long_output.shape, (self.batch_size, expected_time_long, self.dim)
        )

    def test_rotary_embedding_integration(self):
        backbone = MoonshineBackbone(
            dim=self.dim,
            inner_dim=self.inner_dim,
            n_head=self.n_head,
            enc_n_layers=self.enc_n_layers,
            enc_ff_mult=self.enc_ff_mult,
        )
        rot_emb = backbone.encoder.rot_pos_emb
        expected_rot_dim = max(self.inner_dim // self.n_head // 2, 32)
        self.assertEqual(rot_emb.dim, expected_rot_dim)
        seq_len = 10
        position_ids = ops.arange(seq_len)
        rot_pos_emb = rot_emb(position_ids)
        self.assertEqual(rot_pos_emb.shape, (seq_len, expected_rot_dim))

    def test_preprocessor_output_validity(self):
        backbone = MoonshineBackbone(
            dim=self.dim,
            inner_dim=self.inner_dim,
            n_head=self.n_head,
            enc_n_layers=self.enc_n_layers,
            enc_ff_mult=self.enc_ff_mult,
        )
        preprocessed = backbone.preprocessor(self.audio_input)
        self.assertEqual(preprocessed.shape[-1], self.dim)
        self.assertTrue(np.all(np.isfinite(ops.convert_to_numpy(preprocessed))))

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="tf.GradientTape() requires Tensorflow",
    )
    def test_gradient_flow(self):
        import tensorflow as tf

        backbone = MoonshineBackbone(
            dim=self.dim,
            inner_dim=self.inner_dim,
            n_head=self.n_head,
            enc_n_layers=self.enc_n_layers,
            enc_ff_mult=self.enc_ff_mult,
        )
        with tf.GradientTape() as tape:
            outputs = backbone(self.audio_input)
            loss = tf.reduce_mean(outputs)
        grads = tape.gradient(loss, backbone.trainable_variables)
        for grad in grads:
            self.assertIsNotNone(grad)
            self.assertTrue(np.all(np.isfinite(ops.convert_to_numpy(grad))))

    def test_encoder_internals(self):
        backbone = MoonshineBackbone(
            dim=self.dim,
            inner_dim=self.inner_dim,
            n_head=self.n_head,
            enc_n_layers=self.enc_n_layers,
            enc_ff_mult=self.enc_ff_mult,
        )
        preprocessed = backbone.preprocessor(self.audio_input)
        seq_len = ops.convert_to_tensor(
            [ops.shape(preprocessed)[1]], dtype="int32"
        )
        encoder_output = backbone.encoder(preprocessed, seq_len)
        self.assertEqual(encoder_output.shape, preprocessed.shape)
