import numpy as np
import tensorflow as tf
from keras.src import ops
from moonshine_backbone import MoonshineBackbone
from moonshine_custom_attention import MHACausalWithRope
from moonshine_custom_attention import MHAPrecomputedKV
from moonshine_custom_feedforward import FFLinearGelu
from moonshine_preprocessor import AudioPreprocessor
from moonshine_utils import RotaryEmbedding

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
        self.inner_dim = 128
        self.n_head = 8
        self.enc_n_layers = 2
        self.enc_ff_mult = 4
        self.enc_ff_swiglu = False
        self.batch_size = 2
        self.seq_length = 16
        self.n_heads = 4
        self.inner_dim = self.dim
        self.ff_mult = 4

        self.batch_size = 2
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
            ff_mult=self.enc_ff_mult,
            enc_ff_swiglu=self.enc_ff_swiglu,
        )
        # Call the backbone with the audio input.
        outputs = backbone(self.audio_input)
        # Our call method returns a tensor (or, for compatibility with test
        # expectations, a dict)
        # Here we assume that outputs is the encoder feature tensor.
        expected_encoder_shape = (self.batch_size, self.expected_time, self.dim)

        print("Expected encoder_sequence_output shape:", expected_encoder_shape)
        print("Got encoder_sequence_output shape:", outputs.shape)

        self.assertEqual(outputs.shape, expected_encoder_shape)

    def test_serialization(self):
        backbone = MoonshineBackbone(
            dim=self.dim,
            inner_dim=self.inner_dim,
            n_head=self.n_head,
            enc_n_layers=self.enc_n_layers,
            ff_mult=self.enc_ff_mult,
            enc_ff_swiglu=self.enc_ff_swiglu,
        )
        config = backbone.get_config()
        new_backbone = MoonshineBackbone.from_config(config)
        outputs = new_backbone(self.audio_input)

        print("Serialization test passed. Model restored successfully.")
        # Check that the output tensor has the expected shape.
        expected_encoder_shape = (self.batch_size, self.expected_time, self.dim)
        self.assertEqual(outputs.shape, expected_encoder_shape)

    def test_tf_function(self):
        backbone = MoonshineBackbone(
            dim=self.dim,
            inner_dim=self.inner_dim,
            n_head=self.n_head,
            enc_n_layers=self.enc_n_layers,
            ff_mult=self.enc_ff_mult,
            enc_ff_swiglu=self.enc_ff_swiglu,
        )

        @tf.function
        def run_model(audio):
            return backbone(audio)

        outputs = run_model(self.audio_input)

        print("Running model with tf.function")
        print("Output shape:", outputs.shape)

        expected_encoder_shape = (self.batch_size, self.expected_time, self.dim)
        self.assertEqual(outputs.shape, expected_encoder_shape)

    def test_audio_preprocessor(self):
        preprocessor = AudioPreprocessor(dim=self.dim)

        # Create sample audio input
        audio_length = 16000  # 1 second at 16kHz
        audio_input = ops.random.uniform((self.batch_size, audio_length, 1))

        output = preprocessor(audio_input)
        self.assertEqual(output.shape[-1], self.dim)  # Check output dimension

    def test_rotary_embedding(self):
        rot_emb = RotaryEmbedding(dim=self.dim)
        position_ids = ops.arange(self.seq_length, dtype="float32")

        output = rot_emb(position_ids)
        self.assertEqual(output.shape, (self.seq_length, self.dim))

    def test_feedforward_network_linear_gelu(self):
        ff_network = FFLinearGelu(dim=self.dim, ff_mult=self.ff_mult)

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
