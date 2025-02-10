import numpy as np
import tensorflow as tf
from moonshine_backbone import MoonshineBackbone

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
