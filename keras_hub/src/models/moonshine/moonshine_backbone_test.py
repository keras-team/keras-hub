import keras
import pytest

from keras_hub.src.models.moonshine.moonshine_backbone import MoonshineBackbone
from keras_hub.src.models.moonshine.moonshine_layers import MoonshineArange
from keras_hub.src.models.moonshine.moonshine_layers import MoonshineLinearGeLU
from keras_hub.src.models.moonshine.moonshine_layers import (
    MoonshineRotaryEmbedding,
)
from keras_hub.src.models.moonshine.moonshine_layers import MoonshineSwiGLU


def compute_expected_time_length(time_steps, kernel_sizes, strides):
    t = time_steps
    for k, s in zip(kernel_sizes, strides):
        t = (t - k) // s + 1
    return t


# Skipped for now (not subclassed from TestCase).
class MoonshineBackboneTest:
    def setUp(self):
        self.init_kwargs = {
            "num_layers": 2,
            "hidden_dim": 64,
            "inner_dim": 512,
            "num_heads": 8,
            "ff_mult": 4,
        }
        self.input_data = {
            "encoder_sequence": keras.random.uniform((2, 16, 64)),
            "sequence_length": keras.ops.convert_to_tensor(
                [[16]], dtype="int32"
            ),
        }
        self.expected_output_shape = {"encoder_output": (2, 16, 64)}
        super(MoonshineBackboneTest, self).setUp()

    # ---------------
    # Component Tests
    # ---------------

    def test_forward_pass(self):
        backbone = MoonshineBackbone(**self.init_kwargs)
        outputs = backbone(self.input_data)
        self.assertEqual(outputs["encoder_output"].shape, (2, 16, 64))

    def test_serialization(self):
        backbone = MoonshineBackbone(**self.init_kwargs)
        config = backbone.get_config()
        new_backbone = MoonshineBackbone.from_config(config)
        outputs = new_backbone(self.input_data)
        self.assertEqual(outputs["encoder_output"].shape, (2, 16, 64))

    def test_rotary_embedding(self):
        rot_dim = max(
            self.init_kwargs["inner_dim"] // self.init_kwargs["num_heads"] // 2,
            32,
        )
        rot_emb = MoonshineRotaryEmbedding(dim=rot_dim)
        position_ids = keras.ops.arange(16, dtype="float32")
        output = rot_emb(position_ids)
        self.assertEqual(output.shape, (16, rot_dim))

    def test_swiglu_feedforward(self):
        backbone = MoonshineBackbone(ff_swiglu=True, **self.init_kwargs)
        outputs = backbone(self.input_data)
        self.assertEqual(outputs["encoder_output"].shape, (2, 16, 64))

    def test_linear_gelu_layer(self):
        ff_layer = MoonshineLinearGeLU(
            hidden_dim=self.init_kwargs["hidden_dim"],
            multiplier=self.init_kwargs["ff_mult"],
        )
        outputs = ff_layer(self.input_data["encoder_sequence"])
        self.assertEqual(
            outputs.shape, self.input_data["encoder_sequence"].shape
        )

    def test_swiglu_layer(self):
        ff_layer = MoonshineSwiGLU(
            hidden_dim=self.init_kwargs["hidden_dim"],
            multiplier=self.init_kwargs["ff_mult"],
        )
        outputs = ff_layer(self.input_data["encoder_sequence"])
        self.assertEqual(
            outputs.shape, self.input_data["encoder_sequence"].shape
        )

    def test_arange_layer(self):
        arange_layer = MoonshineArange()
        length = keras.ops.convert_to_tensor([10], dtype="int32")
        output = arange_layer(length)
        self.assertEqual(output.shape, (10,))

    def test_different_sequence_lengths(self):
        backbone = MoonshineBackbone(**self.init_kwargs)
        short_seq = keras.random.uniform((2, 8, 64))
        short_len = keras.ops.convert_to_tensor([[8]], dtype="int32")
        short_output = backbone(
            {"encoder_sequence": short_seq, "sequence_length": short_len}
        )
        self.assertEqual(short_output["encoder_output"].shape, (2, 8, 64))

        long_seq = keras.random.uniform((2, 32, 64))
        long_len = keras.ops.convert_to_tensor([[32]], dtype="int32")
        long_output = backbone(
            {"encoder_sequence": long_seq, "sequence_length": long_len}
        )
        self.assertEqual(long_output["encoder_output"].shape, (2, 32, 64))

    @pytest.mark.skipif(
        keras.backend.backend() != "tensorflow",
        reason="tf.GradientTape() requires Tensorflow",
    )
    def test_gradient_flow(self):
        import tensorflow as tf

        backbone = MoonshineBackbone(**self.init_kwargs)
        with tf.GradientTape() as tape:
            outputs = backbone(self.input_data)
            loss = tf.reduce_mean(outputs["encoder_output"])
        grads = tape.gradient(loss, backbone.trainable_variables)
        for grad in grads:
            self.assertIsNotNone(grad)
            self.assertTrue(keras.ops.all(keras.ops.isfinite(grad)))

    def test_predict_model(self):
        import numpy as np

        backbone = MoonshineBackbone(**self.init_kwargs)
        encoder_sequence = self.input_data["encoder_sequence"]
        batch_size = encoder_sequence.shape[0]
        sequence_length = np.full((batch_size,), 16, dtype="int32")
        inputs = {
            "encoder_sequence": encoder_sequence,
            "sequence_length": sequence_length,
        }
        outputs = backbone.predict(inputs)
        self.assertEqual(
            outputs["encoder_output"].shape,
            (batch_size, 16, self.init_kwargs["hidden_dim"]),
        )

    def test_varying_batch_sizes(self):
        backbone = MoonshineBackbone(**self.init_kwargs)
        # Test with several batch sizes.
        for batch_size in [1, 3, 5]:
            seq_length = 16
            encoder_seq = keras.random.uniform(
                (batch_size, seq_length, self.init_kwargs["hidden_dim"])
            )
            sequence_length = keras.ops.convert_to_tensor(
                [[seq_length]], dtype="int32"
            )
            outputs = backbone(
                {
                    "encoder_sequence": encoder_seq,
                    "sequence_length": sequence_length,
                }
            )
            self.assertEqual(
                outputs["encoder_output"].shape,
                (batch_size, seq_length, self.init_kwargs["hidden_dim"]),
            )

    # ------------------
    # Standardized tests
    # ------------------
    # TODO: Define presets and preset tests.

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MoonshineBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
