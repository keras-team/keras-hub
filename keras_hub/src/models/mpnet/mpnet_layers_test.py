import numpy as np
from keras import ops
from keras import random

from keras_hub.src.models.mpnet.mpnet_layers import MPNetAttention
from keras_hub.src.models.mpnet.mpnet_layers import MPNetRelativePositionBias
from keras_hub.src.models.mpnet.mpnet_layers import MPNetTransformerLayer
from keras_hub.src.tests.test_case import TestCase


class MPNetRelativePositionBiasTest(TestCase):
    def test_layer_basics(self):
        self.run_layer_test(
            cls=MPNetRelativePositionBias,
            init_kwargs={"num_heads": 2},
            input_data=random.uniform(shape=(2, 5, 4)),
            expected_output_shape=(1, 2, 5, 5),
            expected_num_trainable_weights=1,
            # The bias does not have a batch dimension.
            run_training_check=False,
        )

    def test_buckets_match_reference(self):
        # Expected buckets were computed with the reference (Hugging Face)
        # implementation of `relative_position_bucket`. The distances 16, 32
        # and 64 land exactly on a bucket boundary.
        expected = {
            0: 0,
            -1: 1,
            1: 17,
            -7: 7,
            -8: 8,
            -16: 10,
            -32: 12,
            -64: 14,
            -127: 15,
            -128: 15,
            128: 31,
            -500: 15,
        }
        layer = MPNetRelativePositionBias(num_heads=1)
        layer.build((None, None, 4))
        # With an identity table, the bias values are the bucket ids.
        layer.embeddings.assign(np.arange(32, dtype="float32")[:, None])
        bias = ops.convert_to_numpy(layer(ops.zeros((1, 520, 4))))
        for relative_position, bucket in expected.items():
            query = 0 if relative_position > 0 else 510
            key = query + relative_position
            self.assertEqual(
                bias[0, 0, query, key],
                bucket,
                msg=f"relative_position={relative_position}",
            )

    def test_bias_is_translation_invariant(self):
        layer = MPNetRelativePositionBias(num_heads=2)
        bias = ops.convert_to_numpy(layer(ops.zeros((1, 6, 4))))
        # The bias only depends on the key position minus the query position.
        self.assertAllClose(bias[:, :, 1:, 1:], bias[:, :, :-1, :-1])


class MPNetAttentionTest(TestCase):
    def test_layer_basics(self):
        self.run_layer_test(
            cls=MPNetAttention,
            init_kwargs={"num_heads": 2},
            input_data=random.uniform(shape=(2, 5, 4)),
            expected_output_shape=(2, 5, 4),
            expected_num_trainable_weights=8,
        )

    def test_position_bias_changes_output(self):
        layer = MPNetAttention(num_heads=2)
        x = random.uniform(shape=(1, 3, 4))
        bias = random.uniform(shape=(1, 2, 3, 3))
        self.assertNotAllClose(layer(x), layer(x, position_bias=bias))

    def test_masked_keys_are_ignored(self):
        layer = MPNetAttention(num_heads=2)
        x = np.random.uniform(size=(1, 3, 4)).astype("float32")
        mask = np.array([[[1, 1, 0]]], dtype="int32")
        output = layer(x, attention_mask=mask)
        # Changing a masked key must not change the outputs.
        x_changed = x.copy()
        x_changed[:, 2, :] += 10.0
        output_changed = layer(x_changed, attention_mask=mask)
        self.assertAllClose(output[:, :2, :], output_changed[:, :2, :])

    def test_invalid_num_heads(self):
        layer = MPNetAttention(num_heads=3)
        with self.assertRaises(ValueError):
            layer.build((None, None, 4))


class MPNetTransformerLayerTest(TestCase):
    def test_layer_basics(self):
        self.run_layer_test(
            cls=MPNetTransformerLayer,
            init_kwargs={
                "num_heads": 2,
                "intermediate_dim": 8,
                "dropout": 0.1,
            },
            input_data=random.uniform(shape=(2, 5, 4)),
            expected_output_shape=(2, 5, 4),
            expected_num_trainable_weights=16,
            expected_num_non_trainable_variables=3,  # dropout rng seeds
        )

    def test_padding_mask_and_position_bias(self):
        layer = MPNetTransformerLayer(num_heads=2, intermediate_dim=8)
        x = random.uniform(shape=(2, 5, 4))
        padding_mask = np.array([[1, 1, 1, 0, 0]] * 2, dtype="int32")
        bias = random.uniform(shape=(1, 2, 5, 5))
        output = layer(x, padding_mask=padding_mask, position_bias=bias)
        self.assertEqual(output.shape, (2, 5, 4))
