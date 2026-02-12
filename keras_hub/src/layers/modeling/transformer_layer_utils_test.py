from keras import ops
from keras import random

import keras_hub.src.layers.modeling.transformer_layer_utils as utils
from keras_hub.src.tests.test_case import TestCase


class TransformerLayerUtilsTest(TestCase):
    def test_compute_causal_mask(self):
        mask = utils.compute_causal_mask(1, 2, 2)
        self.assertAllEqual(mask, [[[1, 0], [1, 1]]])

    def test_merge_padding_and_attention_mask(self):
        padding_mask = ops.array([[1, 1, 0]])
        attention_mask = ops.array([[[0, 0, 1], [0, 1, 0], [1, 0, 0]]])
        inputs = random.uniform(shape=[1, 3, 2])
        merged_mask = utils.merge_padding_and_attention_mask(
            inputs,
            padding_mask,
            attention_mask,
        )
        self.assertAllEqual(merged_mask, [[[0, 0, 0], [0, 1, 0], [1, 0, 0]]])

    def test_bad_mask_shapes(self):
        with self.assertRaises(ValueError):
            padding_mask = ops.array([[[1, 1, 0], [1, 0, 0]]])
            attention_mask = ops.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
            inputs = random.uniform(shape=[1, 3, 2])
            utils.merge_padding_and_attention_mask(
                inputs,
                padding_mask,
                attention_mask,
            )

        with self.assertRaises(ValueError):
            padding_mask = ops.array([[1, 1, 0]])
            attention_mask = ops.array([[0, 0, 1], [1, 0, 0]])
            inputs = random.uniform(shape=[1, 3, 2])
            utils.merge_padding_and_attention_mask(
                inputs,
                padding_mask,
                attention_mask,
            )

    def test_compute_positions_from_mask(self):
        mask = ops.array(
            [
                [False, False, True, True, False],
                [True, False, True, False, True],
            ]
        )
        output = utils.compute_positions_from_mask(mask)

        expected_output = ops.array([[0, 0, 0, 1, 1], [0, 0, 1, 1, 2]])
        self.assertAllEqual(output, expected_output)
