"""Tests for BLIP-2 Q-Former."""

import numpy as np

from keras_hub.src.models.blip2.blip2_qformer import BLIP2QFormer
from keras_hub.src.tests.test_case import TestCase


class BLIP2QFormerTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "num_query_tokens": 32,
            "num_layers": 4,
            "num_heads": 4,
            "hidden_dim": 64,
            "intermediate_dim": 128,
            "vision_dim": 128,
            "cross_attention_frequency": 2,
            "dropout": 0.0,
            "layer_norm_epsilon": 1e-12,
        }
        self.vision_features = np.random.uniform(size=(2, 257, 128)).astype(
            "float32"
        )

    def test_qformer_output_shape(self):
        qformer = BLIP2QFormer(**self.init_kwargs)
        output = qformer(self.vision_features)
        self.assertEqual(output.shape, (2, 32, 64))

    def test_serialization(self):
        qformer = BLIP2QFormer(**self.init_kwargs)
        qformer(self.vision_features)

        restored = BLIP2QFormer.from_config(qformer.get_config())
        restored(self.vision_features)
        restored.set_weights(qformer.get_weights())

        self.assertEqual(qformer.get_config(), restored.get_config())
        self.assertAllClose(
            qformer(self.vision_features), restored(self.vision_features)
        )

    def test_cross_attention_every_layer(self):
        kwargs = {**self.init_kwargs, "cross_attention_frequency": 1}
        qformer = BLIP2QFormer(**kwargs)
        output = qformer(self.vision_features)
        self.assertEqual(output.shape, (2, 32, 64))

    def test_dropout_training_vs_inference(self):
        kwargs = {**self.init_kwargs, "dropout": 0.5}
        qformer = BLIP2QFormer(**kwargs)
        vision_features = np.random.uniform(size=(1, 5, 128)).astype("float32")

        out_a = qformer(vision_features, training=True)
        out_b = qformer(vision_features, training=True)
        self.assertNotAllClose(out_a, out_b)

        out_c = qformer(vision_features, training=False)
        out_d = qformer(vision_features, training=False)
        self.assertAllClose(out_c, out_d)

    def test_batch_elements_are_independent(self):
        qformer = BLIP2QFormer(**self.init_kwargs)
        vision_features = np.random.uniform(size=(2, 5, 128)).astype("float32")
        outputs = qformer(vision_features)
        self.assertNotAllClose(outputs[0], outputs[1])

    def test_float16_dtype(self):
        qformer = BLIP2QFormer(**self.init_kwargs, dtype="float16")
        vision_features = np.random.uniform(size=(1, 5, 128)).astype("float16")
        outputs = qformer(vision_features)
        self.assertIn("float16", str(outputs.dtype))
