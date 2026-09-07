import pytest
import numpy as np
from keras import ops

from keras_hub.src.models.swin_transformer.swin_transformer_backbone import (
    SwinTransformerBackbone,
)
from keras_hub.src.tests.test_case import TestCase


class SwinTransformerBackboneTest(TestCase):
    def setUp(self):
        super().setUp()
        self.init_kwargs = {
            "image_shape": (32, 32, 3),
            "embed_dim": 32,
            "depths": (2, 2),
            "num_heads": (2, 4),
            "window_size": 4,
        }
        self.input_data = ops.ones((1, 32, 32, 3))

    def test_backbone_basics(self):
        self.run_vision_backbone_test(
            cls=SwinTransformerBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(1, 16, 64),
            run_data_format_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=SwinTransformerBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
    def test_drop_path_training_step(self):
        """DropPath should not corrupt tensor rank during training."""
        backbone = SwinTransformerBackbone(
                embed_dim=96, depths=(2, 2, 6, 2),
                num_heads=(3, 6, 12, 24), window_size=7,
                )
        inputs = keras.Input(shape=(224, 224, 3))
        feat = backbone(inputs)
        pooled = keras.layers.GlobalAveragePooling1D()(feat)
        outputs = keras.layers.Dense(10, activation="softmax")(pooled)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy")
        x = np.random.randn(2, 224, 224, 3).astype(np.float32)
        y = np.random.randint(0, 10, size=(2,))
        # Before fix: ValueError: too many values to unpack (expected 3)
        model.train_on_batch(x, y)
