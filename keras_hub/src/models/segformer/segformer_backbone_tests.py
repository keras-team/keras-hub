import numpy as np
import pytest
from keras import ops

from keras_hub.api.models import MiTBackbone
from keras_hub.api.models import SegFormerBackbone
from keras_hub.src.tests.test_case import TestCase


class SegFormerTest(TestCase):
    def setUp(self):
        encoder_init_kwargs = {
            "layerwise_depths": [2, 2],
            "image_shape": (32, 32, 3),
            "hidden_dims": [4, 8],
            "num_layers": 2,
            "layerwise_num_heads": [1, 2],
            "layerwise_sr_ratios": [8, 4],
            "max_drop_path_rate": 0.1,
            "layerwise_patch_sizes": [7, 3],
            "layerwise_strides": [4, 2],
        }
        image_encoder = MiTBackbone(**encoder_init_kwargs)
        projection_filters = 256
        self.input_size = 32
        self.input_data = ops.ones((2, self.input_size, self.input_size, 3))

        self.init_kwargs = {
            "projection_filters": projection_filters,
            "image_encoder": image_encoder,
        }

    def test_segformer_backbone_construction(self):
        SegFormerBackbone(**self.init_kwargs)

    @pytest.mark.large
    def test_segformer_call(self):
        segformer_backbone = SegFormerBackbone(
            image_encoder=self.init_kwargs["image_encoder"],
            projection_filters=self.init_kwargs["projection_filters"],
        )

        images = np.random.uniform(size=(2, 32, 32, 3))
        segformer_output = segformer_backbone(images)
        segformer_predict = segformer_backbone.predict(images)

        assert segformer_output.shape == (2, 8, 8, 256)
        assert segformer_predict.shape == (2, 8, 8, 256)

    def test_backbone_basics(self):
        self.run_vision_backbone_test(
            cls=SegFormerBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 8, 8, 256),
            run_mixed_precision_check=False,
            run_quantization_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=SegFormerBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
