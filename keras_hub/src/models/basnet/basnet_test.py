import pytest
from keras import ops

from keras_hub.src.models.basnet.basnet import BASNetImageSegmenter
from keras_hub.src.models.basnet.basnet_preprocessor import BASNetPreprocessor
from keras_hub.src.models.resnet.resnet_backbone import ResNetBackbone
from keras_hub.src.tests.test_case import TestCase


class BASNetTest(TestCase):
    def setUp(self):
        self.images = ops.ones((2, 64, 64, 3))
        self.labels = ops.concatenate(
            (ops.zeros((2, 32, 64, 1)), ops.ones((2, 32, 64, 1))), axis=1
        )
        self.backbone = ResNetBackbone(
            input_conv_filters=[64],
            input_conv_kernel_sizes=[7],
            stackwise_num_filters=[64, 128, 256, 512],
            stackwise_num_blocks=[2, 2, 2, 2],
            stackwise_num_strides=[1, 2, 2, 2],
            block_type="basic_block",
        )
        self.preprocessor = BASNetPreprocessor()
        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": self.preprocessor,
            "num_classes": 1,
        }
        self.train_data = (self.images, self.labels)

    def test_basics(self):
        self.run_task_test(
            cls=BASNetImageSegmenter,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=[(2, 64, 64, 1)] * 8,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=BASNetImageSegmenter,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )

    def test_end_to_end_model_predict(self):
        model = BASNetImageSegmenter(**self.init_kwargs)
        outputs = model.predict(self.images)
        self.assertAllEqual(
            [output.shape for output in outputs], [(2, 64, 64, 1)] * 8
        )

    @pytest.mark.skip(reason="disabled until preset's been uploaded to Kaggle")
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in BASNetImageSegmenter.presets:
            self.run_preset_test(
                cls=BASNetImageSegmenter,
                preset=preset,
                input_data=self.images,
                expected_output_shape=[(2, 64, 64, 1)] * 8,
            )
