import pytest
from keras import ops

from keras_hub.src.models.resnet.resnet_backbone import ResNetBackbone
from keras_hub.src.models.resnet.resnet_image_classifier import (
    ResNetImageClassifier,
)
from keras_hub.src.tests.test_case import TestCase


class ResNetImageClassifierTest(TestCase):
    def setUp(self):
        self.images = ops.ones((2, 16, 16, 3))
        self.labels = [0, 3]
        self.backbone = ResNetBackbone(
            input_conv_filters=[64],
            input_conv_kernel_sizes=[7],
            stackwise_num_filters=[64, 64, 64],
            stackwise_num_blocks=[2, 2, 2],
            stackwise_num_strides=[1, 2, 2],
            block_type="basic_block",
            use_pre_activation=True,
            image_shape=(16, 16, 3),
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "num_classes": 2,
            "pooling": "avg",
            "activation": "softmax",
        }
        self.train_data = (self.images, self.labels)

    def test_classifier_basics(self):
        pytest.skip(
            reason="TODO: enable after preprocessor flow is figured out"
        )
        self.run_task_test(
            cls=ResNetImageClassifier,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 2),
        )

    def test_head_dtype(self):
        model = ResNetImageClassifier(**self.init_kwargs, head_dtype="bfloat16")
        self.assertEqual(model.output_dense.compute_dtype, "bfloat16")

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        # Test that our forward pass is stable!
        image_batch = self.load_test_image()[None, ...] / 255.0
        self.run_preset_test(
            cls=ResNetImageClassifier,
            preset="resnet_18_imagenet",
            input_data=image_batch,
            expected_output_shape=(1, 1000),
            expected_labels=[85],
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=ResNetImageClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )

    def test_litert_export(self):
        """Test LiteRT export for ResNetImageClassifier with small test
        model."""
        model = ResNetImageClassifier(**self.init_kwargs)
        expected_output_shape = (2, 2)  # 2 images, 2 classes

        self.run_litert_export_test(
            model=model,
            input_data=self.images,
            expected_output_shape=expected_output_shape,
            comparison_mode="statistical",
            output_thresholds={"*": {"max": 5e-5, "mean": 1e-5}},
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in ResNetImageClassifier.presets:
            self.run_preset_test(
                cls=ResNetImageClassifier,
                preset=preset,
                init_kwargs={"num_classes": 2},
                input_data=self.images,
                expected_output_shape=(2, 2),
            )
