import pytest
from keras import ops

from keras_hub.src.models.efficientnet.efficientnet_backbone import (
    EfficientNetBackbone,
)
from keras_hub.src.models.efficientnet.efficientnet_image_classifier import (
    EfficientNetImageClassifier,
)
from keras_hub.src.models.efficientnet.efficientnet_image_classifier_preprocessor import (  # noqa: E501
    EfficientNetImageClassifierPreprocessor,
)
from keras_hub.src.models.efficientnet.efficientnet_image_converter import (
    EfficientNetImageConverter,
)
from keras_hub.src.tests.test_case import TestCase


class EfficientNetImageClassifierTest(TestCase):
    def setUp(self):
        self.images = ops.ones((2, 16, 16, 3))
        self.labels = [0, 3]
        backbone = EfficientNetBackbone(
            width_coefficient=1.0,
            depth_coefficient=1.0,
            stackwise_kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
            stackwise_num_repeats=[1, 2, 2, 3, 3, 4, 1],
            stackwise_input_filters=[32, 16, 24, 40, 80, 112, 192],
            stackwise_output_filters=[16, 24, 40, 80, 112, 192, 320],
            stackwise_expansion_ratios=[1, 6, 6, 6, 6, 6, 6],
            stackwise_strides=[1, 2, 2, 2, 1, 2, 1],
            stackwise_block_types=["v1"] * 7,
            stackwise_squeeze_and_excite_ratios=[0.25] * 7,
            min_depth=None,
            include_stem_padding=True,
            use_depth_divisor_as_min_depth=True,
            cap_round_filter_decrease=True,
            stem_conv_padding="valid",
            batch_norm_momentum=0.9,
            batch_norm_epsilon=1e-5,
            dropout=0,
            projection_activation=None,
        )
        self.init_kwargs = {
            "backbone": backbone,
            "num_classes": 1000,
            "preprocessor": EfficientNetImageClassifierPreprocessor(
                image_converter=EfficientNetImageConverter(image_size=(16, 16))
            ),
        }
        self.train_data = (self.images, self.labels)

    def test_classifier_basics(self):
        pytest.skip(
            reason="TODO: enable after preprocessor flow is figured out"
        )
        self.run_task_test(
            cls=EfficientNetImageClassifier,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 2),
        )

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        # Test that our forward pass is stable!
        image_batch = self.load_test_image()[None, ...] / 255.0
        self.run_preset_test(
            cls=EfficientNetImageClassifier,
            preset="efficientnet_b0_ra_imagenet",
            input_data=image_batch,
            expected_output_shape=(1, 1000),
            expected_labels=[85],
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=EfficientNetImageClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in EfficientNetImageClassifier.presets:
            self.run_preset_test(
                cls=EfficientNetImageClassifier,
                preset=preset,
                init_kwargs={"num_classes": 2},
                input_data=self.images,
                expected_output_shape=(2, 2),
            )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=EfficientNetImageClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )
