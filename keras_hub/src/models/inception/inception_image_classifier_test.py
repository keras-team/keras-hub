import pytest
from keras import ops

 
from keras_hub.src.models.inception.inception_backbone import InceptionBackbone
from keras_hub.src.models.inception.inception_image_classifier import (
    InceptionImageClassifier,
)
from keras_hub.src.tests.test_case import TestCase


class InceptionImageClassifierTest(TestCase):
    def setUp(self):
        self.images = ops.ones((2, 16, 16, 3))
        self.labels = [0, 3]
        self.backbone = InceptionBackbone(
            initial_filters=[64, 192],
            initial_strides=[2, 1],
            inception_config=[
                [64, 96, 128, 16, 32, 32],
                [128, 128, 192, 32, 96, 64],
                [192, 96, 208, 16, 48, 64],
            ],
            aux_classifiers=False,
            image_shape=(16, 16, 3),
            dtype="float32",  # Add explicit dtype
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "num_classes": 2,
            "pooling": "avg",
            "activation": "softmax",
            "aux_classifiers": False,
        }
        self.train_data = (self.images, self.labels)

    def test_classifier_basics(self):
        pytest.skip(
            reason="TODO: enable after preprocessor flow is figured out"
        )
        backbone = InceptionBackbone(
            initial_filters=[64],
            initial_strides=[2],
            inception_config=[
                [64, 96, 128, 16, 32, 32],
                [128, 128, 192, 32, 96, 64],
                [192, 96, 208, 16, 48, 64],
            ],
            aux_classifiers=False,
            image_shape=(16, 16, 3),
            dtype="float32",  # Add explicit dtype
        )
        init_kwargs = {
            "backbone": backbone,
            "num_classes": 2,
            "pooling": "avg",
            "activation": "softmax",
            "aux_classifiers": False,
        }
        self.run_task_test(
            cls=InceptionImageClassifier,
            init_kwargs=init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 2),
        )

    def test_head_dtype(self):
        model = InceptionImageClassifier(
            backbone=InceptionBackbone(
                initial_filters=[64],
                initial_strides=[2],
                inception_config=[
                    [64, 96, 128, 16, 32, 32],
                    [128, 128, 192, 32, 96, 64],
                    [192, 96, 208, 16, 48, 64],
                ],
                aux_classifiers=False,
                image_shape=(16, 16, 3),
                dtype="float32",  # Add explicit dtype
            ),
            num_classes=2,
            pooling="avg",
            activation="softmax",
            aux_classifiers=False,
            head_dtype="bfloat16"
        )
        self.assertEqual(model.output_dense.compute_dtype, "bfloat16")

    def test_auxiliary_branches(self):
        # Test with auxiliary branches enabled
        backbone = InceptionBackbone(
            initial_filters=[64],
            initial_strides=[2],
            inception_config=[
                [64, 96, 128, 16, 32, 32],
                [128, 128, 192, 32, 96, 64],
                [192, 96, 208, 16, 48, 64],
            ],
            aux_classifiers=False,
            image_shape=(16, 16, 3),
        )
        
        init_kwargs = {
            "backbone": backbone,
            "num_classes": 2,
            "pooling": "avg",
            "activation": "softmax",
            "aux_classifiers": False,
        }
        
        # Create model with auxiliary branches
        aux_backbone = InceptionBackbone(
            initial_filters=[64],
            initial_strides=[2],
            inception_config=[
                [64, 96, 128, 16, 32, 32],
                [128, 128, 192, 32, 96, 64],
                [192, 96, 208, 16, 48, 64],
            ],
            aux_classifiers=True,
            image_shape=(16, 16, 3),
        )
        
        aux_kwargs = init_kwargs.copy()
        aux_kwargs["backbone"] = aux_backbone
        aux_kwargs["aux_classifiers"] = True
        
        model = InceptionImageClassifier(**aux_kwargs)
        outputs = model(self.images, training=True)
        
        # Check if we have main and auxiliary outputs
        self.assertIsInstance(outputs, dict)
        self.assertIn("main", outputs)
        self.assertIn("aux1", outputs)
        self.assertIn("aux2", outputs)
        
        # Check output shapes
        self.assertEqual(outputs["main"].shape, (2, 2))
        self.assertEqual(outputs["aux1"].shape, (2, 2))
        self.assertEqual(outputs["aux2"].shape, (2, 2))

    @pytest.mark.large
    def test_smallest_preset(self):
        # Test that our forward pass is stable!
        image_batch = self.load_test_image()[None, ...] / 255.0
        self.run_preset_test(
            cls=InceptionImageClassifier,
            preset="inception_v3_imagenet",
            input_data=image_batch,
            expected_output_shape=(1, 1000),
            expected_labels=[85],
        )

    @pytest.mark.large
    def test_saved_model(self):
        backbone = InceptionBackbone(
            initial_filters=[64],
            initial_strides=[2],
            inception_config=[
                [64, 96, 128, 16, 32, 32],
                [128, 128, 192, 32, 96, 64],
                [192, 96, 208, 16, 48, 64],
            ],
            aux_classifiers=False,
            image_shape=(16, 16, 3),
        )
        init_kwargs = {
            "backbone": backbone,
            "num_classes": 2,
            "pooling": "avg",
            "activation": "softmax",
            "aux_classifiers": False,
        }
        self.run_model_saving_test(
            cls=InceptionImageClassifier,
            init_kwargs=init_kwargs,
            input_data=self.images,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in InceptionImageClassifier.presets:
            self.run_preset_test(
                cls=InceptionImageClassifier,
                preset=preset,
                init_kwargs={"num_classes": 2},
                input_data=self.images,
                expected_output_shape=(2, 2),
            )