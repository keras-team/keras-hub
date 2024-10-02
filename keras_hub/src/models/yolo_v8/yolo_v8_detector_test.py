import os

import keras
import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized
from keras import ops

import keras_hub
from keras_hub.src import bounding_box
from keras_hub.src.models.yolo_v8 import yolo_v8_detector_presets
from keras_hub.src.tests.test_case import TestCase

test_backbone_presets = [
    "csp_darknet_tiny",
    "densenet121",
    "efficientnetv2_b0",
    "mobilenet_v3_small",
    "resnet18",
    "resnet18_v2",
]


def _create_bounding_box_dataset(
    bounding_box_format, use_dictionary_box_format=False
):
    # Just about the easiest dataset you can have, all classes are 0, all boxes
    # are exactly the same. [1, 1, 2, 2] are the coordinates in xyxy.
    xs = np.random.normal(size=(1, 512, 512, 3))
    xs = np.tile(xs, [5, 1, 1, 1])

    y_classes = np.zeros((5, 3), "float32")

    ys = np.array(
        [
            [0.1, 0.1, 0.23, 0.23],
            [0.67, 0.75, 0.23, 0.23],
            [0.25, 0.25, 0.23, 0.23],
        ],
        "float32",
    )
    ys = np.expand_dims(ys, axis=0)
    ys = np.tile(ys, [5, 1, 1])
    ys = ops.convert_to_numpy(
        bounding_box.convert_format(
            ys,
            source="rel_xywh",
            target=bounding_box_format,
            images=xs,
            dtype="float32",
        )
    )
    num_dets = np.ones([5])

    if use_dictionary_box_format:
        return tf.data.Dataset.from_tensor_slices(
            {
                "images": xs,
                "bounding_boxes": {
                    "boxes": ys,
                    "classes": y_classes,
                    "num_dets": num_dets,
                },
            }
        ).batch(5, drop_remainder=True)
    else:
        return xs, {"boxes": ys, "classes": y_classes}


class YOLOV8DetectorTest(TestCase):
    @pytest.mark.large  # Fit is slow, so mark these large.
    def test_fit(self):
        bounding_box_format = "xywh"
        yolo = keras_hub.models.YOLOV8ObjectDetector(
            num_classes=2,
            fpn_depth=1,
            bounding_box_format=bounding_box_format,
            backbone=keras_hub.models.YOLOV8Backbone.from_preset(
                "yolo_v8_xs_backbone"
            ),
        )

        yolo.compile(
            optimizer="adam",
            classification_loss="binary_crossentropy",
            box_loss="ciou",
        )
        xs, ys = _create_bounding_box_dataset(bounding_box_format)

        yolo.fit(x=xs, y=ys, epochs=1)

    @pytest.mark.tf_keras_only
    @pytest.mark.large  # Fit is slow, so mark these large.
    def test_fit_with_ragged_tensors(self):
        bounding_box_format = "xywh"
        yolo = keras_hub.models.YOLOV8ObjectDetector(
            num_classes=2,
            fpn_depth=1,
            bounding_box_format=bounding_box_format,
            backbone=keras_hub.models.YOLOV8Backbone.from_preset(
                "yolo_v8_xs_backbone"
            ),
        )

        yolo.compile(
            optimizer="adam",
            classification_loss="binary_crossentropy",
            box_loss="ciou",
        )
        xs, ys = _create_bounding_box_dataset(bounding_box_format)
        ys = bounding_box.to_ragged(ys)

        yolo.fit(x=xs, y=ys, epochs=1)

    @pytest.mark.large  # Fit is slow, so mark these large.
    def test_fit_with_no_valid_gt_bbox(self):
        bounding_box_format = "xywh"
        yolo = keras_hub.models.YOLOV8ObjectDetector(
            num_classes=1,
            fpn_depth=1,
            bounding_box_format=bounding_box_format,
            backbone=keras_hub.models.YOLOV8Backbone.from_preset(
                "yolo_v8_xs_backbone"
            ),
        )

        yolo.compile(
            optimizer="adam",
            classification_loss="binary_crossentropy",
            box_loss="ciou",
        )
        xs, ys = _create_bounding_box_dataset(bounding_box_format)
        # Make all bounding_boxes invalid and filter out them
        ys["classes"] = -np.ones_like(ys["classes"])

        yolo.fit(x=xs, y=ys, epochs=1)

    @pytest.mark.skip(reason="Missing presets")
    def test_trainable_weight_count(self):
        yolo = keras_hub.models.YOLOV8ObjectDetector(
            num_classes=2,
            fpn_depth=1,
            bounding_box_format="xywh",
            backbone=keras_hub.models.YOLOV8Backbone.from_preset(
                "yolo_v8_s_backbone"
            ),
        )

        self.assertEqual(len(yolo.trainable_weights), 195)

    @pytest.mark.skip(reason="Missing presets")
    def test_bad_loss(self):
        yolo = keras_hub.models.YOLOV8ObjectDetector(
            num_classes=2,
            fpn_depth=1,
            bounding_box_format="xywh",
            backbone=keras_hub.models.YOLOV8Backbone.from_preset(
                "yolo_v8_xs_backbone"
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "Invalid box loss",
        ):
            yolo.compile(
                box_loss="bad_loss", classification_loss="binary_crossentropy"
            )

        with self.assertRaisesRegex(
            ValueError,
            "Invalid classification loss",
        ):
            yolo.compile(box_loss="ciou", classification_loss="bad_loss")

    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_model(self):
        model = keras_hub.models.YOLOV8ObjectDetector(
            num_classes=20,
            bounding_box_format="xywh",
            fpn_depth=1,
            backbone=keras_hub.models.YOLOV8Backbone.from_preset(
                "yolo_v8_xs_backbone"
            ),
        )
        xs, _ = _create_bounding_box_dataset("xywh")
        model_output = model(xs)
        save_path = os.path.join(
            self.get_temp_dir(), "yolo_v8_xs_detector.keras"
        )
        model.save(save_path)
        # TODO: Remove the need to pass the `custom_objects` parameter.
        restored_model = keras.saving.load_model(
            save_path,
            custom_objects={"YOLOV8ObjectDetector": keras_hub.models.YOLOV8ObjectDetector},
        )

        # Check we got the real object back.
        self.assertIsInstance(restored_model, keras_hub.models.YOLOV8ObjectDetector)

        # Check that output matches.
        restored_output = restored_model(xs)
        self.assertAllClose(
            ops.convert_to_numpy(model_output["boxes"]),
            ops.convert_to_numpy(restored_output["boxes"]),
        )
        self.assertAllClose(
            ops.convert_to_numpy(model_output["classes"]),
            ops.convert_to_numpy(restored_output["classes"]),
        )

    # TODO(tirthasheshpatel): Support updating prediction decoder in Keras Core.
    @pytest.mark.skip(reason="Missing presets")
    @pytest.mark.tf_keras_only
    def test_update_prediction_decoder(self):
        yolo = keras_hub.models.YOLOV8ObjectDetector(
            num_classes=2,
            fpn_depth=1,
            bounding_box_format="xywh",
            backbone=keras_hub.models.YOLOV8Backbone.from_preset(
                "yolo_v8_s_backbone"
            ),
            prediction_decoder=keras_hub.layers.NonMaxSuppression(
                bounding_box_format="xywh",
                from_logits=False,
                confidence_threshold=0.0,
                iou_threshold=1.0,
            ),
        )

        image = np.ones((1, 512, 512, 3))

        outputs = yolo.predict(image)
        # We predicted at least 1 box with confidence_threshold 0
        self.assertGreater(outputs["boxes"].shape[0], 0)

        yolo.prediction_decoder = keras_hub.layers.NonMaxSuppression(
            bounding_box_format="xywh",
            from_logits=False,
            confidence_threshold=1.0,
            iou_threshold=1.0,
        )

        outputs = yolo.predict(image)
        # We predicted no boxes with confidence threshold 1
        self.assertAllEqual(outputs["boxes"], -np.ones_like(outputs["boxes"]))
        self.assertAllEqual(
            outputs["confidence"], -np.ones_like(outputs["confidence"])
        )
        self.assertAllEqual(
            outputs["classes"], -np.ones_like(outputs["classes"])
        )


@pytest.mark.large
class YOLOV8ObjectDetectorSmokeTest(TestCase):
    @pytest.mark.skip(reason="Missing presets")
    @parameterized.named_parameters(
        *[(preset, preset) for preset in test_backbone_presets]
    )
    @pytest.mark.extra_large
    def test_backbone_preset(self, preset):
        model = keras_hub.models.YOLOV8ObjectDetector.from_preset(
            preset,
            num_classes=20,
            bounding_box_format="xywh",
        )
        xs, _ = _create_bounding_box_dataset(bounding_box_format="xywh")
        output = model(xs)

        # 64 represents number of parameters in a box
        # 5376 is the number of anchors for a 512x512 image
        self.assertEqual(output["boxes"].shape, (xs.shape[0], 5376, 64))

    @pytest.mark.skip(reason="Missing presets")
    def test_preset_with_forward_pass(self):
        model = keras_hub.models.YOLOV8Detector.from_preset(
            "yolo_v8_m_pascalvoc",
            bounding_box_format="xywh",
        )

        image = np.ones((1, 512, 512, 3))
        encoded_predictions = model(image)

        self.assertAllClose(
            ops.convert_to_numpy(encoded_predictions["boxes"][0, 0:5, 0]),
            [-0.8303556, 0.75213313, 1.809204, 1.6576759, 1.4134747],
        )
        self.assertAllClose(
            ops.convert_to_numpy(encoded_predictions["classes"][0, 0:5, 0]),
            [
                7.6146556e-08,
                8.0103280e-07,
                9.7873999e-07,
                2.2314548e-06,
                2.5051115e-06,
            ],
        )


@pytest.mark.extra_large
class YOLOV8DetectorPresetFullTest(TestCase):
    """
    Test the full enumeration of our presets.
    This every presets for YOLOV8Detector and is only run manually.
    Run with:
    `pytest keras_hub/models/object_detection/yolo_v8/yolo_v8_detector_test.py --run_extra_large`
    """  # noqa: E501

    @pytest.mark.skip(reason="Missing presets")
    def test_load_yolo_v8_detector(self):
        input_data = np.ones(shape=(2, 224, 224, 3))
        for preset in yolo_v8_detector_presets:
            model = keras_hub.models.YOLOV8Detector.from_preset(
                preset, bounding_box_format="xywh"
            )
            model(input_data)
