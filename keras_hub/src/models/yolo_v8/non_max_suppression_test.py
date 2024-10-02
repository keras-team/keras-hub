import numpy as np
from keras import ops

from keras_hub.src.models.yolo_v8.non_max_suppression import NonMaxSuppression
from keras_hub.src.tests.test_case import TestCase


class NonMaxSupressionTest(TestCase):
    def test_layer_behaviors(self):
        batch_size = 2
        expected_output_shape = {
            "idx": (batch_size, 2),
            "boxes": (batch_size, 2, 4),
            "confidence": (batch_size, 2),
            "classes": (batch_size, 2),
            "num_detections": (batch_size,),
        }
        init_kwargs = {
            "bounding_box_format": "yxyx",
            "from_logits": False,
            "iou_threshold": 1.0,
            "confidence_threshold": 0.45,
            "max_detections": 2,
        }
        boxes = np.random.uniform(low=0, high=1, size=(batch_size, 5, 4))
        boxes = boxes.astype("float32")
        classes = np.array([[0.1, 0.1, 0.4, 0.5, 0.9],
                            [0.7, 0.5, 0.3, 0.0, 0.0]], "float32")
        classes = np.expand_dims(classes, axis=-1)
        self.run_layer_test(
            cls=NonMaxSuppression,
            init_kwargs=init_kwargs,
            input_data={"box_prediction": boxes, "class_prediction": classes},
            expected_output_shape=expected_output_shape,
            run_training_check=False,
            run_precision_checks=False
        )

    def test_confidence_threshold(self):
        boxes = np.random.uniform(low=0, high=1, size=(2, 5, 4))
        classes = ops.expand_dims(
            np.array(
                [[0.1, 0.1, 0.4, 0.9, 0.5], [0.7, 0.5, 0.3, 0.0, 0.0]],
                "float32",
            ),
            axis=-1,
        )

        nms = NonMaxSuppression(
            bounding_box_format="yxyx",
            from_logits=False,
            iou_threshold=1.0,
            confidence_threshold=0.45,
            max_detections=2,
        )

        outputs = nms(boxes, classes)

        self.assertAllClose(
            outputs["boxes"], [boxes[0][-2:, ...], boxes[1][:2, ...]]
        )
        self.assertAllClose(outputs["classes"], [[0.0, 0.0], [0.0, 0.0]])
        self.assertAllClose(outputs["confidence"], [[0.9, 0.5], [0.7, 0.5]])

    def test_max_detections(self):
        boxes = np.random.uniform(low=0, high=1, size=(2, 5, 4))
        classes = ops.expand_dims(
            np.array(
                [[0.1, 0.1, 0.4, 0.5, 0.9], [0.7, 0.5, 0.3, 0.0, 0.0]],
                "float32",
            ),
            axis=-1,
        )

        nms = NonMaxSuppression(
            bounding_box_format="yxyx",
            from_logits=False,
            iou_threshold=1.0,
            confidence_threshold=0.1,
            max_detections=1,
        )

        outputs = nms(boxes, classes)

        self.assertAllClose(
            outputs["boxes"], [boxes[0][-1:, ...], boxes[1][:1, ...]]
        )
        self.assertAllClose(outputs["classes"], [[0.0], [0.0]])
        self.assertAllClose(outputs["confidence"], [[0.9], [0.7]])
