import keras
import numpy as np
import pytest
from keras import ops
from packaging import version

from keras_hub.src.layers.modeling.non_max_supression import NonMaxSuppression
from keras_hub.src.tests.test_case import TestCase


class NonMaxSupressionTest(TestCase):
    @pytest.mark.skipif(
        version.parse(keras.__version__) < version.parse("3.8.0"),
        reason="Bbox utils are not supported before keras < 3.8.0",
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
        self.assertAllClose(outputs["labels"], [[0.0, 0.0], [0.0, 0.0]])
        self.assertAllClose(outputs["confidence"], [[0.9, 0.5], [0.7, 0.5]])

    @pytest.mark.skipif(
        version.parse(keras.__version__) < version.parse("3.8.0"),
        reason="Bbox utils are not supported before keras < 3.8.0",
    )
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
        self.assertAllClose(outputs["labels"], [[0.0], [0.0]])
        self.assertAllClose(outputs["confidence"], [[0.9], [0.7]])
