import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.bounding_box.converters import _decode_deltas_to_boxes
from keras_hub.src.bounding_box.converters import convert_format
from keras_hub.src.models.image_object_detector import ImageObjectDetector
from keras_hub.src.models.retinanet.non_max_supression import NonMaxSuppression
from keras_hub.src.models.retinanet.prediction_head import PredictionHead
from keras_hub.src.models.retinanet.retinanet_backbone import RetinaNetBackbone
from keras_hub.src.models.retinanet.retinanet_object_detector_preprocessor import (
    RetinaNetObjectDetectorPreprocessor,
)

BOX_VARIANCE = [1.0, 1.0, 1.0, 1.0]


@keras_hub_export("keras_hub.models.RetinaNetObjectDetector")
class RetinaNetObjectDetector(ImageObjectDetector):
    """RetinaNet object detector model.

    This class implements the RetinaNet object detection architecture.
    It consists of a feature extractor backbone, a feature pyramid network(FPN),
    and two prediction heads for classification and regression.

    Args:
        backbone: `keras.Model`. A `keras.models.RetinaNetBackbone` class, defining the
            backbone network architecture.
        label_encoder: `keras.layers.Layer`. A `RetinaNetLabelEncoder` class
            that accepts an image Tensor, a bounding box Tensor and a bounding
            box class Tensor to its `call()` method, and returns
            `RetinaNetObjectDetector` training targets.
        anchor_generator: A `keras_Hub.layers.AnchorGenerator`.
        num_classes: The number of object classes to be detected.
        ground_truth_bounding_box_format: Ground truth bounding box format.
            Refer TODO: https://github.com/keras-team/keras-hub/issues/1907
            Ensure that ground truth boxes follow one of the following formats.
                - `rel_xyxy`
                - `rel_yxyx`
                - `rel_xywh`
        target_bounding_box_format: Target bounding box format.
            Refer TODO: https://github.com/keras-team/keras-hub/issues/1907
        use_prediction_head_norm: bool. Whether to use Group Normalization after
            the convolution layers in prediction head. Defaults to `False`.
        preprocessor: Optional. An instance of the
            `RetinaNetObjectDetectorPreprocessor` class or a custom preprocessor.
        activation: Optional. The activation function to be used in the
            classification head.
        dtype: Optional. The data type for the prediction heads.
        prediction_decoder: Optional. A `keras.layers.Layer` that is
            responsible for transforming RetinaNet predictions into usable
            bounding box Tensors.
            Defaults to `NonMaxSuppression` class instance.
    """

    backbone_cls = RetinaNetBackbone
    preprocessor_cls = RetinaNetObjectDetectorPreprocessor

    def __init__(
        self,
        backbone,
        label_encoder,
        anchor_generator,
        num_classes,
        ground_truth_bounding_box_format,
        target_bounding_box_format,
        use_prediction_head_norm=False,
        classification_head_prior_probability=0.01,
        preprocessor=None,
        activation=None,
        dtype=None,
        prediction_decoder=None,
        **kwargs,
    ):
        if "rel" not in ground_truth_bounding_box_format:
            raise ValueError(
                f"Only relative bounding box formats are supported "
                f"Received ground_truth_bounding_box_format="
                f"`{ground_truth_bounding_box_format}`. "
                f"Please provide a `ground_truth_bounding_box_format` from one of "
                f"the following `rel_xyxy` or `rel_yxyx` or `rel_xywh`. "
                f"Ensure that the provided ground truth bounding boxes are "
                f"normalized and relative to the image size. "
            )
        # === Layers ===
        image_input = keras.layers.Input(backbone.image_shape, name="images")
        head_dtype = dtype or backbone.dtype_policy

        box_head = PredictionHead(
            output_filters=anchor_generator.num_base_anchors * 4,
            num_conv_layers=4,
            num_filters=256,
            use_group_norm=use_prediction_head_norm,
            use_prior_probability=True,
            prior_probability=classification_head_prior_probability,
            dtype=head_dtype,
            name="box_head",
        )
        classification_head = PredictionHead(
            output_filters=anchor_generator.num_base_anchors * num_classes,
            num_conv_layers=4,
            num_filters=256,
            use_group_norm=use_prediction_head_norm,
            dtype=head_dtype,
            name="classification_head",
        )

        # === Functional Model ===
        feature_map = backbone(image_input)

        cls_pred = []
        box_pred = []
        for level in feature_map:
            box_pred.append(
                keras.layers.Reshape((-1, 4), name=f"box_pred_{level}")(
                    box_head(feature_map[level])
                )
            )
            cls_pred.append(
                keras.layers.Reshape(
                    (-1, num_classes), name=f"cls_pred_{level}"
                )(classification_head(feature_map[level]))
            )

        cls_pred = keras.layers.Concatenate(axis=1, name="classification")(
            cls_pred
        )
        # box_pred is always in "center_yxhw" delta-encoded no matter what
        # format you pass in.
        box_pred = keras.layers.Concatenate(axis=1, name="box")(box_pred)

        outputs = {"box": box_pred, "classification": cls_pred}

        super().__init__(
            inputs=image_input,
            outputs=outputs,
            **kwargs,
        )

        # === Config ===
        self.ground_truth_bounding_box_format = ground_truth_bounding_box_format
        self.target_bounding_box_format = target_bounding_box_format
        self.use_prediction_head_norm = use_prediction_head_norm
        self.num_classes = num_classes
        self.backbone = backbone
        self.preprocessor = preprocessor
        self.label_encoder = label_encoder
        self.anchor_generator = anchor_generator
        self.activation = activation
        self.box_head = box_head
        self.classification_head = classification_head
        self._prediction_decoder = prediction_decoder or NonMaxSuppression(
            from_logits=(activation != keras.activations.sigmoid),
            bounding_box_format=self.target_bounding_box_format,
        )

    def compute_loss(self, x, y, y_pred, sample_weight, **kwargs):
        y_for_label_encoder = convert_format(
            y,
            source=self.ground_truth_bounding_box_format,
            target=self.label_encoder.bounding_box_format,
            images=x,
        )

        boxes, classes = self.label_encoder(
            images=x,
            gt_boxes=y_for_label_encoder["boxes"],
            gt_classes=y_for_label_encoder["classes"],
        )

        box_pred = y_pred["box"]
        cls_pred = y_pred["classification"]

        if boxes.shape[-1] != 4:
            raise ValueError(
                "boxes should have shape (None, None, 4). Got "
                f"boxes.shape={tuple(boxes.shape)}"
            )

        if box_pred.shape[-1] != 4:
            raise ValueError(
                "box_pred should have shape (None, None, 4). Got "
                f"box_pred.shape={tuple(box_pred.shape)}. Does your model's "
                "`num_classes` parameter match your losses `num_classes` "
                "parameter?"
            )
        if cls_pred.shape[-1] != self.num_classes:
            raise ValueError(
                "cls_pred should have shape (None, None, 4). Got "
                f"cls_pred.shape={tuple(cls_pred.shape)}. Does your model's "
                "`num_classes` parameter match your losses `num_classes` "
                "parameter?"
            )

        cls_labels = ops.one_hot(
            ops.cast(classes, "int32"), self.num_classes, dtype="float32"
        )
        positive_mask = ops.cast(ops.greater(classes, -1.0), dtype="float32")
        normalizer = ops.sum(positive_mask)
        cls_weights = ops.cast(ops.not_equal(classes, -2.0), dtype="float32")
        cls_weights /= normalizer
        box_weights = positive_mask / normalizer

        y_true = {
            "box": boxes,
            "classification": cls_labels,
        }
        sample_weights = {
            "box": box_weights,
            "classification": cls_weights,
        }
        zero_weight = {
            "box": ops.zeros_like(box_weights),
            "classification": ops.zeros_like(cls_weights),
        }

        sample_weight = ops.cond(
            normalizer == 0,
            lambda: zero_weight,
            lambda: sample_weights,
        )
        return super().compute_loss(
            x=x, y=y_true, y_pred=y_pred, sample_weight=sample_weight, **kwargs
        )

    def predict_step(self, *args):
        outputs = super().predict_step(*args)
        if isinstance(outputs, tuple):
            return self.decode_predictions(outputs[0], args[-1]), outputs[1]
        return self.decode_predictions(outputs, *args)

    @property
    def prediction_decoder(self):
        return self._prediction_decoder

    @prediction_decoder.setter
    def prediction_decoder(self, prediction_decoder):
        if prediction_decoder.bounding_box_format != self.bounding_box_format:
            raise ValueError(
                "Expected `prediction_decoder` and `RetinaNet` to "
                "use the same `bounding_box_format`, but got "
                "`prediction_decoder.bounding_box_format="
                f"{prediction_decoder.bounding_box_format}`, and "
                "`self.bounding_box_format="
                f"{self.bounding_box_format}`."
            )
        self._prediction_decoder = prediction_decoder
        self.make_predict_function(force=True)
        self.make_train_function(force=True)
        self.make_test_function(force=True)

    def decode_predictions(self, predictions, data):
        box_pred, cls_pred = predictions["box"], predictions["classification"]
        # box_pred is on "center_yxhw" format, convert to target format.
        if isinstance(data, list) or isinstance(data, tuple):
            images, _ = data
        else:
            images = data
        image_shape = ops.shape(images)[1:]
        anchor_boxes = self.anchor_generator(images)
        anchor_boxes = ops.concatenate(list(anchor_boxes.values()), axis=0)
        box_pred = _decode_deltas_to_boxes(
            anchors=anchor_boxes,
            boxes_delta=box_pred,
            anchor_format=self.anchor_generator.bounding_box_format,
            box_format=self.target_bounding_box_format,
            variance=BOX_VARIANCE,
            image_shape=image_shape,
        )
        # box_pred is now in "self.target_bounding_box_format" format
        box_pred = convert_format(
            box_pred,
            source=self.target_bounding_box_format,
            target=self.prediction_decoder.bounding_box_format,
            image_shape=image_shape,
        )
        y_pred = self.prediction_decoder(
            box_pred, cls_pred, image_shape=image_shape
        )
        y_pred["boxes"] = convert_format(
            y_pred["boxes"],
            source=self.prediction_decoder.bounding_box_format,
            target=self.target_bounding_box_format,
            image_shape=image_shape,
        )
        return y_pred

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "ground_truth_bounding_box_format": self.ground_truth_bounding_box_format,
                "use_prediction_head_norm": self.use_prediction_head_norm,
                "target_bounding_box_format": self.target_bounding_box_format,
                "anchor_generator": keras.layers.serialize(
                    self.anchor_generator
                ),
                "label_encoder": keras.layers.serialize(self.label_encoder),
                "prediction_decoder": keras.layers.serialize(
                    self._prediction_decoder
                ),
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        if "label_encoder" in config and isinstance(
            config["label_encoder"], dict
        ):
            config["label_encoder"] = keras.layers.deserialize(
                config["label_encoder"]
            )

        if "anchor_generator" in config and isinstance(
            config["anchor_generator"], dict
        ):
            config["anchor_generator"] = keras.layers.deserialize(
                config["anchor_generator"]
            )

        if "prediction_decoder" in config and isinstance(
            config["prediction_decoder"], dict
        ):
            config["prediction_decoder"] = keras.layers.deserialize(
                config["prediction_decoder"]
            )

        return super().from_config(config)
