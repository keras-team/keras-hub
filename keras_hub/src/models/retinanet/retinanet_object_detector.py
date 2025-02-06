import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.anchor_generator import AnchorGenerator
from keras_hub.src.layers.modeling.non_max_supression import NonMaxSuppression
from keras_hub.src.models.object_detector import ObjectDetector
from keras_hub.src.models.retinanet.prediction_head import PredictionHead
from keras_hub.src.models.retinanet.retinanet_backbone import RetinaNetBackbone
from keras_hub.src.models.retinanet.retinanet_label_encoder import (
    RetinaNetLabelEncoder,
)
from keras_hub.src.models.retinanet.retinanet_object_detector_preprocessor import (  # noqa: E501
    RetinaNetObjectDetectorPreprocessor,
)
from keras_hub.src.utils.tensor_utils import assert_bounding_box_support


@keras_hub_export("keras_hub.models.RetinaNetObjectDetector")
class RetinaNetObjectDetector(ObjectDetector):
    """RetinaNet object detector model.

    This class implements the RetinaNet object detection architecture.
    It consists of a feature extractor backbone, a feature pyramid network(FPN),
    and two prediction heads (for classification and bounding box regression).

    Args:
        backbone: `keras.Model`. A `keras.models.RetinaNetBackbone` class,
            defining the backbone network architecture. Provides feature maps
            for detection.
        anchor_generator: A `keras_hub.layers.AnchorGenerator` instance.
            Generates anchor boxes at different scales and aspect ratios
            across the image. If None, a default `AnchorGenerator` is
            created with the following parameters:
                - `bounding_box_format`:  Same as the model's
                   `bounding_box_format`.
                - `min_level`: The backbone's `min_level`.
                - `max_level`: The backbone's `max_level`.
                - `num_scales`: 3.
                - `aspect_ratios`: [0.5, 1.0, 2.0].
                - `anchor_size`: 4.0.
            You can create a custom `AnchorGenerator` by instantiating the
            `keras_hub.layers.AnchorGenerator` class and passing the desired
            arguments.
        num_classes: int. The number of object classes to be detected.
        bounding_box_format: str. Dataset bounding box format (e.g., "xyxy",
            "yxyx"). Defaults to `yxyx`.
        label_encoder: Optional. A `RetinaNetLabelEncoder` instance.  Encodes
            ground truth boxes and classes into training targets. It matches
            ground truth boxes to anchors based on IoU and encodes box
            coordinates as offsets. If `None`, a default encoder is created.
            See the `RetinaNetLabelEncoder` class for details. If None, a
            default encoder is created with standard parameters.
                - `anchor_generator`: Same as the model's.
                - `bounding_box_format`:  Same as the model's
                   `bounding_box_format`.
                - `positive_threshold`: 0.5
                - `negative_threshold`: 0.4
                - `encoding_format`: "center_xywh"
                - `box_variance`: [1.0, 1.0, 1.0, 1.0]
                - `background_class`: -1
                - `ignore_class`: -2
        use_prediction_head_norm: bool. Whether to use Group Normalization after
            the convolution layers in the prediction heads. Defaults to `False`.
        classification_head_prior_probability: float.  Prior probability for the
            classification head (used for focal loss). Defaults to 0.01.
        pre_logits_num_conv_layers: int. The number of convolutional layers in
            the head before the logits layer. These convolutional layers are
            applied before the final linear layer (logits) that produces the
            output predictions (bounding box regressions,
            classification scores).
        preprocessor: Optional. An instance of
            `RetinaNetObjectDetectorPreprocessor`or a custom preprocessor.
            Handles image preprocessing before feeding into the backbone.
        activation: Optional. The activation function to be used in the
            classification head. If None, sigmoid is used.
        dtype: Optional. The data type for the prediction heads. Defaults to the
            backbone's dtype policy.
        prediction_decoder: Optional. A `keras.layers.Layer` instance
            responsible for transforming RetinaNet predictions
            (box regressions and classifications) into final bounding boxes and
            classes with confidence scores. Defaults to a `NonMaxSuppression`
            instance.
    """

    backbone_cls = RetinaNetBackbone
    preprocessor_cls = RetinaNetObjectDetectorPreprocessor

    def __init__(
        self,
        backbone,
        num_classes,
        bounding_box_format="yxyx",
        anchor_generator=None,
        label_encoder=None,
        use_prediction_head_norm=False,
        classification_head_prior_probability=0.01,
        pre_logits_num_conv_layers=4,
        preprocessor=None,
        activation=None,
        dtype=None,
        prediction_decoder=None,
        **kwargs,
    ):
        # Check whether current version of keras support bounding box utils
        assert_bounding_box_support(self.__class__.__name__)

        # === Layers ===
        image_input = keras.layers.Input(backbone.image_shape, name="images")
        head_dtype = dtype or backbone.dtype_policy

        anchor_generator = anchor_generator or AnchorGenerator(
            bounding_box_format,
            min_level=backbone.min_level,
            max_level=backbone.max_level,
            num_scales=3,
            aspect_ratios=[0.5, 1.0, 2.0],
            anchor_size=4,
        )
        # As weights are ported from torch they use encoded format
        # as "center_xywh"
        label_encoder = label_encoder or RetinaNetLabelEncoder(
            anchor_generator,
            bounding_box_format=bounding_box_format,
            encoding_format="center_xywh",
        )

        box_head = PredictionHead(
            output_filters=anchor_generator.num_base_anchors * 4,
            num_conv_layers=pre_logits_num_conv_layers,
            num_filters=256,
            use_group_norm=use_prediction_head_norm,
            use_prior_probability=True,
            prior_probability=classification_head_prior_probability,
            dtype=head_dtype,
            name="box_head",
        )
        classification_head = PredictionHead(
            output_filters=anchor_generator.num_base_anchors * num_classes,
            num_conv_layers=pre_logits_num_conv_layers,
            num_filters=256,
            use_group_norm=use_prediction_head_norm,
            dtype=head_dtype,
            name="classification_head",
        )

        # === Functional Model ===
        feature_map = backbone(image_input)

        class_predictions = []
        box_predictions = []

        # Iterate through the feature pyramid levels (e.g., P3, P4, P5, P6, P7).
        for level in feature_map:
            box_predictions.append(
                keras.layers.Reshape((-1, 4), name=f"box_pred_{level}")(
                    box_head(feature_map[level])
                )
            )
            class_predictions.append(
                keras.layers.Reshape(
                    (-1, num_classes), name=f"cls_pred_{level}"
                )(classification_head(feature_map[level]))
            )

        # Concatenate predictions from all FPN levels.
        class_predictions = keras.layers.Concatenate(axis=1, name="cls_logits")(
            class_predictions
        )
        # box_pred is always in "center_xywh" delta-encoded no matter what
        # format you pass in.
        box_predictions = keras.layers.Concatenate(
            axis=1, name="bbox_regression"
        )(box_predictions)

        outputs = {
            "bbox_regression": box_predictions,
            "cls_logits": class_predictions,
        }

        super().__init__(
            inputs=image_input,
            outputs=outputs,
            **kwargs,
        )

        # === Config ===
        self.bounding_box_format = bounding_box_format
        self.use_prediction_head_norm = use_prediction_head_norm
        self.num_classes = num_classes
        self.backbone = backbone
        self.preprocessor = preprocessor
        self.activation = activation
        self.pre_logits_num_conv_layers = pre_logits_num_conv_layers
        self.box_head = box_head
        self.classification_head = classification_head
        self.anchor_generator = anchor_generator
        self.label_encoder = label_encoder
        self._prediction_decoder = prediction_decoder or NonMaxSuppression(
            from_logits=(activation != keras.activations.sigmoid),
            bounding_box_format=bounding_box_format,
        )

    def compute_loss(self, x, y, y_pred, sample_weight, **kwargs):
        _, height, width, _ = keras.ops.shape(x)
        y_for_label_encoder = keras.utils.bounding_boxes.convert_format(
            y,
            source=self.bounding_box_format,
            target=self.label_encoder.bounding_box_format,
            height=height,
            width=width,
        )

        boxes, labels = self.label_encoder(
            images=x,
            gt_boxes=y_for_label_encoder["boxes"],
            gt_classes=y_for_label_encoder["labels"],
        )

        box_pred = y_pred["bbox_regression"]
        cls_pred = y_pred["cls_logits"]

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
            ops.cast(labels, "int32"), self.num_classes, dtype="float32"
        )
        positive_mask = ops.cast(ops.greater(labels, -1.0), dtype="float32")
        normalizer = ops.sum(positive_mask)
        cls_weights = ops.cast(ops.not_equal(labels, -2.0), dtype="float32")
        cls_weights /= normalizer
        box_weights = positive_mask / normalizer

        y_true = {
            "bbox_regression": boxes,
            "cls_logits": cls_labels,
        }
        sample_weights = {
            "bbox_regression": box_weights,
            "cls_logits": cls_weights,
        }
        zero_weight = {
            "bbox_regression": ops.zeros_like(box_weights),
            "cls_logits": ops.zeros_like(cls_weights),
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
        box_pred = predictions["bbox_regression"]
        cls_pred = predictions["cls_logits"]
        # box_pred is on "center_yxhw" format, convert to target format.
        if isinstance(data, list) or isinstance(data, tuple):
            images, _ = data
        else:
            images = data
        height, width, channels = ops.shape(images)[1:]
        anchor_boxes = self.anchor_generator(images)
        anchor_boxes = ops.concatenate(list(anchor_boxes.values()), axis=0)
        box_pred = keras.utils.bounding_boxes.decode_deltas_to_boxes(
            anchors=anchor_boxes,
            boxes_delta=box_pred,
            encoded_format="center_xywh",
            anchor_format=self.anchor_generator.bounding_box_format,
            box_format=self.bounding_box_format,
            image_shape=(height, width, channels),
        )
        # box_pred is now in "self.bounding_box_format" format
        box_pred = keras.utils.bounding_boxes.convert_format(
            box_pred,
            source=self.bounding_box_format,
            target=self.prediction_decoder.bounding_box_format,
            height=height,
            width=width,
        )
        y_pred = self.prediction_decoder(box_pred, cls_pred, images=images)
        y_pred["boxes"] = keras.utils.bounding_boxes.convert_format(
            y_pred["boxes"],
            source=self.prediction_decoder.bounding_box_format,
            target=self.bounding_box_format,
            height=height,
            width=width,
        )
        return y_pred

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "use_prediction_head_norm": self.use_prediction_head_norm,
                "pre_logits_num_conv_layers": self.pre_logits_num_conv_layers,
                "bounding_box_format": self.bounding_box_format,
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
