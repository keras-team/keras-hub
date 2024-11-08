# import copy
from keras import ops
from keras import Model
from keras.layers import Conv2D, Activation, Reshape, Input, Concatenate
from keras.saving import serialize_keras_object, deserialize_keras_object

from keras_hub.src import bounding_box
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.object_detector import ObjectDetector
from keras_hub.src.models.yolo_v8.non_max_suppression import NonMaxSuppression
from keras_hub.src.models.yolo_v8.yolo_v8_backbone import YOLOV8Backbone
from keras_hub.src.models.yolo_v8.yolo_v8_label_encoder import (
    YOLOV8LabelEncoder,
)
from keras_hub.src.models.yolo_v8.yolo_v8_layers import apply_conv_bn
from keras_hub.src.models.yolo_v8.yolo_v8_layers import apply_CSP


def unwrap_data(data):
    if type(data) is dict:
        return data["images"], data["bounding_boxes"]
    else:
        return data


def get_anchors(image_shape, strides=[8, 16, 32], base_anchors=[0.5, 0.5]):
    """Gets anchor points for YOLOV8.

    YOLOV8 uses anchor points representing the center of proposed boxes, and
    matches ground truth boxes to anchors based on center points.

    Args:
        image_shape: tuple or list of two integers representing the height and
            width of input images, respectively.
        strides: tuple of list of integers, the size of the strides across the
            image size that should be used to create anchors.
        base_anchors: tuple or list of two integers representing offset from
            (0,0) to start creating the center of anchor boxes, relative to the
            stride. For example, using the default (0.5, 0.5) creates the first
            anchor box for each stride such that its center is half of a stride
            from the edge of the image.

    Returns:
        A tuple of anchor centerpoints and anchor strides. Multiplying the
        two together will yield the centerpoints in absolute x,y format.

    """
    base_anchors = ops.array(base_anchors, dtype="float32")

    all_anchors = []
    all_strides = []
    for stride in strides:
        hh_centers = ops.arange(0, image_shape[0], stride)
        ww_centers = ops.arange(0, image_shape[1], stride)
        ww_grid, hh_grid = ops.meshgrid(ww_centers, hh_centers)
        grid = ops.cast(
            ops.reshape(ops.stack([hh_grid, ww_grid], 2), [-1, 1, 2]),
            "float32",
        )
        anchors = (
            ops.expand_dims(
                base_anchors * ops.array([stride, stride], "float32"), 0
            ) + grid)
        anchors = ops.reshape(anchors, [-1, 2])
        all_anchors.append(anchors)
        all_strides.append(ops.repeat(stride, anchors.shape[0]))

    all_anchors = ops.cast(ops.concatenate(all_anchors, axis=0), "float32")
    all_strides = ops.cast(ops.concatenate(all_strides, axis=0), "float32")

    all_anchors = all_anchors / all_strides[:, None]

    # Swap the x and y coordinates of the anchors.
    all_anchors = ops.concatenate(
        [all_anchors[:, 1, None], all_anchors[:, 0, None]], axis=-1
    )
    return all_anchors, all_strides


def upsample(x, size=2):
    return ops.repeat(ops.repeat(x, size, axis=1), size, axis=2)


def merge_upper_level(lower_level, upper_level, depth, name):
    x = upsample(upper_level)
    x = ops.concatenate([x, lower_level], axis=-1)
    channels = lower_level.shape[-1]
    return apply_CSP(x, channels, depth, False, 0.5, "swish", name)


def merge_lower_level(lower_level, upper_level, name):
    x = apply_conv_bn(lower_level, lower_level.shape[-1], 3, 2, "swish", name)
    x = ops.concatenate([x, upper_level], axis=-1)
    channels = upper_level.shape[-1]
    x = apply_CSP(x, channels, 2, False, 0.5, "swish", f"{name}_block")
    return x


def apply_path_aggregation_FPN(features, depth=3, name="fpn"):
    p3, p4, p5 = features
    p4p5 = merge_upper_level(p4, p5, depth, f"{name}_p4p5")
    p3p4p5 = merge_upper_level(p3, p4p5, depth, f"{name}_p3p4p5")
    p3p4p5_d1 = merge_lower_level(p3p4p5, p4p5, f"{name}_p3p4p5_downsample1")
    p3p4p5_d2 = merge_lower_level(p3p4p5_d1, p5, f"{name}_p3p4p5_downsample2")
    return [p3p4p5, p3p4p5_d1, p3p4p5_d2]


def get_boxes_channels(x, default=64):
    # If the input has a large resolution e.g. P3 has >256 channels,
    # additional channels are used in intermediate conv layers.
    return max(default, x.shape[-1] // 4)


def get_class_channels(x, num_classes):
    # We use at least num_classes channels for intermediate conv layer for
    # class predictions. In most cases, the P3 input has more channels than the
    # number of classes, so we preserve those channels until the final layer.
    return max(num_classes, x.shape[-1])


def apply_boxes_block(x, boxes_channels, name):
    x = apply_conv_bn(x, boxes_channels, 3, 1, "swish", f"{name}_box_1")
    x = apply_conv_bn(x, boxes_channels, 3, 1, "swish", f"{name}_box_2")
    BOX_REGRESSION_CHANNELS = 64   # 16 values per corner offset from center.
    x = Conv2D(BOX_REGRESSION_CHANNELS, 1, name=f"{name}_box_3_conv")(x)
    return x


def apply_class_block(x, class_channels, num_classes, name):
    x = apply_conv_bn(x, class_channels, 3, 1, "swish", f"{name}_class_1")
    x = apply_conv_bn(x, class_channels, 3, 1, "swish", f"{name}_class_2")
    x = Conv2D(num_classes, 1, name=f"{name}_class_3_conv")(x)
    x = Activation("sigmoid", name=f"{name}_classifier")(x)
    return x


def apply_branch_head(x, boxes_channels, class_channels, num_classes, name):
    boxes_predictions = apply_boxes_block(x, boxes_channels, name)
    class_predictions = apply_class_block(x, class_channels, num_classes, name)
    branch = ops.concatenate([boxes_predictions, class_predictions], axis=-1)
    branch_shape = [-1, branch.shape[-1]]
    branch = Reshape(branch_shape, name=f"{name}_output_reshape")(branch)
    return branch


def apply_detection_head(inputs, num_classes, name="yolo_v8_head"):
    boxes_channels = get_boxes_channels(inputs[0])
    class_channels = get_class_channels(inputs[0], num_classes)
    branch_args = (boxes_channels, class_channels, num_classes)
    outputs = []
    for feature_arg, feature in enumerate(inputs):
        feature_name = f"{name}_{feature_arg + 1}"
        outputs.append(apply_branch_head(feature, *branch_args, feature_name))

    x = ops.concatenate(outputs, axis=1)
    x = Activation("linear", dtype="float32", name="box_outputs")(x)
    BOX_REGRESSION_CHANNELS = 64
    boxes_tensor = x[:, :, :BOX_REGRESSION_CHANNELS]
    class_tensor = x[:, :, BOX_REGRESSION_CHANNELS:]
    return {"boxes": boxes_tensor, "classes": class_tensor}


def add_no_op_for_pretty_print(x, name):
    return Concatenate(axis=1, name=name)([x])


def get_feature_extractor(model, layer_names, output_keys=None):
    if not output_keys:
        output_keys = layer_names
    items = zip(output_keys, layer_names)
    outputs = {key: model.get_layer(name).output for key, name in items}
    return Model(inputs=model.inputs, outputs=outputs)


def get_backbone_pyramid_layer_names(backbone, level_names):
    """Gets layer names from the provided pyramid levels inside backbone.

    Args:
        backbone: Keras backbone model with the field "pyramid_level_inputs".
        level_names: list of strings indicating the level names.

    Returns:
        List of layer strings indicating the layer names of each level.
    """
    layer_names = []
    for level_name in level_names:
        layer_names.append(backbone.pyramid_level_inputs[level_name])
    return layer_names


def build_feature_extractor(backbone, level_names):
    """Builds feature extractor directly from the level names

    Args:
        backbone: Keras backbone model with the field "pyramid_level_inputs".
        level_names: list of strings indicating the level names.

    Returns:
        Keras Model with level names as outputs.
    """
    layer_names = get_backbone_pyramid_layer_names(backbone, level_names)
    items = zip(level_names, layer_names)
    outputs = {key: backbone.get_layer(name).output for key, name in items}
    return Model(inputs=backbone.inputs, outputs=outputs)


def extend_branches(inputs, extractor, FPN_depth):
    """Extends extractor model with a feature pyramid network.

    Args:
        inputs: tensor, with image input.
        extractor: Keras Model with level names as outputs.
        FPN_depth: integer representing the feature pyramid depth.

    Returns:
        List of extended branch tensors.
    """
    features = list(extractor(inputs).values())
    branches = apply_path_aggregation_FPN(features, FPN_depth, name="pa_fpn")
    return branches


def extend_backbone(backbone, level_names, FPN_depth):
    """Extends backbone levels with a feature pyramid network.

    Args:
        backbone: Keras backbone model with the field "pyramid_level_inputs".
        level_names: list of strings indicating the level names.
        trainable: boolean indicating if backbone should be optimized.
        FPN_depth: integer representing the feature pyramid depth.

    Return:
        Tuple with input image tensor, and list of extended branch tensors.
    """
    feature_extractor = build_feature_extractor(backbone, level_names)
    image = Input(feature_extractor.input_shape[1:])
    branches = extend_branches(image, feature_extractor, FPN_depth)
    return image, branches


def decode_regression_to_boxes(preds):
    """Decodes the results of the YOLOV8Detector forward-pass into boxes.

    Returns left / top / right / bottom predictions with respect to anchor
    points.

    Each coordinate is encoded with 16 predicted values. Those predictions are
    softmaxed and multiplied by [0..15] to make predictions. The resulting
    predictions are relative to the stride of an anchor box
    (and correspondingly relative to the scale of the feature map from which
    the predictions came).
    """
    BOX_REGRESSION_CHANNELS = 64
    preds_bbox = Reshape((-1, 4, BOX_REGRESSION_CHANNELS // 4))(preds)
    preds_bbox = ops.nn.softmax(preds_bbox, axis=-1) * ops.arange(
        BOX_REGRESSION_CHANNELS // 4, dtype="float32")
    return ops.sum(preds_bbox, axis=-1)


def dist2bbox(distance, anchor_points):
    """Decodes distance predictions into xyxy boxes.

    Input left / top / right / bottom predictions are transformed into xyxy box
    predictions based on anchor points.

    The resulting xyxy predictions must be scaled by the stride of their
    corresponding anchor points to yield an absolute xyxy box.
    """
    left_top, right_bottom = ops.split(distance, 2, axis=-1)
    x1y1 = anchor_points - left_top
    x2y2 = anchor_points + right_bottom
    return ops.concatenate((x1y1, x2y2), axis=-1)  # xyxy bbox


@keras_hub_export(["keras_hub.models.YOLOV8ObjectDetector"])
class YOLOV8ObjectDetector(ObjectDetector):
    """Implements the YOLOV8 architecture for object detection.

    Args:
        backbone: `keras.Model`, must implement the `pyramid_level_inputs`
            property with keys "P3", "P4", and "P5" and layer names as values.
            A sensible backbone to use is the `keras_hub.models.YOLOV8Backbone`.
        num_classes: integer, the number of classes in your dataset excluding the
            background class. Classes should be represented by integers in the
            range [0, num_classes).
        bounding_box_format: string, the format of bounding boxes of input dataset.
            Refer
            [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/)
            for more details on supported bounding box formats.
        fpn_depth: integer, a specification of the depth of the CSP blocks in
            the Feature Pyramid Network. This is usually 1, 2, or 3, depending
            on the size of your YOLOV8Detector model. We recommend using 3 for
            "yolo_v8_l_backbone" and "yolo_v8_xl_backbone". Defaults to 2.
        label_encoder: (Optional)  A `YOLOV8LabelEncoder` that is
            responsible for transforming input boxes into trainable labels for
            YOLOV8Detector. If not provided, a default is provided.
        prediction_decoder: (Optional)  A `keras.layers.Layer` that is
            responsible for transforming YOLOV8 predictions into usable
            bounding boxes. If not provided, a default is provided. The
            default `prediction_decoder` layer is a
            `keras_hub.layers.MultiClassNonMaxSuppression` layer, which uses
            a Non-Max Suppression for box pruning.

    Example:
    ```python
    images = tf.ones(shape=(1, 512, 512, 3))
    labels = {
        "boxes": tf.constant([
            [
                [0, 0, 100, 100],
                [100, 100, 200, 200],
                [300, 300, 100, 100],
            ]
        ], dtype=tf.float32),
        "classes": tf.constant([[1, 1, 1]], dtype=tf.int64),
    }

    model = keras_hub.models.YOLOV8Detector(
        num_classes=20,
        bounding_box_format="xywh",
        backbone=keras_hub.models.YOLOV8Backbone.from_preset(
            "yolo_v8_m_backbone_coco"
        ),
        fpn_depth=2
    )

    # Evaluate model without box decoding and NMS
    model(images)

    # Prediction with box decoding and NMS
    model.predict(images)

    # Train model
    model.compile(
        classification_loss='binary_crossentropy',
        box_loss='ciou',
        optimizer=tf.optimizers.SGD(global_clipnorm=10.0),
        jit_compile=False,
    )
    model.fit(images, labels)
    ```
    """  # noqa: E501

    backbone_cls = YOLOV8Backbone

    def __init__(self, backbone, num_classes, bounding_box_format, fpn_depth=2,
                 preprocessor=None, label_encoder=None,
                 prediction_decoder=None, **kwargs):
        level_names = ["P3", "P4", "P5"]
        image, branches = extend_backbone(backbone, level_names, fpn_depth)
        head = apply_detection_head(branches, num_classes)
        boxes_tensor = add_no_op_for_pretty_print(head["boxes"], "box")
        class_tensor = add_no_op_for_pretty_print(head["classes"], "class")
        outputs = {"boxes": boxes_tensor, "classes": class_tensor}
        super().__init__(inputs=image, outputs=outputs, **kwargs)

        self.bounding_box_format = bounding_box_format
        self._prediction_decoder = prediction_decoder or NonMaxSuppression(
            bounding_box_format=bounding_box_format,
            from_logits=False,
            confidence_threshold=0.2,
            iou_threshold=0.7,
        )
        self.backbone = backbone
        self.fpn_depth = fpn_depth
        self.num_classes = num_classes
        self.label_encoder = label_encoder or YOLOV8LabelEncoder(
            num_classes=num_classes)

    def train_step(self, *args):
        # This is done for tf.data pipelines that don't unwrap dictionaries.
        data = args[-1]
        args = args[:-1]
        x, y = unwrap_data(data)
        return super().train_step(*args, (x, y))

    def test_step(self, *args):
        # This is done for tf.data pipelines that don't unwrap dictionaries.
        data = args[-1]
        args = args[:-1]
        x, y = unwrap_data(data)
        return super().test_step(*args, (x, y))

    def compute_loss(self, x, y, y_pred, sample_weight=None, **kwargs):
        box_pred, cls_pred = y_pred["boxes"], y_pred["classes"]

        pred_boxes = decode_regression_to_boxes(box_pred)
        pred_scores = cls_pred

        anchor_points, stride_tensor = get_anchors(image_shape=x.shape[1:])
        stride_tensor = ops.expand_dims(stride_tensor, axis=-1)

        gt_labels = y["classes"]
        mask_gt = ops.all(y["boxes"] > -1.0, axis=-1, keepdims=True)
        gt_bboxes = bounding_box.convert_format(
            y["boxes"],
            source=self.bounding_box_format,
            target="xyxy",
            images=x,
        )

        pred_bboxes = dist2bbox(pred_boxes, anchor_points)

        target_bboxes, target_scores, fg_mask = self.label_encoder(
            pred_scores,
            ops.cast(pred_bboxes * stride_tensor, gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_bboxes /= stride_tensor
        target_scores_sum = ops.maximum(ops.sum(target_scores), 1)
        box_weight = ops.expand_dims(
            ops.sum(target_scores, axis=-1) * fg_mask,
            axis=-1,
        )

        y_true = {
            "box": target_bboxes * fg_mask[..., None],
            "class": target_scores,
        }
        y_pred = {
            "box": pred_bboxes * fg_mask[..., None],
            "class": pred_scores,
        }
        sample_weights = {
            "box": self.box_loss_weight * box_weight / target_scores_sum,
            "class": self.classification_loss_weight / target_scores_sum,
        }

        return super().compute_loss(
            x=x, y=y_true, y_pred=y_pred,
            sample_weight=sample_weights, **kwargs
        )

    def decode_predictions(self, pred, images):
        boxes = pred["boxes"]
        scores = pred["classes"]

        boxes = decode_regression_to_boxes(boxes)

        anchor_points, stride_tensor = get_anchors(
            image_shape=images.shape[1:])
        stride_tensor = ops.expand_dims(stride_tensor, axis=-1)

        box_preds = dist2bbox(boxes, anchor_points) * stride_tensor
        box_preds = bounding_box.convert_format(
            box_preds,
            source="xyxy",
            target=self.bounding_box_format,
            images=images,
        )

        return self.prediction_decoder(box_preds, scores)

    def predict_step(self, *args):
        outputs = super().predict_step(*args)
        if isinstance(outputs, tuple):
            return self.decode_predictions(outputs[0], args[-1]), outputs[1]
        else:
            return self.decode_predictions(outputs, args[-1])

    @property
    def prediction_decoder(self):
        return self._prediction_decoder

    @prediction_decoder.setter
    def prediction_decoder(self, prediction_decoder):
        if prediction_decoder.bounding_box_format != self.bounding_box_format:
            raise ValueError(
                "Expected `prediction_decoder` and YOLOV8Detector to "
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

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "bounding_box_format": self.bounding_box_format,
            "fpn_depth": self.fpn_depth,
            "backbone": serialize_keras_object(self.backbone),
            "label_encoder": serialize_keras_object(
                self.label_encoder
            ),
            "prediction_decoder": serialize_keras_object(
                self._prediction_decoder
            ),
        }

    @classmethod
    def from_config(cls, config):
        config["backbone"] = deserialize_keras_object(
            config["backbone"]
        )
        label_encoder = config.get("label_encoder")
        if label_encoder is not None and isinstance(label_encoder, dict):
            config["label_encoder"] = deserialize_keras_object(
                label_encoder
            )
        prediction_decoder = config.get("prediction_decoder")
        if prediction_decoder is not None and isinstance(
            prediction_decoder, dict
        ):
            config["prediction_decoder"] = (
                deserialize_keras_object(prediction_decoder)
            )
        return cls(**config)
