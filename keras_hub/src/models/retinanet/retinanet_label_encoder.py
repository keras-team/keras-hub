import keras
from keras import ops

from keras_hub.src.bounding_box.converters import _encode_box_to_deltas
from keras_hub.src.bounding_box.iou import compute_iou
from keras_hub.src.models.retinanet.anchor_generator import AnchorGenerator
from keras_hub.src.models.retinanet.box_matcher import BoxMatcher
from keras_hub.src.utils import tensor_utils


class RetinaNetLabelEncoder(keras.layers.Layer):
    """Transforms the raw labels into targets for training.

    RetinaNet is a single-stage object detection network that uses a feature
    pyramid network and focal loss. This class is crucial for preparing the
    ground truth data to match the network's anchor-based detection approach.

    This class generates targets for a batch of samples which consists of input
    images, bounding boxes for the objects present, and their class ids. It
    matches ground truth boxes to anchor boxes based on IoU (Intersection over
    Union) and encodes the box coordinates as offsets from the anchors.

    Targets are always represented in 'center_yxwh' format for numerical
    consistency during training, regardless of the input format.

    Args:
        bounding_box_format: str. The format of bounding boxes of input dataset.
            Refer TODO: Add link to Keras Core Docs.
        min_level: int. Minimum level of the output feature pyramid.
        max_level: int. Maximum level of the output feature pyramid.
        num_scales: int. Number of intermediate scales added on each level.
            For example, num_scales=2 adds one additional intermediate anchor
            scale [2^0, 2^0.5] on each level.
        aspect_ratios: List[float]. Aspect ratios of anchors added on
            each level. Each number indicates the ratio of width to height.
        anchor_size: float. Scale of size of the base anchor relative to the
            feature stride 2^level.
        positive_threshold:  float. the threshold to set an anchor to positive
            match to gt box. Values above it are positive matches.
            Defaults to `0.5`
        negative_threshold: float. the threshold to set an anchor to negative
            match to gt box. Values below it are negative matches.
            Defaults to `0.4`
        box_variance: List[float]. The scaling factors used to scale the
            bounding box targets.
            Defaults to `[0.1, 0.1, 0.2, 0.2]`.
        background_class: int. The class ID used for the background class,
            Defaults to `-1`.
        ignore_class: int. The class ID used for the ignore class,
            Defaults to `-2`.
        box_matcher_match_values: List[int]. Representing
            matched results (e.g. positive or negative or ignored match).
            `len(match_values)` must equal to `len(thresholds) + 1`.
            Defaults to `[-1, -2, -1]`.
        box_matcher_force_match_for_each_col: bool. If True, each column
            (ground truth box) will be matched to at least one row (anchor box).
            This means some columns may be matched to multiple rows while others
            may not be matched to any.
            Defaults to `False`.

    Note: `tf.RaggedTensor` are not supported.
    """

    def __init__(
        self,
        bounding_box_format,
        min_level,
        max_level,
        num_scales,
        aspect_ratios,
        anchor_size,
        positive_threshold=0.5,
        negative_threshold=0.4,
        box_variance=[0.1, 0.1, 0.2, 0.2],
        background_class=-1.0,
        ignore_class=-2.0,
        box_matcher_match_values=[-1, -2, 1],
        box_matcher_force_match_for_each_col=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bounding_box_format = bounding_box_format
        self.min_level = min_level
        self.max_level = max_level
        self.num_scales = num_scales
        self.aspect_ratios = aspect_ratios
        self.anchor_size = anchor_size
        self.positive_threshold = positive_threshold
        self.box_variance = box_variance
        self.negative_threshold = negative_threshold
        self.background_class = background_class
        self.ignore_class = ignore_class

        self.anchor_generator = AnchorGenerator(
            bounding_box_format=bounding_box_format,
            min_level=min_level,
            max_level=max_level,
            num_scales=num_scales,
            aspect_ratios=aspect_ratios,
            anchor_size=anchor_size,
        )

        self.box_matcher = BoxMatcher(
            thresholds=[negative_threshold, positive_threshold],
            match_values=box_matcher_match_values,
            force_match_for_each_col=box_matcher_force_match_for_each_col,
        )

    def build(self, images_shape, gt_boxes_shape, gt_classes_shape):
        self.built = True

    def call(self, images, gt_boxes, gt_classes):
        """Creates box and classification targets for a batch.

        Args:
            images: A Tensor. The  input images argument should be
                of shape `[B, H, W, C]` or `[B, C, H, W]`.
            gt_boxes: A Tensor with shape of `[B, num_boxes, 4]`.
            gt_labels: A Tensor with shape of `[B, num_boxes, num_classes]`

        Returns:
            box_targets: A Tensor of shape `[batch_size, num_anchors, 4]`
                containing the encoded box targets.
            class_targets: A Tensor of shape `[batch_size, num_anchors, 1]`
                containing the class targets for each anchor.
        """

        images_shape = ops.shape(images)
        if len(images_shape) != 4:
            raise ValueError(
                "`RetinaNetLabelEncoder`'s `call()` method does not "
                "support  unbatched inputs for the `images` argument. "
                f"Received `shape(images)={images_shape}`."
            )
        image_shape = images_shape[1:]

        if len(ops.shape(gt_classes)) == 2:
            gt_classes = ops.expand_dims(gt_classes, axis=-1)

        anchor_boxes = self.anchor_generator(images)
        anchor_boxes = ops.concatenate(list(anchor_boxes.values()), axis=0)

        box_targets, class_targets = self._encode_sample(
            gt_boxes, gt_classes, anchor_boxes, image_shape
        )
        box_targets = ops.reshape(
            box_targets, (-1, ops.shape(box_targets)[1], 4)
        )
        return box_targets, class_targets

    def _encode_sample(self, gt_boxes, gt_classes, anchor_boxes, image_shape):
        """Creates box and classification targets for a batched sample.

        Matches ground truth boxes to anchor boxes based on IOU.
        1. Calculates the pairwise IOU for the M `anchor_boxes` and N `gt_boxes`
            to get a `(M, N)` shaped matrix.
        2. The ground truth box with the maximum IOU in each row is assigned to
            the anchor box provided the IOU is greater than `match_iou`.
        3. If the maximum IOU in a row is less than `ignore_iou`, the anchor
            box is assigned with the background class.
        4. The remaining anchor boxes that do not have any class assigned are
            ignored during training.

        Args:
            gt_boxes: A Tensor of shape `[B, num_boxes, 4]`. Should be in
                `bounding_box_format`.
            gt_classes: A Tensor fo shape `[B, num_boxes, num_classes, 1]`.
            anchor_boxes: A Tensor with the shape `[total_anchors, 4]`
                representing all the anchor boxes for a given input image shape,
                where each anchor box is of the format `[x, y, width, height]`.
            image_shape: Tuple indicating the image shape `[H, W, C]`.

        Returns:
            Encoded boudning boxes in the format of `center_yxwh` and
            corresponding labels for each encoded bounding box.
        """

        iou_matrix = compute_iou(
            anchor_boxes,
            gt_boxes,
            bounding_box_format=self.bounding_box_format,
            image_shape=image_shape,
        )

        matched_gt_idx, matched_vals = self.box_matcher(iou_matrix)
        matched_vals = ops.expand_dims(matched_vals, axis=-1)
        positive_mask = ops.cast(ops.equal(matched_vals, 1), self.dtype)
        ignore_mask = ops.cast(ops.equal(matched_vals, -2), self.dtype)

        matched_gt_boxes = tensor_utils.target_gather(gt_boxes, matched_gt_idx)

        matched_gt_boxes = ops.reshape(
            matched_gt_boxes, (-1, ops.shape(matched_gt_boxes)[1], 4)
        )

        box_target = _encode_box_to_deltas(
            anchors=anchor_boxes,
            boxes=matched_gt_boxes,
            anchor_format=self.bounding_box_format,
            box_format=self.bounding_box_format,
            variance=self.box_variance,
            image_shape=image_shape,
        )

        matched_gt_cls_ids = tensor_utils.target_gather(
            gt_classes, matched_gt_idx
        )
        cls_target = ops.where(
            ops.not_equal(positive_mask, 1.0),
            self.background_class,
            matched_gt_cls_ids,
        )
        cls_target = ops.where(
            ops.equal(ignore_mask, 1.0), self.ignore_class, cls_target
        )
        label = ops.concatenate(
            [box_target, ops.cast(cls_target, box_target.dtype)], axis=-1
        )

        # In the case that a box in the corner of an image matches with an all
        # -1 box that is outside the image, we should assign the box to the
        # ignore class. There are rare cases where a -1 box can be matched,
        # resulting in a NaN during training. The unit test passing all -1s to
        # the label encoder ensures that we properly handle this edge-case.
        label = ops.where(
            ops.expand_dims(ops.any(ops.isnan(label), axis=-1), axis=-1),
            self.ignore_class,
            label,
        )

        return label[:, :, :4], label[:, :, 4]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "bounding_box_format": self.bounding_box_format,
                "min_level": self.min_level,
                "max_level": self.max_level,
                "num_scales": self.num_scales,
                "aspect_ratios": self.aspect_ratios,
                "anchor_size": self.anchor_size,
                "positive_threshold": self.positive_threshold,
                "box_variance": self.box_variance,
                "negative_threshold": self.negative_threshold,
                "background_class": self.background_class,
                "ignore_class": self.ignore_class,
            }
        )
        return config

    def compute_output_shape(
        self, images_shape, gt_boxes_shape, gt_classes_shape
    ):
        min_level = self.anchor_generator.min_level
        max_level = self.anchor_generator.max_level
        batch_size, image_H, image_W = images_shape[:-1]

        total_num_anchors = 0
        for i in range(min_level, max_level + 1):
            total_num_anchors += (
                (image_H // 2 ** (i))
                * (image_W // 2 ** (i))
                * self.anchor_generator.anchors_per_location
            )

        return (batch_size, total_num_anchors, 4), (
            batch_size,
            total_num_anchors,
        )
