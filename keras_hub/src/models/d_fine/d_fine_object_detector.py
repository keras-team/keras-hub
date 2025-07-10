import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.non_max_supression import NonMaxSuppression
from keras_hub.src.models.d_fine.d_fine_backbone import DFineBackbone
from keras_hub.src.models.d_fine.d_fine_object_detector_preprocessor import (
    DFineObjectDetectorPreprocessor,
)
from keras_hub.src.models.d_fine.d_fine_utils import center_to_corners_format
from keras_hub.src.models.d_fine.d_fine_utils import weighting_function
from keras_hub.src.models.object_detector import ObjectDetector
from keras_hub.src.utils.tensor_utils import assert_bounding_box_support


@keras_hub_export("keras_hub.models.DFineObjectDetector")
class DFineObjectDetector(ObjectDetector):
    """D-FINE Object Detector model.

    This class wraps the `DFineBackbone` and adds the final prediction and loss
    computation logic for end-to-end object detection. It is responsible for:
    1.  Defining the functional model that connects the `DFineBackbone` to the
        input layers.
    2.  Implementing the `compute_loss` method, which uses a Hungarian matcher
        to assign predictions to ground truth targets and calculates a weighted
        sum of multiple loss components (classification, bounding box, etc.).
    3.  Post-processing the raw outputs from the backbone into final, decoded
        predictions (boxes, labels, confidence scores) during inference.

    Args:
        backbone: A `keras_hub.models.Backbone` instance, specifically a
            `DFineBackbone`, serving as the feature extractor for the object
            detector.
        num_classes: An integer representing the number of object classes to
            detect.
        bounding_box_format: A string specifying the format of the bounding
            boxes. Default is `"yxyx"`. Must be a supported format (e.g.,
            `"yxyx"`, `"xyxy"`).
        preprocessor: Optional. An instance of `DFineObjectDetectorPreprocessor`
            for input data preprocessing.
        matcher_class_cost: A float representing the cost for class mismatch in
            the Hungarian matcher. Default is `2.0`.
        matcher_bbox_cost: A float representing the cost for bounding box
            mismatch in the Hungarian matcher. Default is `5.0`.
        matcher_giou_cost: A float representing the cost for generalized IoU
            mismatch in the Hungarian matcher. Default is `2.0`.
        use_focal_loss: A boolean indicating whether to use focal loss for
            classification. Default is `True`.
        matcher_alpha: A float parameter for the focal loss alpha. Default is
            `0.25`.
        matcher_gamma: A float parameter for the focal loss gamma. Default is
            `2.0`.
        weight_loss_vfl: Weight for the classification loss. Default is `1.0`.
        weight_loss_bbox: Weight for the bounding box regression loss. Default
            is `5.0`.
        weight_loss_giou: Weight for the generalized IoU loss. Default is `2.0`.
        weight_loss_fgl: Weight for the focal grid loss. Default is `0.15`.
        weight_loss_ddf: Weight for the DDF loss. Default is `1.5`.

    Examples:

    **Creating a DFineObjectDetector without labels:**

    ```python
    import numpy as np
    from keras_hub.src.models.d_fine.d_fine_backbone import DFineBackbone
    from keras_hub.src.models.d_fine.d_fine_object_detector import (
        DFineObjectDetector
    )

    # Initialize the backbone without labels.
    backbone = DFineBackbone(
        stackwise_stage_filters=[
            [16, 16, 64, 1, 3, 3],
            [64, 32, 256, 1, 3, 3],
            [256, 64, 512, 2, 3, 5],
            [512, 128, 1024, 1, 3, 5],
        ],
        apply_downsample=[False, True, True, True],
        use_lightweight_conv_block=[False, False, True, True],
        image_shape=(256, 256, 3),
        out_features=["stage3", "stage4"],
        num_denoising=100,
        num_queries=300,
        hidden_dim=128,
        encoder_layers=1,
        decoder_layers=3,
    )

    # Create the detector.
    detector = DFineObjectDetector(
        backbone=backbone,
        num_classes=80,
        bounding_box_format="yxyx",
    )
    ```

    **Creating a DFineObjectDetector with labels for the backbone:**

    ```python
    import numpy as np
    from keras_hub.src.models.d_fine.d_fine_backbone import DFineBackbone
    from keras_hub.src.models.d_fine.d_fine_object_detector import (
        DFineObjectDetector
    )

    # Define labels for the backbone.
    labels = [
        {
            "boxes": np.array([[0.5, 0.5, 0.2, 0.2], [0.4, 0.4, 0.1, 0.1]]),
            "labels": np.array([1, 10])
        },
        {"boxes": np.array([[0.6, 0.6, 0.3, 0.3]]), "labels": np.array([20])},
    ]

    # Backbone is initialized with labels.
    backbone = DFineBackbone(
        stackwise_stage_filters=[
            [16, 16, 64, 1, 3, 3],
            [64, 32, 256, 1, 3, 3],
            [256, 64, 512, 2, 3, 5],
            [512, 128, 1024, 1, 3, 5],
        ],
        apply_downsample=[False, True, True, True],
        use_lightweight_conv_block=[False, False, True, True],
        image_shape=(256, 256, 3),
        out_features=["stage3", "stage4"],
        num_denoising=100,
        num_queries=300,
        hidden_dim=128,
        encoder_layers=1,
        decoder_layers=3,
        labels=labels,
        box_noise_scale=1.0,
        label_noise_ratio=0.5,
    )

    # Create the detector.
    detector = DFineObjectDetector(
        backbone=backbone,
        num_classes=80,
        bounding_box_format="yxyx",
    )
    ```

    **Using the detector for training:**

    ```python
    import numpy as np
    from keras_hub.src.models.d_fine.d_fine_backbone import DFineBackbone
    from keras_hub.src.models.d_fine.d_fine_object_detector import (
        DFineObjectDetector
    )

    # Initialize backbone and detector.
    backbone = DFineBackbone(
        stackwise_stage_filters=[
            [16, 16, 64, 1, 3, 3],
            [64, 32, 256, 1, 3, 3],
            [256, 64, 512, 2, 3, 5],
            [512, 128, 1024, 1, 3, 5],
        ],
        apply_downsample=[False, True, True, True],
        use_lightweight_conv_block=[False, False, True, True],
        image_shape=(256, 256, 3),
        out_features=["stage3", "stage4"],
        num_denoising=100,
        num_queries=300,
        hidden_dim=128,
        encoder_layers=1,
        decoder_layers=3,
    )
    detector = DFineObjectDetector(
        backbone=backbone,
        num_classes=80,
        bounding_box_format="yxyx",
    )

    # Sample training data.
    images = np.random.uniform(
        low=0, high=255, size=(2, 256, 256, 3)
    ).astype("float32")
    bounding_boxes = {
        "boxes": np.array([
            [[10.0, 20.0, 20.0, 30.0], [20.0, 30.0, 30.0, 40.0]],
            [[15.0, 25.0, 25.0, 35.0]]
        ]),
        "labels": np.array([[0, 2], [1]])
    }

    # Compile the model.
    detector.compile(
        optimizer="adam",
        loss=detector.compute_loss,
    )

    # Train the model.
    detector.fit(x=images, y=bounding_boxes, epochs=1, batch_size=1)
    ```

    **Making predictions:**

    ```python
    import numpy as np
    from keras_hub.src.models.d_fine.d_fine_backbone import DFineBackbone
    from keras_hub.src.models.d_fine.d_fine_object_detector import (
        DFineObjectDetector
    )

    # Initialize backbone and detector.
    backbone = DFineBackbone(
        stackwise_stage_filters=[
            [16, 16, 64, 1, 3, 3],
            [64, 32, 256, 1, 3, 3],
            [256, 64, 512, 2, 3, 5],
            [512, 128, 1024, 1, 3, 5],
        ],
        apply_downsample=[False, True, True, True],
        use_lightweight_conv_block=[False, False, True, True],
        image_shape=(256, 256, 3),
        out_features=["stage3", "stage4"],
        num_denoising=100,
        num_queries=300,
        hidden_dim=128,
        encoder_layers=1,
        decoder_layers=3,
    )
    detector = DFineObjectDetector(
        backbone=backbone,
        num_classes=80,
        bounding_box_format="yxyx",
    )

    # Sample test image.
    test_image = np.random.uniform(
        low=0, high=255, size=(1, 256, 256, 3)
    ).astype("float32")

    # Make predictions.
    predictions = detector.predict(test_image)

    # Access predictions.
    boxes = predictions["boxes"]                    # Shape: (1, 100, 4)
    labels = predictions["labels"]                  # Shape: (1, 100)
    confidence = predictions["confidence"]          # Shape: (1, 100)
    num_detections = predictions["num_detections"]  # Shape: (1,)
    ```
    """

    backbone_cls = DFineBackbone
    preprocessor_cls = DFineObjectDetectorPreprocessor

    def __init__(
        self,
        backbone,
        num_classes,
        bounding_box_format="yxyx",
        preprocessor=None,
        matcher_class_cost=2.0,
        matcher_bbox_cost=5.0,
        matcher_giou_cost=2.0,
        use_focal_loss=True,
        matcher_alpha=0.25,
        matcher_gamma=2.0,
        weight_loss_vfl=1.0,
        weight_loss_bbox=5.0,
        weight_loss_giou=2.0,
        weight_loss_fgl=0.15,
        weight_loss_ddf=1.5,
        prediction_decoder=None,
        activation=None,
        **kwargs,
    ):
        assert_bounding_box_support(self.__class__.__name__)

        # === Layers ===
        image_input = keras.layers.Input(
            shape=backbone.image_shape, name="images"
        )
        outputs = backbone(image_input)
        intermediate_logits = outputs["intermediate_logits"]
        intermediate_reference_points = outputs["intermediate_reference_points"]
        intermediate_predicted_corners = outputs[
            "intermediate_predicted_corners"
        ]
        initial_reference_points = outputs["initial_reference_points"]
        logits = intermediate_logits[:, -1, :, :]
        pred_boxes = intermediate_reference_points[:, -1, :, :]
        model_outputs = {
            "logits": logits,
            "pred_boxes": pred_boxes,
            "intermediate_logits": intermediate_logits,
            "intermediate_reference_points": intermediate_reference_points,
            "intermediate_predicted_corners": intermediate_predicted_corners,
            "initial_reference_points": initial_reference_points,
            "enc_topk_logits": outputs["enc_topk_logits"],
            "enc_topk_bboxes": outputs["enc_topk_bboxes"],
        }
        if "dn_num_group" in outputs:
            model_outputs["dn_positive_idx"] = outputs["dn_positive_idx"]
            model_outputs["dn_num_group"] = outputs["dn_num_group"]
            model_outputs["dn_num_split"] = outputs["dn_num_split"]

        # === Functional Model ===
        super().__init__(
            inputs=image_input,
            outputs=model_outputs,
            **kwargs,
        )

        # === Config ===
        self.backbone = backbone
        self.num_classes = num_classes
        self.bounding_box_format = bounding_box_format
        self.preprocessor = preprocessor
        self.matcher_class_cost = matcher_class_cost
        self.matcher_bbox_cost = matcher_bbox_cost
        self.matcher_giou_cost = matcher_giou_cost
        self.use_focal_loss = use_focal_loss
        self.matcher_alpha = matcher_alpha
        self.matcher_gamma = matcher_gamma
        self.weight_dict = {
            "loss_vfl": weight_loss_vfl,
            "loss_bbox": weight_loss_bbox,
            "loss_giou": weight_loss_giou,
            "loss_fgl": weight_loss_fgl,
            "loss_ddf": weight_loss_ddf,
        }
        self.activation = activation
        self._prediction_decoder = prediction_decoder or NonMaxSuppression(
            from_logits=(self.activation != keras.activations.sigmoid),
            bounding_box_format=self.bounding_box_format,
        )

    def compute_loss(self, x, y, y_pred, sample_weight, **kwargs):
        gt_boxes = y["boxes"]
        gt_labels = y["labels"]
        batch_size = keras.ops.shape(gt_labels)[0]
        max_objects = keras.ops.shape(gt_labels)[1]
        batch_idx = keras.ops.arange(batch_size)
        object_idx = keras.ops.arange(max_objects)
        batch_indices_all = keras.ops.expand_dims(batch_idx, axis=1)
        object_indices_all = keras.ops.expand_dims(object_idx, axis=0)
        batch_indices_all = keras.ops.broadcast_to(
            batch_indices_all, (batch_size, max_objects)
        )
        object_indices_all = keras.ops.broadcast_to(
            object_indices_all, (batch_size, max_objects)
        )
        batch_indices = keras.ops.reshape(batch_indices_all, [-1])
        object_indices = keras.ops.reshape(object_indices_all, [-1])
        flat_labels = keras.ops.reshape(gt_labels, [-1])
        flat_boxes = keras.ops.reshape(gt_boxes, [-1, 4])
        linear_indices = (
            batch_indices * keras.ops.shape(gt_labels)[1] + object_indices
        )
        labels_for_item = keras.ops.take(flat_labels, linear_indices, axis=0)
        boxes_for_item = keras.ops.take(flat_boxes, linear_indices, axis=0)
        targets = {"labels": labels_for_item, "boxes": boxes_for_item}

        logits = y_pred["logits"]
        pred_boxes = y_pred["pred_boxes"]
        predicted_corners = y_pred["intermediate_predicted_corners"]
        initial_reference_points = y_pred["initial_reference_points"]
        auxiliary_outputs = {
            "intermediate_logits": y_pred["intermediate_logits"][:, :-1, :, :],
            "intermediate_reference_points": y_pred[
                "intermediate_reference_points"
            ][:, :-1, :, :],
            "enc_topk_logits": y_pred["enc_topk_logits"],
            "enc_topk_bboxes": y_pred["enc_topk_bboxes"],
            "predicted_corners": predicted_corners[:, :-1, :, :],
            "initial_reference_points": initial_reference_points[:, :-1, :, :],
        }
        if "dn_num_group" in y_pred:
            denoising_meta_values = {
                "dn_positive_idx": y_pred["dn_positive_idx"],
                "dn_num_group": y_pred["dn_num_group"],
                "dn_num_split": y_pred["dn_num_split"],
            }
        else:
            denoising_meta_values = None
        auxiliary_outputs["denoising_meta_values"] = denoising_meta_values
        outputs_class = keras.ops.concatenate(
            [
                auxiliary_outputs["intermediate_logits"],
                keras.ops.expand_dims(logits, 1),
            ],
            axis=1,
        )
        outputs_coord = keras.ops.concatenate(
            [
                auxiliary_outputs["intermediate_reference_points"],
                keras.ops.expand_dims(pred_boxes, 1),
            ],
            axis=1,
        )
        enc_topk_logits = auxiliary_outputs["enc_topk_logits"]
        enc_topk_bboxes = auxiliary_outputs["enc_topk_bboxes"]

        denoising_meta_values = auxiliary_outputs["denoising_meta_values"]
        if denoising_meta_values is not None:
            num_denoising = self.backbone.num_denoising
            main_queries_start = 2 * num_denoising
        else:
            main_queries_start = 0
        outputs_without_aux = {
            "logits": logits[:, main_queries_start:],
            "pred_boxes": keras.ops.clip(
                pred_boxes[:, main_queries_start:], 0, 1
            ),
        }
        indices = self.hungarian_matcher(outputs_without_aux, [targets])
        num_boxes = keras.ops.shape(labels_for_item)[0]
        num_boxes = keras.ops.convert_to_tensor(num_boxes, dtype="float32")
        num_boxes = keras.ops.maximum(num_boxes, 1.0)
        losses = {}
        vfl_loss = self.compute_vfl_loss(
            outputs_without_aux, [targets], indices, num_boxes
        )
        losses.update(
            {
                k: vfl_loss[k] * self.weight_dict[k]
                for k in vfl_loss
                if k in self.weight_dict
            }
        )
        box_losses = self.compute_box_losses(
            outputs_without_aux, [targets], indices, num_boxes
        )
        losses.update(
            {
                k: box_losses[k] * self.weight_dict[k]
                for k in box_losses
                if k in self.weight_dict
            }
        )
        local_losses = self.compute_local_losses(
            {
                **outputs_without_aux,
                "pred_corners": predicted_corners[:, -1, main_queries_start:],
                "ref_points": initial_reference_points[
                    :, -1, main_queries_start:
                ],
                "teacher_corners": keras.ops.zeros_like(
                    predicted_corners[:, -1, main_queries_start:]
                ),
                "teacher_logits": keras.ops.zeros_like(
                    logits[:, main_queries_start:]
                ),
            },
            [targets],
            indices,
            num_boxes,
            compute_ddf=False,
        )
        losses.update(
            {
                k: local_losses[k] * self.weight_dict[k]
                for k in local_losses
                if k in self.weight_dict
            }
        )

        auxiliary_outputs_list = [
            {
                "logits": outputs_class[:, i, main_queries_start:, :],
                "pred_boxes": keras.ops.clip(
                    outputs_coord[:, i, main_queries_start:, :], 0, 1
                ),
                "pred_corners": predicted_corners[:, i, main_queries_start:, :],
                "ref_points": initial_reference_points[
                    :, i, main_queries_start:, :
                ],
                "teacher_corners": predicted_corners[
                    :, -1, main_queries_start:, :
                ]
                if i < self.backbone.decoder_layers - 1
                else None,
                "teacher_logits": outputs_class[:, -1, main_queries_start:, :]
                if i < self.backbone.decoder_layers - 1
                else None,
            }
            for i in range(self.backbone.decoder_layers - 1)
        ]
        for i, aux_output in enumerate(auxiliary_outputs_list):
            aux_indices = self.hungarian_matcher(aux_output, [targets])
            aux_vfl_loss = self.compute_vfl_loss(
                aux_output, [targets], aux_indices, num_boxes
            )
            aux_box_losses = self.compute_box_losses(
                aux_output, [targets], aux_indices, num_boxes
            )
            aux_local_losses = self.compute_local_losses(
                aux_output, [targets], aux_indices, num_boxes
            )
            aux_losses = {**aux_vfl_loss, **aux_box_losses, **aux_local_losses}
            weighted_aux_losses = {
                k + f"_aux_{i}": aux_losses[k] * self.weight_dict[k]
                for k in aux_losses
                if k in self.weight_dict
            }
            losses.update(weighted_aux_losses)
        auxiliary_outputs_list.append(
            {
                "logits": enc_topk_logits[:, main_queries_start:],
                "pred_boxes": keras.ops.clip(
                    enc_topk_bboxes[:, main_queries_start:], 0, 1
                ),
            }
        )

        if denoising_meta_values is not None:
            dn_num_split = denoising_meta_values["dn_num_split"]
            if keras.ops.ndim(dn_num_split) > 1:
                dn_num_split = dn_num_split[0]
            max_dn_layers = self.backbone.decoder_layers
            dn_indices = self.get_cdn_matched_indices(
                denoising_meta_values, [targets]
            )
            dn_num_group = denoising_meta_values["dn_num_group"]
            if keras.ops.ndim(dn_num_group) > 0:
                dn_num_group = dn_num_group[0]
            num_boxes_dn = num_boxes * keras.ops.cast(dn_num_group, "float32")
            for i in range(max_dn_layers):
                is_valid = keras.ops.less(i, dn_num_split[0])
                is_not_last_layer = keras.ops.less(i, max_dn_layers - 1)
                teacher_idx = keras.ops.minimum(
                    dn_num_split[0] - 1, max_dn_layers - 1
                )
                dn_aux_output = {
                    "logits": outputs_class[:, i, :, :],
                    "pred_boxes": keras.ops.clip(
                        outputs_coord[:, i, :, :], 0, 1
                    ),
                    "pred_corners": predicted_corners[:, i, :, :],
                    "ref_points": initial_reference_points[:, i, :, :],
                    "teacher_corners": predicted_corners[:, teacher_idx, :, :],
                    "teacher_logits": outputs_class[:, teacher_idx, :, :],
                }
                vfl_loss = self.compute_vfl_loss(
                    dn_aux_output, [targets], dn_indices, num_boxes_dn
                )
                box_losses = self.compute_box_losses(
                    dn_aux_output, [targets], dn_indices, num_boxes_dn
                )
                local_losses = self.compute_local_losses(
                    dn_aux_output,
                    [targets],
                    dn_indices,
                    num_boxes_dn,
                    compute_ddf=is_not_last_layer,
                )
                all_losses = {**vfl_loss, **box_losses, **local_losses}
                weighted_losses = {
                    k + f"_dn_{i}": keras.ops.where(
                        is_valid, all_losses[k] * self.weight_dict[k], 0.0
                    )
                    for k in all_losses
                    if k in self.weight_dict
                }
                losses.update(weighted_losses)
        total_loss = keras.ops.sum([v for v in losses.values()])
        return total_loss

    @property
    def prediction_decoder(self):
        return self._prediction_decoder

    @prediction_decoder.setter
    def prediction_decoder(self, prediction_decoder):
        if prediction_decoder.bounding_box_format != self.bounding_box_format:
            raise ValueError(
                "Expected `prediction_decoder` and `DFineObjectDetector` to "
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
        if isinstance(data, (list, tuple)):
            images, _ = data
        else:
            images = data
        logits = predictions["logits"]
        pred_boxes = predictions["pred_boxes"]
        height, width, _ = keras.ops.shape(images)[1:]
        denormalized_boxes = keras.ops.stack(
            [
                pred_boxes[..., 0] * width,  # center_x
                pred_boxes[..., 1] * height,  # center_y
                pred_boxes[..., 2] * width,  # width
                pred_boxes[..., 3] * height,  # height
            ],
            axis=-1,
        )
        pred_boxes_xyxy = center_to_corners_format(denormalized_boxes)
        pred_boxes_yxyx = keras.ops.stack(
            [
                pred_boxes_xyxy[..., 1],  # y_min
                pred_boxes_xyxy[..., 0],  # x_min
                pred_boxes_xyxy[..., 3],  # y_max
                pred_boxes_xyxy[..., 2],  # x_max
            ],
            axis=-1,
        )
        y_pred = self.prediction_decoder(pred_boxes_yxyx, logits, images=images)
        return y_pred

    def _upcast(self, t):
        if keras.backend.is_float_dtype(t.dtype):
            return (
                t
                if t.dtype in ("float32", "float64")
                else keras.ops.cast(t, "float32")
            )
        return (
            t if t.dtype in ("int32", "int64") else keras.ops.cast(t, "int32")
        )

    def box_area(self, boxes):
        boxes = self._upcast(boxes)
        return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])

    def box_iou(self, boxes1, boxes2):
        area1 = self.box_area(boxes1)
        area2 = self.box_area(boxes2)
        left_top = keras.ops.maximum(
            keras.ops.expand_dims(boxes1[..., :2], axis=1),
            keras.ops.expand_dims(boxes2[..., :2], axis=0),
        )
        right_bottom = keras.ops.minimum(
            keras.ops.expand_dims(boxes1[..., 2:], axis=1),
            keras.ops.expand_dims(boxes2[..., 2:], axis=0),
        )
        width_height = keras.ops.maximum(right_bottom - left_top, 0.0)
        inter = width_height[..., 0] * width_height[..., 1]
        union = (
            keras.ops.expand_dims(area1, axis=1)
            + keras.ops.expand_dims(area2, axis=0)
            - inter
        )
        iou = inter / (union + 1e-6)
        return iou, union

    def generalized_box_iou(self, boxes1, boxes2):
        iou, union = self.box_iou(boxes1, boxes2)
        top_left = keras.ops.minimum(
            keras.ops.expand_dims(boxes1[..., :2], axis=1),
            keras.ops.expand_dims(boxes2[..., :2], axis=0),
        )
        bottom_right = keras.ops.maximum(
            keras.ops.expand_dims(boxes1[..., 2:], axis=1),
            keras.ops.expand_dims(boxes2[..., 2:], axis=0),
        )
        width_height = keras.ops.maximum(bottom_right - top_left, 0.0)
        area = width_height[..., 0] * width_height[..., 1]
        return iou - (area - union) / (area + 1e-6)

    def gather_along_first_two_dims(self, tensor, batch_idx, src_idx):
        batch_size, num_queries, *feature_dims = keras.ops.shape(tensor)
        linear_idx = batch_idx * num_queries + src_idx
        flat_tensor = keras.ops.reshape(
            tensor, (batch_size * num_queries, *feature_dims)
        )
        gathered = keras.ops.take(flat_tensor, linear_idx, axis=0)
        return gathered

    def hungarian_assignment(self, cost_matrix):
        num_rows, num_cols = keras.ops.shape(cost_matrix)
        matrix_size = num_rows
        cost = keras.ops.cast(cost_matrix, dtype="float32")
        row_covered = keras.ops.zeros((num_rows,), dtype="bool")
        col_covered = keras.ops.zeros((num_cols,), dtype="bool")
        assignments = keras.ops.full((matrix_size, 2), -1, dtype="int64")
        step = keras.ops.convert_to_tensor(1, dtype="int32")
        iteration = keras.ops.convert_to_tensor(0, dtype="int32")

        def condition(
            step, cost, row_covered, col_covered, assignments, iteration
        ):
            return keras.ops.logical_and(step <= 4, iteration < num_cols * 2)

        def body(step, cost, row_covered, col_covered, assignments, iteration):
            def step_1():
                row_min = keras.ops.min(cost, axis=1, keepdims=True)
                new_cost = cost - row_min
                return (
                    keras.ops.convert_to_tensor(2),
                    new_cost,
                    row_covered,
                    col_covered,
                    assignments,
                )

            def step_2():
                col_min = keras.ops.min(cost, axis=0, keepdims=True)
                new_cost = cost - col_min
                return (
                    keras.ops.convert_to_tensor(3),
                    new_cost,
                    row_covered,
                    col_covered,
                    assignments,
                )

            def step_3():
                zero_mask = keras.ops.abs(cost) < 1e-6
                assigned_count = keras.ops.convert_to_tensor(0, dtype="int32")

                def assign_loop_cond(ac, current_rm, current_cm, assign):
                    uncovered_mask = keras.ops.logical_not(
                        current_rm[:, None] | current_cm[None, :]
                    )
                    has_uncovered_zero = keras.ops.any(
                        zero_mask & uncovered_mask
                    )
                    return keras.ops.logical_and(
                        ac < num_cols, has_uncovered_zero
                    )

                def assign_loop_body(ac, current_rm, current_cm, assign):
                    uncovered_mask = keras.ops.logical_not(
                        current_rm[:, None] | current_cm[None, :]
                    )
                    potential_zeros = zero_mask & uncovered_mask
                    potential_zeros_flat = keras.ops.reshape(
                        potential_zeros, [-1]
                    )
                    first_idx = keras.ops.argmax(
                        keras.ops.cast(potential_zeros_flat, "int32")
                    )
                    r = first_idx // num_cols
                    c = first_idx % num_cols

                    r_indices = keras.ops.reshape(
                        keras.ops.cast(r, "int64"), (1, 1)
                    )
                    c_indices = keras.ops.reshape(
                        keras.ops.cast(c, "int64"), (1, 1)
                    )
                    current_rm = keras.ops.scatter_update(
                        current_rm, r_indices, [True]
                    )
                    current_cm = keras.ops.scatter_update(
                        current_cm, c_indices, [True]
                    )

                    assign_indices = keras.ops.reshape(
                        keras.ops.cast(ac, "int64"), (1, 1)
                    )
                    assign_updates = keras.ops.reshape(
                        keras.ops.stack([r, c]), (1, 2)
                    )
                    assign = keras.ops.scatter_update(
                        assign,
                        assign_indices,
                        keras.ops.cast(assign_updates, assign.dtype),
                    )

                    return ac + 1, current_rm, current_cm, assign

                (
                    _,
                    row_covered_updated,
                    col_covered_updated,
                    assignments_updated,
                ) = keras.ops.while_loop(
                    assign_loop_cond,
                    assign_loop_body,
                    (
                        assigned_count,
                        row_covered,
                        col_covered,
                        assignments,
                    ),
                    maximum_iterations=num_cols,
                )
                num_assigned = keras.ops.sum(
                    keras.ops.cast(assignments_updated[:, 0] >= 0, "int32")
                )
                next_step = keras.ops.where(num_assigned == num_cols, 4, 3)
                return (
                    next_step,
                    cost,
                    row_covered_updated,
                    col_covered_updated,
                    assignments_updated,
                )

            def step_4():
                large_value = keras.ops.cast(1e10, dtype=cost.dtype)
                uncovered_cost = keras.ops.where(
                    keras.ops.logical_not(
                        keras.ops.expand_dims(row_covered, 1)
                        | keras.ops.expand_dims(col_covered, 0)
                    ),
                    cost,
                    large_value,
                )
                min_val = keras.ops.min(uncovered_cost)

                def large_value_case():
                    return (
                        keras.ops.convert_to_tensor(4),
                        cost,
                        row_covered,
                        col_covered,
                        assignments,
                    )

                def normal_case():
                    new_cost = cost - keras.ops.where(
                        keras.ops.logical_not(row_covered)[:, None]
                        & keras.ops.logical_not(col_covered)[None, :],
                        min_val,
                        0.0,
                    )
                    new_cost = new_cost + keras.ops.where(
                        row_covered[:, None] & col_covered[None, :],
                        min_val,
                        0.0,
                    )
                    return (
                        keras.ops.convert_to_tensor(3),
                        new_cost,
                        row_covered,
                        col_covered,
                        assignments,
                    )

                return keras.ops.cond(
                    keras.ops.equal(min_val, large_value),
                    large_value_case,
                    normal_case,
                )

            (
                next_step,
                new_cost,
                new_row_covered,
                new_col_covered,
                new_assignments,
            ) = keras.ops.switch(
                step - 1,
                [step_1, step_2, step_3, step_4],
            )
            return (
                next_step,
                new_cost,
                new_row_covered,
                new_col_covered,
                new_assignments,
                iteration + 1,
            )

        (
            final_step,
            final_cost,
            final_row_covered,
            final_col_covered,
            final_assignments,
            _,
        ) = keras.ops.while_loop(
            condition,
            body,
            (step, cost, row_covered, col_covered, assignments, iteration),
            maximum_iterations=num_cols * 2,
        )
        valid_mask = final_assignments[:, 0] >= 0
        valid_indices_mask = keras.ops.cast(valid_mask, "int32")
        num_valid = keras.ops.sum(valid_indices_mask)
        valid_positions = keras.ops.cumsum(valid_indices_mask, axis=0) - 1
        max_valid_pos = keras.ops.maximum(num_valid - 1, 0)
        valid_positions = keras.ops.minimum(valid_positions, max_valid_pos)
        row_ind = keras.ops.where(valid_mask, final_assignments[:, 0], -1)
        col_ind = keras.ops.where(valid_mask, final_assignments[:, 1], -1)
        valid_row_mask = row_ind >= 0
        valid_col_mask = col_ind >= 0
        row_ind = keras.ops.where(valid_row_mask, row_ind, 0)
        col_ind = keras.ops.where(valid_col_mask, col_ind, 0)
        return row_ind, col_ind

    def hungarian_matcher(self, outputs, targets):
        batch_size = keras.ops.shape(outputs["logits"])[0]
        num_queries = keras.ops.shape(outputs["logits"])[1]
        out_logits_flat = keras.ops.reshape(
            outputs["logits"], (-1, self.num_classes)
        )
        out_bbox_flat = keras.ops.reshape(outputs["pred_boxes"], (-1, 4))
        target_ids_list = [keras.ops.cast(targets[0]["labels"], dtype="int32")]
        boxes = targets[0]["boxes"]
        target_bbox = keras.ops.cond(
            keras.ops.equal(keras.ops.ndim(boxes), 3),
            lambda: keras.ops.reshape(boxes, (-1, keras.ops.shape(boxes)[-1])),
            lambda: boxes,
        )
        target_bbox_list = [target_bbox]
        target_ids_concat = keras.ops.concatenate(target_ids_list, axis=0)
        target_bbox_concat = keras.ops.concatenate(target_bbox_list, axis=0)
        if self.use_focal_loss:
            out_prob_flat = keras.ops.sigmoid(out_logits_flat)
            prob_for_target_classes = keras.ops.take(
                out_prob_flat, target_ids_concat, axis=1
            )
            p = prob_for_target_classes
            pos_cost = (
                self.matcher_alpha
                * keras.ops.power(1 - p, self.matcher_gamma)
                * (-keras.ops.log(p + 1e-8))
            )
            neg_cost = (
                (1 - self.matcher_alpha)
                * keras.ops.power(p, self.matcher_gamma)
                * (-keras.ops.log(1 - p + 1e-8))
            )
            class_cost = pos_cost - neg_cost
        else:
            out_prob_softmax_flat = keras.ops.softmax(out_logits_flat, axis=-1)
            prob_for_target_classes = keras.ops.take(
                out_prob_softmax_flat, target_ids_concat, axis=1
            )
            class_cost = -prob_for_target_classes

        bbox_cost = keras.ops.sum(
            keras.ops.abs(
                keras.ops.expand_dims(out_bbox_flat, 1)
                - keras.ops.expand_dims(target_bbox_concat, 0)
            ),
            axis=2,
        )
        out_bbox_corners = center_to_corners_format(out_bbox_flat)
        target_bbox_corners = center_to_corners_format(target_bbox_concat)
        giou_cost = -self.generalized_box_iou(
            out_bbox_corners, target_bbox_corners
        )

        cost_matrix_flat = (
            self.matcher_bbox_cost * bbox_cost
            + self.matcher_class_cost * class_cost
            + self.matcher_giou_cost * giou_cost
        )
        num_targets = keras.ops.shape(target_ids_concat)[0]
        cost_matrix = keras.ops.reshape(
            cost_matrix_flat, (batch_size, num_queries, num_targets)
        )
        max_matches = num_queries
        row_indices_init = keras.ops.zeros(
            (batch_size, max_matches), dtype="int64"
        )
        col_indices_init = keras.ops.zeros(
            (batch_size, max_matches), dtype="int64"
        )
        valid_masks_init = keras.ops.zeros(
            (batch_size, max_matches), dtype="bool"
        )

        def loop_condition(i, row_indices, col_indices, valid_masks):
            return keras.ops.less(i, batch_size)

        def loop_body(i, row_indices, col_indices, valid_masks):
            row_idx, col_idx = self.hungarian_assignment(cost_matrix[i, :, :])
            valid_mask = keras.ops.ones(
                (keras.ops.shape(row_idx)[0],), dtype="bool"
            )
            pad_size = max_matches - keras.ops.shape(row_idx)[0]
            row_idx = keras.ops.pad(
                row_idx, [[0, pad_size]], constant_values=-1
            )
            col_idx = keras.ops.pad(
                col_idx, [[0, pad_size]], constant_values=-1
            )
            valid_mask = keras.ops.pad(
                valid_mask, [[0, pad_size]], constant_values=False
            )
            row_indices = keras.ops.scatter_update(
                row_indices, [[i]], keras.ops.expand_dims(row_idx, axis=0)
            )
            col_indices = keras.ops.scatter_update(
                col_indices, [[i]], keras.ops.expand_dims(col_idx, axis=0)
            )
            valid_masks = keras.ops.scatter_update(
                valid_masks, [[i]], keras.ops.expand_dims(valid_mask, axis=0)
            )
            return i + 1, row_indices, col_indices, valid_masks

        _, row_indices, col_indices, valid_masks = keras.ops.while_loop(
            loop_condition,
            loop_body,
            (
                keras.ops.convert_to_tensor(0, dtype="int32"),
                row_indices_init,
                col_indices_init,
                valid_masks_init,
            ),
            maximum_iterations=batch_size,
        )
        return (row_indices, col_indices, valid_masks)

    def compute_vfl_loss(self, outputs, targets, indices, num_boxes):
        _, col_indices, valid_masks = indices
        batch_idx, src_idx = self._get_source_permutation_idx(indices)
        src_boxes = self.gather_along_first_two_dims(
            outputs["pred_boxes"], batch_idx, src_idx
        )
        flat_col_indices = keras.ops.reshape(col_indices, (-1,))
        flat_valid_masks = keras.ops.reshape(valid_masks, (-1,))
        src_logits = outputs["logits"]
        target_classes_init = keras.ops.full(
            shape=keras.ops.shape(src_logits)[:2],
            fill_value=self.num_classes,
            dtype="int32",
        )
        target_score_original = keras.ops.zeros_like(
            target_classes_init, dtype=src_logits.dtype
        )
        update_indices = keras.ops.stack([batch_idx, src_idx], axis=-1)

        def process_targets():
            target_labels_tensor = keras.ops.stack(
                [t["labels"] for t in targets], axis=0
            )
            target_boxes_tensor = keras.ops.stack(
                [t["boxes"] for t in targets], axis=0
            )
            if keras.ops.ndim(target_labels_tensor) == 3:
                target_labels_tensor = keras.ops.squeeze(
                    target_labels_tensor, axis=1
                )
            if keras.ops.ndim(target_boxes_tensor) == 4:
                target_boxes_tensor = keras.ops.squeeze(
                    target_boxes_tensor, axis=1
                )
            flat_target_labels = keras.ops.reshape(target_labels_tensor, (-1,))
            flat_target_boxes = keras.ops.reshape(target_boxes_tensor, (-1, 4))
            num_targets = keras.ops.shape(flat_target_labels)[0]
            num_targets = keras.ops.cast(
                num_targets, dtype=flat_col_indices.dtype
            )
            safe_flat_col_indices = keras.ops.where(
                (flat_col_indices >= 0) & (flat_col_indices < num_targets),
                flat_col_indices,
                0,
            )
            target_classes_flat = keras.ops.take(
                flat_target_labels, safe_flat_col_indices, axis=0
            )
            target_boxes_flat = keras.ops.take(
                flat_target_boxes, safe_flat_col_indices, axis=0
            )
            target_classes_flat = keras.ops.where(
                flat_valid_masks, target_classes_flat, self.num_classes
            )
            target_boxes_flat = keras.ops.where(
                keras.ops.expand_dims(flat_valid_masks, axis=-1),
                target_boxes_flat,
                0.0,
            )
            src_boxes_corners = center_to_corners_format(
                keras.ops.stop_gradient(src_boxes)
            )
            target_boxes_corners = center_to_corners_format(target_boxes_flat)
            ious_matrix, _ = self.box_iou(
                src_boxes_corners, target_boxes_corners
            )
            ious = keras.ops.diagonal(ious_matrix)
            target_classes_flat = keras.ops.cast(
                target_classes_flat, dtype="int32"
            )
            ious = keras.ops.cast(ious, dtype=src_logits.dtype)
            target_classes_updated = keras.ops.scatter_update(
                target_classes_init, update_indices, target_classes_flat
            )
            target_score_updated = keras.ops.scatter_update(
                target_score_original, update_indices, ious
            )
            return target_classes_updated, target_score_updated

        target_classes, target_score_original = process_targets()
        target_one_hot = keras.ops.one_hot(
            target_classes, num_classes=self.num_classes + 1
        )[..., :-1]
        target_score = (
            keras.ops.expand_dims(target_score_original, axis=-1)
            * target_one_hot
        )
        pred_score_sigmoid = keras.ops.sigmoid(
            keras.ops.stop_gradient(src_logits)
        )
        weight = (
            self.matcher_alpha
            * keras.ops.power(pred_score_sigmoid, self.matcher_gamma)
            * (1 - target_one_hot)
            + target_score
        )
        loss_vfl = keras.ops.binary_crossentropy(
            target_score, src_logits, from_logits=True
        )
        loss_vfl = loss_vfl * weight
        loss_vfl = (
            keras.ops.sum(keras.ops.mean(loss_vfl, axis=1))
            * keras.ops.cast(
                keras.ops.shape(src_logits)[1], dtype=loss_vfl.dtype
            )
            / num_boxes
        )
        return {"loss_vfl": loss_vfl}

    def compute_box_losses(self, outputs, targets, indices, num_boxes):
        _, col_indices, valid_masks = indices
        batch_idx, src_idx = self._get_source_permutation_idx(indices)
        src_boxes = self.gather_along_first_two_dims(
            outputs["pred_boxes"], batch_idx, src_idx
        )
        target_boxes_all = targets[0]["boxes"]
        if keras.ops.ndim(target_boxes_all) == 3:
            target_boxes_all = keras.ops.squeeze(target_boxes_all, axis=0)
        col_indices_flat = keras.ops.reshape(col_indices, [-1])
        valid_masks_flat = keras.ops.reshape(valid_masks, [-1])
        max_box_idx = keras.ops.maximum(
            keras.ops.shape(target_boxes_all)[0] - 1, 0
        )
        max_box_idx = keras.ops.cast(max_box_idx, dtype=col_indices_flat.dtype)
        safe_col_indices = keras.ops.clip(col_indices_flat, 0, max_box_idx)
        target_boxes = keras.ops.take(
            target_boxes_all, safe_col_indices, axis=0
        )
        valid_masks_expanded = keras.ops.expand_dims(valid_masks_flat, axis=-1)
        valid_masks_expanded = keras.ops.cast(
            valid_masks_expanded, target_boxes.dtype
        )
        target_boxes = target_boxes * valid_masks_expanded
        is_empty = keras.ops.logical_or(
            keras.ops.equal(keras.ops.shape(src_boxes)[0], 0),
            keras.ops.equal(keras.ops.shape(target_boxes)[0], 0),
        )
        return keras.ops.cond(
            is_empty,
            lambda: {
                "loss_bbox": keras.ops.convert_to_tensor(
                    0.0, dtype=keras.backend.floatx()
                ),
                "loss_giou": keras.ops.convert_to_tensor(
                    0.0, dtype=keras.backend.floatx()
                ),
            },
            lambda: {
                "loss_bbox": keras.ops.sum(
                    keras.ops.abs(src_boxes - target_boxes)
                )
                / num_boxes,
                "loss_giou": (
                    keras.ops.sum(
                        1.0
                        - keras.ops.diagonal(
                            self.generalized_box_iou(
                                center_to_corners_format(src_boxes),
                                center_to_corners_format(target_boxes),
                            )
                        )
                    )
                    / num_boxes
                ),
            },
        )

    def compute_local_losses(
        self, outputs, targets, indices, num_boxes, T=5, compute_ddf=None
    ):
        losses = {}
        if (
            "pred_corners" not in outputs
            or outputs["pred_corners"] is None
            or "ref_points" not in outputs
            or outputs["ref_points"] is None
        ):
            losses["loss_fgl"] = keras.ops.convert_to_tensor(
                0.0, dtype=keras.backend.floatx()
            )
            losses["loss_ddf"] = keras.ops.convert_to_tensor(
                0.0, dtype=keras.backend.floatx()
            )
            return losses

        if compute_ddf is None:
            compute_ddf = (
                "teacher_corners" in outputs
                and outputs["teacher_corners"] is not None
                and "teacher_logits" in outputs
            )

        _, col_indices, valid_masks = indices
        batch_idx, src_idx = self._get_source_permutation_idx(indices)
        col_indices_flat = keras.ops.reshape(col_indices, [-1])
        valid_masks_flat = keras.ops.reshape(valid_masks, [-1])
        target_boxes_all = targets[0]["boxes"]
        if keras.ops.ndim(target_boxes_all) == 3:
            target_boxes_all = keras.ops.squeeze(target_boxes_all, axis=0)
        max_box_idx = keras.ops.maximum(
            keras.ops.shape(target_boxes_all)[0] - 1, 0
        )
        max_box_idx = keras.ops.cast(max_box_idx, dtype=col_indices_flat.dtype)
        safe_col_indices = keras.ops.clip(col_indices_flat, 0, max_box_idx)
        target_boxes_matched_center = keras.ops.take(
            target_boxes_all, safe_col_indices, axis=0
        )
        valid_masks_expanded = keras.ops.expand_dims(valid_masks_flat, axis=-1)
        valid_masks_expanded = keras.ops.cast(
            valid_masks_expanded, target_boxes_matched_center.dtype
        )
        target_boxes_matched_center = (
            target_boxes_matched_center * valid_masks_expanded
        )

        def compute_losses_fn():
            pred_corners_matched_flat = self.gather_along_first_two_dims(
                outputs["pred_corners"], batch_idx, src_idx
            )
            pred_corners_matched = keras.ops.reshape(
                pred_corners_matched_flat, (-1, self.backbone.max_num_bins + 1)
            )
            ref_points_matched = self.gather_along_first_two_dims(
                outputs["ref_points"], batch_idx, src_idx
            )
            ref_points_matched = keras.ops.stop_gradient(ref_points_matched)
            target_boxes_corners_matched = center_to_corners_format(
                target_boxes_matched_center
            )
            reg_scale_tensor = self.backbone.decoder.reg_scale
            up_tensor = self.backbone.decoder.up
            target_corners_dist, weight_right, weight_left = self.bbox2distance(
                ref_points_matched,
                target_boxes_corners_matched,
                self.backbone.max_num_bins,
                reg_scale_tensor,
                up_tensor,
            )
            pred_boxes_matched_center = self.gather_along_first_two_dims(
                outputs["pred_boxes"], batch_idx, src_idx
            )
            pred_boxes_corners_matched = center_to_corners_format(
                pred_boxes_matched_center
            )
            ious_pairwise, _ = self.box_iou(
                pred_boxes_corners_matched, target_boxes_corners_matched
            )
            ious = keras.ops.diagonal(ious_pairwise)
            weight_targets_fgl = keras.ops.reshape(
                keras.ops.tile(keras.ops.expand_dims(ious, 1), [1, 4]),
                [-1],
            )
            weight_targets_fgl = keras.ops.stop_gradient(weight_targets_fgl)
            losses["loss_fgl"] = self.unimodal_distribution_focal_loss(
                pred_corners_matched,
                target_corners_dist,
                weight_right,
                weight_left,
                weight=weight_targets_fgl,
                avg_factor=num_boxes,
            )

            def ddf_true_fn():
                pred_corners_all = keras.ops.reshape(
                    outputs["pred_corners"],
                    (-1, self.backbone.max_num_bins + 1),
                )
                target_corners_all = keras.ops.reshape(
                    keras.ops.stop_gradient(outputs["teacher_corners"]),
                    (-1, self.backbone.max_num_bins + 1),
                )

                def compute_ddf_loss_fn():
                    weight_targets_local = keras.ops.max(
                        keras.ops.sigmoid(outputs["teacher_logits"]), axis=-1
                    )
                    mask = keras.ops.zeros_like(
                        weight_targets_local, dtype="bool"
                    )
                    mask_flat = keras.ops.scatter_update(
                        keras.ops.reshape(mask, (-1,)),
                        keras.ops.expand_dims(src_idx, axis=-1),
                        keras.ops.ones_like(batch_idx, dtype="bool"),
                    )
                    mask = keras.ops.reshape(
                        mask_flat, keras.ops.shape(weight_targets_local)
                    )
                    weight_targets_local_matched = keras.ops.scatter_update(
                        keras.ops.reshape(weight_targets_local, (-1,)),
                        keras.ops.expand_dims(src_idx, axis=-1),
                        ious,
                    )
                    weight_targets_local = keras.ops.reshape(
                        weight_targets_local_matched,
                        keras.ops.shape(weight_targets_local),
                    )
                    weight_targets_local_expanded = keras.ops.reshape(
                        keras.ops.tile(
                            keras.ops.expand_dims(
                                weight_targets_local, axis=-1
                            ),
                            [1, 1, 4],
                        ),
                        [-1],
                    )
                    weight_targets_local_expanded = keras.ops.stop_gradient(
                        weight_targets_local_expanded
                    )
                    pred_softmax = keras.ops.softmax(
                        pred_corners_all / T, axis=-1
                    )
                    target_softmax = keras.ops.softmax(
                        target_corners_all / T, axis=-1
                    )
                    kl_div = keras.ops.sum(
                        target_softmax
                        * (
                            keras.ops.log(target_softmax + 1e-8)
                            - keras.ops.log(pred_softmax + 1e-8)
                        ),
                        axis=-1,
                    )
                    loss_match_local = (
                        weight_targets_local_expanded * (T**2) * kl_div
                    )
                    mask_expanded = keras.ops.expand_dims(mask, axis=-1)
                    mask_expanded = keras.ops.tile(mask_expanded, [1, 1, 4])
                    mask_flat = keras.ops.reshape(mask_expanded, (-1,))
                    loss_match_local1 = keras.ops.cond(
                        keras.ops.any(mask_flat),
                        lambda: keras.ops.sum(
                            loss_match_local
                            * keras.ops.cast(mask_flat, loss_match_local.dtype)
                        )
                        / keras.ops.sum(
                            keras.ops.cast(mask_flat, loss_match_local.dtype)
                        ),
                        lambda: keras.ops.convert_to_tensor(
                            0.0, dtype=loss_match_local.dtype
                        ),
                    )
                    neg_mask_flat = keras.ops.logical_not(mask_flat)
                    loss_match_local2 = keras.ops.cond(
                        keras.ops.any(neg_mask_flat),
                        lambda: keras.ops.sum(
                            loss_match_local
                            * keras.ops.cast(
                                neg_mask_flat, loss_match_local.dtype
                            )
                        )
                        / keras.ops.sum(
                            keras.ops.cast(
                                neg_mask_flat, loss_match_local.dtype
                            )
                        ),
                        lambda: keras.ops.convert_to_tensor(
                            0.0, dtype=loss_match_local.dtype
                        ),
                    )
                    batch_scale = 1.0 / keras.ops.cast(
                        keras.ops.shape(outputs["pred_boxes"])[0],
                        dtype="float32",
                    )
                    num_pos = keras.ops.sqrt(
                        keras.ops.sum(keras.ops.cast(mask, dtype="float32"))
                        * batch_scale
                    )
                    num_neg = keras.ops.sqrt(
                        keras.ops.sum(keras.ops.cast(~mask, dtype="float32"))
                        * batch_scale
                    )
                    return (
                        loss_match_local1 * num_pos
                        + loss_match_local2 * num_neg
                    ) / (num_pos + num_neg + 1e-8)

                all_equal = keras.ops.all(
                    keras.ops.equal(pred_corners_all, target_corners_all)
                )
                return keras.ops.cond(
                    all_equal,
                    lambda: keras.ops.sum(pred_corners_all) * 0.0,
                    compute_ddf_loss_fn,
                )

            def ddf_false_fn():
                return keras.ops.convert_to_tensor(
                    0.0, dtype=keras.backend.floatx()
                )

            losses["loss_ddf"] = keras.ops.cond(
                compute_ddf, ddf_true_fn, ddf_false_fn
            )
            return losses

        def empty_case_fn():
            losses["loss_fgl"] = keras.ops.convert_to_tensor(
                0.0, dtype=keras.backend.floatx()
            )
            losses["loss_ddf"] = keras.ops.convert_to_tensor(
                0.0, dtype=keras.backend.floatx()
            )
            return losses

        is_empty = keras.ops.equal(
            keras.ops.shape(target_boxes_matched_center)[0], 0
        )
        return keras.ops.cond(is_empty, empty_case_fn, compute_losses_fn)

    def _translate_gt_valid_case(
        self, gt_flat, valid_idx_mask, function_values, max_num_bins, mask
    ):
        closest_left_indices = (
            keras.ops.sum(keras.ops.cast(mask, "int32"), axis=1) - 1
        )
        indices_float = keras.ops.cast(
            closest_left_indices, dtype=gt_flat.dtype
        )
        weight_right = keras.ops.zeros_like(indices_float)
        weight_left = keras.ops.zeros_like(indices_float)
        valid_indices_int = keras.ops.arange(keras.ops.shape(valid_idx_mask)[0])
        valid_indices_int = keras.ops.where(
            valid_idx_mask, valid_indices_int, -1
        )
        valid_indices_int = keras.ops.where(
            valid_indices_int >= 0, valid_indices_int, 0
        )
        valid_indices_long = keras.ops.cast(
            keras.ops.where(
                valid_idx_mask,
                keras.ops.take(indices_float, valid_indices_int, axis=0),
                0.0,
            ),
            "int32",
        )
        gt_valid = keras.ops.where(
            valid_idx_mask,
            keras.ops.take(gt_flat, valid_indices_int, axis=0),
            0.0,
        )
        left_values = keras.ops.take(
            function_values, valid_indices_long, axis=0
        )
        right_values = keras.ops.take(
            function_values,
            keras.ops.clip(
                valid_indices_long + 1,
                0,
                keras.ops.shape(function_values)[0] - 1,
            ),
            axis=0,
        )
        left_diffs = keras.ops.abs(gt_valid - left_values)
        right_diffs = keras.ops.abs(right_values - gt_valid)
        wr_valid = left_diffs / (left_diffs + right_diffs + 1e-8)
        wl_valid = 1.0 - wr_valid
        weight_right = keras.ops.where(
            keras.ops.expand_dims(valid_idx_mask, axis=-1),
            keras.ops.expand_dims(wr_valid, axis=-1),
            keras.ops.expand_dims(weight_right, axis=-1),
        )
        weight_right = keras.ops.squeeze(weight_right, axis=-1)
        weight_left = keras.ops.where(
            keras.ops.expand_dims(valid_idx_mask, axis=-1),
            keras.ops.expand_dims(wl_valid, axis=-1),
            keras.ops.expand_dims(weight_left, axis=-1),
        )
        weight_left = keras.ops.squeeze(weight_left, axis=-1)
        indices_float = keras.ops.where(
            indices_float < 0,
            keras.ops.zeros_like(indices_float),
            indices_float,
        )
        weight_right = keras.ops.where(
            indices_float < 0, keras.ops.zeros_like(weight_right), weight_right
        )
        weight_left = keras.ops.where(
            indices_float < 0, keras.ops.ones_like(weight_left), weight_left
        )
        indices_float = keras.ops.where(
            indices_float >= max_num_bins,
            keras.ops.cast(max_num_bins - 0.1, dtype=indices_float.dtype),
            indices_float,
        )
        weight_right = keras.ops.where(
            indices_float >= max_num_bins,
            keras.ops.ones_like(weight_right),
            weight_right,
        )
        weight_left = keras.ops.where(
            indices_float >= max_num_bins,
            keras.ops.zeros_like(weight_left),
            weight_left,
        )
        return indices_float, weight_right, weight_left

    def translate_gt(self, gt, max_num_bins, reg_scale, up):
        gt_flat = keras.ops.reshape(gt, [-1])
        function_values = weighting_function(max_num_bins, up, reg_scale)
        diffs = keras.ops.expand_dims(
            function_values, axis=0
        ) - keras.ops.expand_dims(gt_flat, axis=1)
        mask = diffs <= 0
        closest_left_indices = (
            keras.ops.sum(keras.ops.cast(mask, "int32"), axis=1) - 1
        )
        indices_float = keras.ops.cast(
            closest_left_indices, dtype=gt_flat.dtype
        )
        weight_right = keras.ops.zeros_like(indices_float)
        weight_left = keras.ops.zeros_like(indices_float)
        valid_idx_mask = (indices_float >= 0) & (indices_float < max_num_bins)
        return keras.ops.cond(
            keras.ops.any(valid_idx_mask),
            lambda: self._translate_gt_valid_case(
                gt_flat, valid_idx_mask, function_values, max_num_bins, mask
            ),
            lambda: (
                keras.ops.zeros_like(indices_float),
                keras.ops.zeros_like(weight_right),
                keras.ops.ones_like(weight_left),
            ),
        )

    def _compute_bbox2distance(
        self, points, bbox, max_num_bins, reg_scale, up, eps=0.1
    ):
        reg_scale_abs = keras.ops.abs(reg_scale)
        left = (points[..., 0] - bbox[..., 0]) / (
            points[..., 2] / reg_scale_abs + 1e-16
        ) - 0.5 * reg_scale_abs
        top = (points[..., 1] - bbox[..., 1]) / (
            points[..., 3] / reg_scale_abs + 1e-16
        ) - 0.5 * reg_scale_abs
        right = (bbox[..., 2] - points[..., 0]) / (
            points[..., 2] / reg_scale_abs + 1e-16
        ) - 0.5 * reg_scale_abs
        bottom = (bbox[..., 3] - points[..., 1]) / (
            points[..., 3] / reg_scale_abs + 1e-16
        ) - 0.5 * reg_scale_abs
        four_lens = keras.ops.stack([left, top, right, bottom], axis=-1)
        up_tensor = (
            keras.ops.convert_to_tensor(up)
            if not isinstance(up, (keras.KerasTensor))
            else up
        )
        four_lens_translated, weight_right, weight_left = self.translate_gt(
            four_lens, max_num_bins, reg_scale_abs, up_tensor
        )
        four_lens_translated = keras.ops.clip(
            four_lens_translated, 0, max_num_bins - eps
        )
        return (
            keras.ops.stop_gradient(four_lens_translated),
            keras.ops.stop_gradient(weight_right),
            keras.ops.stop_gradient(weight_left),
        )

    def bbox2distance(self, points, bbox, max_num_bins, reg_scale, up, eps=0.1):
        expected_flat_size = keras.ops.shape(points)[0] * 4
        return keras.ops.cond(
            keras.ops.equal(keras.ops.shape(points)[0], 0),
            lambda: (
                keras.ops.zeros(
                    (expected_flat_size,), dtype=keras.backend.floatx()
                ),
                keras.ops.zeros(
                    (expected_flat_size,), dtype=keras.backend.floatx()
                ),
                keras.ops.zeros(
                    (expected_flat_size,), dtype=keras.backend.floatx()
                ),
            ),
            lambda: self._compute_bbox2distance(
                points, bbox, max_num_bins, reg_scale, up, eps
            ),
        )

    def unimodal_distribution_focal_loss(
        self,
        pred,
        label,
        weight_right,
        weight_left,
        weight=None,
        reduction="sum",
        avg_factor=None,
    ):
        label_flat = keras.ops.reshape(label, [-1])
        weight_right_flat = keras.ops.reshape(weight_right, [-1])
        weight_left_flat = keras.ops.reshape(weight_left, [-1])
        dis_left = keras.ops.cast(label_flat, "int32")
        dis_right = dis_left + 1
        loss_left = (
            keras.ops.sparse_categorical_crossentropy(
                dis_left, pred, from_logits=True
            )
            * weight_left_flat
        )
        loss_right = (
            keras.ops.sparse_categorical_crossentropy(
                dis_right, pred, from_logits=True
            )
            * weight_right_flat
        )
        loss = loss_left + loss_right
        if weight is not None:
            loss = loss * keras.ops.cast(weight, dtype=loss.dtype)
        if avg_factor is not None:
            loss = keras.ops.sum(loss) / avg_factor
        elif reduction == "mean":
            loss = keras.ops.mean(loss)
        elif reduction == "sum":
            loss = keras.ops.sum(loss)
        return loss

    def _get_source_permutation_idx(self, indices):
        row_indices, _, valid_masks = indices
        batch_size = keras.ops.shape(row_indices)[0]
        max_matches = keras.ops.shape(row_indices)[1]
        row_indices_flat = keras.ops.reshape(row_indices, (-1,))
        valid_masks_flat = keras.ops.reshape(valid_masks, (-1,))
        batch_indices = keras.ops.arange(batch_size, dtype="int32")
        batch_indices = keras.ops.expand_dims(batch_indices, axis=1)
        batch_indices = keras.ops.tile(batch_indices, [1, max_matches])
        batch_indices_flat = keras.ops.reshape(batch_indices, (-1,))
        batch_indices_flat = keras.ops.cast(batch_indices_flat, dtype="int64")
        valid_positions = keras.ops.cast(valid_masks_flat, dtype="int32")
        num_valid = keras.ops.sum(valid_positions)
        valid_batch_indices = keras.ops.where(
            valid_masks_flat,
            batch_indices_flat,
            keras.ops.zeros_like(batch_indices_flat),
        )
        valid_src_indices = keras.ops.where(
            valid_masks_flat,
            keras.ops.cast(row_indices_flat, dtype="int64"),
            keras.ops.zeros_like(
                keras.ops.cast(row_indices_flat, dtype="int64")
            ),
        )

        def non_empty_case():
            return valid_batch_indices, valid_src_indices

        def empty_case():
            return (
                keras.ops.zeros_like(valid_batch_indices),
                keras.ops.zeros_like(valid_src_indices),
            )

        batch_idx, src_idx = keras.ops.cond(
            keras.ops.greater(num_valid, 0),
            non_empty_case,
            empty_case,
        )

        return batch_idx, src_idx

    def get_cdn_matched_indices(self, dn_meta, targets):
        dn_positive_idx = dn_meta["dn_positive_idx"]
        batch_size = keras.ops.shape(dn_positive_idx)[0]
        num_denoising_queries = keras.ops.shape(dn_positive_idx)[1]
        row_indices = keras.ops.tile(
            keras.ops.expand_dims(
                keras.ops.arange(num_denoising_queries, dtype="int64"), 0
            ),
            [batch_size, 1],
        )
        col_indices = dn_positive_idx
        valid_masks = keras.ops.not_equal(col_indices, -1)
        return (row_indices, col_indices, valid_masks)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "bounding_box_format": self.bounding_box_format,
                "matcher_class_cost": self.matcher_class_cost,
                "matcher_bbox_cost": self.matcher_bbox_cost,
                "matcher_giou_cost": self.matcher_giou_cost,
                "use_focal_loss": self.use_focal_loss,
                "matcher_alpha": self.matcher_alpha,
                "matcher_gamma": self.matcher_gamma,
                "weight_loss_vfl": self.weight_dict["loss_vfl"],
                "weight_loss_bbox": self.weight_dict["loss_bbox"],
                "weight_loss_giou": self.weight_dict["loss_giou"],
                "weight_loss_fgl": self.weight_dict["loss_fgl"],
                "weight_loss_ddf": self.weight_dict["loss_ddf"],
                "prediction_decoder": keras.saving.serialize_keras_object(
                    self._prediction_decoder
                ),
            }
        )
        return config

    def predict_step(self, *args):
        outputs = super().predict_step(*args)
        if isinstance(outputs, tuple):
            return self.decode_predictions(outputs[0], args[-1]), outputs[1]
        return self.decode_predictions(outputs, *args)

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        if "backbone" in config and isinstance(config["backbone"], dict):
            config["backbone"] = keras.saving.deserialize_keras_object(
                config["backbone"]
            )
        if "preprocessor" in config and isinstance(
            config["preprocessor"], dict
        ):
            config["preprocessor"] = keras.saving.deserialize_keras_object(
                config["preprocessor"]
            )
        if "prediction_decoder" in config and isinstance(
            config["prediction_decoder"], dict
        ):
            config["prediction_decoder"] = (
                keras.saving.deserialize_keras_object(
                    config["prediction_decoder"]
                )
            )
        return cls(**config)
