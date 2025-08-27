import keras

from keras_hub.src.models.d_fine.d_fine_utils import hungarian_assignment
from keras_hub.src.models.d_fine.d_fine_utils import weighting_function


def gather_along_first_two_dims(tensor, batch_idx, src_idx):
    batch_size, num_queries, *feature_dims = keras.ops.shape(tensor)
    batch_size = keras.ops.cast(batch_size, dtype=batch_idx.dtype)
    num_queries = keras.ops.cast(num_queries, dtype=batch_idx.dtype)
    linear_idx = batch_idx * num_queries + src_idx
    flat_tensor = keras.ops.reshape(tensor, (-1, *feature_dims))
    gathered = keras.ops.take(flat_tensor, linear_idx, axis=0)
    return gathered


def hungarian_matcher(
    outputs,
    targets,
    num_targets_per_image,
    use_focal_loss,
    matcher_alpha,
    matcher_gamma,
    matcher_bbox_cost,
    matcher_class_cost,
    matcher_ciou_cost,
    backbone,
):
    """Performs bipartite matching between predictions and ground truths.

    This method implements the Hungarian matching algorithm to find the
    optimal one-to-one assignment between the model's predictions (queries)
    and the ground truth objects. The cost matrix for the assignment is a
    weighted sum of three components:
    1.  **Class Cost:** The cost of classifying a query into the wrong
        class.
    2.  **Bounding Box Cost:** The L1 distance between the predicted and
        ground truth bounding boxes.
    3.  **CIoU Cost:** The Complete Intersection over Union (CIoU) loss.

    Args:
        outputs: dict, A dictionary containing predicted `"logits"` and
            `"pred_boxes"`.
        targets: list of dict, A list of dictionaries, each containing
            the ground truth `"labels"` and `"boxes"`.
        num_targets_per_image: A tensor of shape `(batch_size,)` indicating
            the number of ground truth objects in each image.

    Returns:
        tuple: A tuple of three tensors `(row_indices, col_indices,
        valid_masks)`. `row_indices` and `col_indices` contain the indices
        of matched predictions and ground truths, while `valid_masks`
        indicates which matches are valid.
    """
    batch_size = keras.ops.shape(outputs["logits"])[0]
    num_queries = keras.ops.shape(outputs["logits"])[1]
    out_logits = outputs["logits"]
    out_bbox = outputs["pred_boxes"]
    target_ids_all = keras.ops.cast(targets[0]["labels"], dtype="int32")
    target_bbox_all = targets[0]["boxes"]
    target_offsets = keras.ops.concatenate(
        [
            keras.ops.zeros((1,), dtype="int32"),
            keras.ops.cumsum(num_targets_per_image),
        ]
    )
    max_matches = num_queries
    row_indices_init = keras.ops.zeros((batch_size, max_matches), dtype="int32")
    col_indices_init = keras.ops.zeros((batch_size, max_matches), dtype="int32")
    valid_masks_init = keras.ops.zeros((batch_size, max_matches), dtype="bool")

    def loop_body(i, loop_vars):
        row_indices, col_indices, valid_masks = loop_vars
        out_logits_i = out_logits[i]
        out_bbox_i = out_bbox[i]
        start = target_offsets[i]
        end = target_offsets[i + 1]
        num_targets_i = end - start
        k = keras.ops.arange(0, num_queries)
        is_valid_target_mask = k < num_targets_i
        target_indices = start + k
        safe_target_indices = keras.ops.minimum(
            target_indices, keras.ops.shape(target_ids_all)[0] - 1
        )
        target_ids_i = keras.ops.take(
            target_ids_all, safe_target_indices, axis=0
        )
        target_bbox_i = keras.ops.take(
            target_bbox_all, safe_target_indices, axis=0
        )

        def compute_cost_matrix():
            if use_focal_loss:
                out_prob_i = keras.ops.sigmoid(out_logits_i)
                safe_ids_for_take = keras.ops.maximum(target_ids_i, 0)
                prob_for_target_classes = keras.ops.take(
                    out_prob_i, safe_ids_for_take, axis=1
                )
                p = prob_for_target_classes
                pos_cost = (
                    matcher_alpha
                    * keras.ops.power(1 - p, matcher_gamma)
                    * (-keras.ops.log(p + 1e-8))
                )
                neg_cost = (
                    (1 - matcher_alpha)
                    * keras.ops.power(p, matcher_gamma)
                    * (-keras.ops.log(1 - p + 1e-8))
                )
                class_cost_i = pos_cost - neg_cost
            else:
                out_prob_softmax_i = keras.ops.softmax(out_logits_i, axis=-1)
                safe_ids_for_take = keras.ops.maximum(target_ids_i, 0)
                prob_for_target_classes = keras.ops.take(
                    out_prob_softmax_i, safe_ids_for_take, axis=1
                )
                class_cost_i = -prob_for_target_classes

            bbox_cost_i = keras.ops.sum(
                keras.ops.abs(
                    keras.ops.expand_dims(out_bbox_i, 1)
                    - keras.ops.expand_dims(target_bbox_i, 0)
                ),
                axis=2,
            )
            out_bbox_corners_i = keras.utils.bounding_boxes.convert_format(
                out_bbox_i,
                source="center_xywh",
                target="xyxy",
            )
            target_bbox_corners_i = keras.utils.bounding_boxes.convert_format(
                target_bbox_i,
                source="center_xywh",
                target="xyxy",
            )
            ciou_cost_i = -keras.utils.bounding_boxes.compute_ciou(
                keras.ops.expand_dims(out_bbox_corners_i, 1),
                keras.ops.expand_dims(target_bbox_corners_i, 0),
                bounding_box_format="xyxy",
            )

            cost_matrix_i = (
                matcher_bbox_cost * bbox_cost_i
                + matcher_class_cost * class_cost_i
                + matcher_ciou_cost * ciou_cost_i
            )
            cost_matrix_i = keras.ops.where(
                keras.ops.expand_dims(is_valid_target_mask, 0),
                cost_matrix_i,
                1e9,
            )
            return cost_matrix_i

        def perform_assignment():
            cost_matrix_i = compute_cost_matrix()
            row_idx, col_idx, valid_mask = hungarian_assignment(
                cost_matrix_i, backbone.num_queries
            )
            valid_mask = keras.ops.logical_and(
                valid_mask, col_idx < num_targets_i
            )
            return row_idx, col_idx, valid_mask

        def skip_assignment():
            return (
                keras.ops.zeros((num_queries,), dtype="int32"),
                keras.ops.zeros((num_queries,), dtype="int32"),
                keras.ops.zeros((num_queries,), dtype="bool"),
            )

        row_idx, col_idx, valid_mask = keras.ops.cond(
            keras.ops.greater(num_targets_i, 0),
            perform_assignment,
            skip_assignment,
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
        return row_indices, col_indices, valid_masks

    row_indices, col_indices, valid_masks = keras.ops.fori_loop(
        0,
        batch_size,
        loop_body,
        (row_indices_init, col_indices_init, valid_masks_init),
    )
    return (row_indices, col_indices, valid_masks)


def compute_vfl_loss(
    outputs,
    targets,
    indices,
    num_boxes,
    num_classes,
    matcher_alpha,
    matcher_gamma,
):
    """Computes the Varifocal Loss (VFL) for classification.

    VFL is an asymmetric focal loss variant designed for dense object
    detection. It treats the Intersection over Union (IoU) between a
    predicted box and its matched ground truth box as the target score for
    positive examples while down-weighting the loss for negative examples.
    This helps the model focus on high-quality localizations.

    Args:
        outputs: dict, A dictionary containing the model's predictions,
            including `"logits"` and `"pred_boxes"`.
        targets: list of dict, A list of dictionaries containing ground
            truth `"labels"` and `"boxes"`.
        indices: tuple, `(row_ind, col_ind, valid_mask)` from the
            Hungarian matcher, indicating the assignments between
            predictions and targets.
        num_boxes: int, The total number of ground truth boxes in the batch,
            used for normalization.

    Returns:
        Dictionary: The computed VFL loss.
    """
    _, col_indices, valid_masks = indices
    batch_idx, src_idx = _get_source_permutation_idx(indices)
    src_boxes = gather_along_first_two_dims(
        outputs["pred_boxes"], batch_idx, src_idx
    )
    flat_col_indices = keras.ops.reshape(col_indices, (-1,))
    flat_valid_masks = keras.ops.reshape(valid_masks, (-1,))
    src_logits = outputs["logits"]
    target_classes_init = keras.ops.full(
        shape=keras.ops.shape(src_logits)[:2],
        fill_value=num_classes,
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
            target_boxes_tensor = keras.ops.squeeze(target_boxes_tensor, axis=1)
        flat_target_labels = keras.ops.reshape(target_labels_tensor, (-1,))
        flat_target_boxes = keras.ops.reshape(target_boxes_tensor, (-1, 4))
        num_targets = keras.ops.shape(flat_target_labels)[0]
        num_targets = keras.ops.cast(num_targets, dtype=flat_col_indices.dtype)
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
            flat_valid_masks, target_classes_flat, num_classes
        )
        target_boxes_flat = keras.ops.where(
            keras.ops.expand_dims(flat_valid_masks, axis=-1),
            target_boxes_flat,
            0.0,
        )
        src_boxes_corners = keras.utils.bounding_boxes.convert_format(
            keras.ops.stop_gradient(src_boxes),
            source="center_xywh",
            target="xyxy",
        )
        target_boxes_corners = keras.utils.bounding_boxes.convert_format(
            target_boxes_flat,
            source="center_xywh",
            target="xyxy",
        )
        ious_matrix = keras.utils.bounding_boxes.compute_iou(
            src_boxes_corners,
            target_boxes_corners,
            bounding_box_format="xyxy",
        )
        ious = keras.ops.diagonal(ious_matrix)
        ious = ious * keras.ops.cast(flat_valid_masks, dtype=ious.dtype)
        target_classes_flat = keras.ops.cast(target_classes_flat, dtype="int32")
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
        target_classes, num_classes=num_classes + 1
    )[..., :-1]
    target_score = (
        keras.ops.expand_dims(target_score_original, axis=-1) * target_one_hot
    )
    pred_score_sigmoid = keras.ops.sigmoid(keras.ops.stop_gradient(src_logits))
    weight = (
        matcher_alpha
        * keras.ops.power(pred_score_sigmoid, matcher_gamma)
        * (1 - target_one_hot)
        + target_score
    )
    loss_vfl = keras.ops.binary_crossentropy(
        target_score, src_logits, from_logits=True
    )
    loss_vfl = loss_vfl * weight
    loss_vfl = (
        keras.ops.sum(keras.ops.mean(loss_vfl, axis=1))
        * keras.ops.cast(keras.ops.shape(src_logits)[1], dtype=loss_vfl.dtype)
        / num_boxes
    )
    return {"loss_vfl": loss_vfl}


def compute_box_losses(outputs, targets, indices, num_boxes):
    """Computes the bounding box regression losses.

    This function calculates two losses for the bounding boxes that were
    successfully matched to ground truth objects by the Hungarian matcher:
    1.  **L1 Loss (`loss_bbox`):** A regression loss that measures the
        absolute difference between the predicted and ground truth box
        coordinates.
    2.  **Complete IoU Loss (`loss_ciou`):** A scale-invariant loss that
        accounts for the shape and orientation of the boxes, providing a
        better gradient signal than the standard IoU, especially for
        non-overlapping boxes.

    Args:
        outputs: dict, A dictionary containing predicted `"pred_boxes"`.
        targets: list of dict, A list of dictionaries containing ground
            truth `"boxes"`.
        indices: tuple, The assignments from the Hungarian matcher.
        num_boxes: int, The total number of ground truth boxes for
            normalization.

    Returns:
        Dictionary: A dictionary containing the L1 and CIoU losses.
    """
    _, col_indices, valid_masks = indices
    batch_idx, src_idx = _get_source_permutation_idx(indices)
    src_boxes = gather_along_first_two_dims(
        outputs["pred_boxes"], batch_idx, src_idx
    )
    target_boxes_all = targets[0]["boxes"]
    if keras.ops.ndim(target_boxes_all) == 3:
        target_boxes_all = keras.ops.squeeze(target_boxes_all, axis=0)
    col_indices_flat = keras.ops.reshape(col_indices, [-1])
    valid_masks_flat = keras.ops.reshape(valid_masks, [-1])
    max_box_idx = keras.ops.maximum(keras.ops.shape(target_boxes_all)[0] - 1, 0)
    max_box_idx = keras.ops.cast(max_box_idx, dtype=col_indices_flat.dtype)
    safe_col_indices = keras.ops.clip(col_indices_flat, 0, max_box_idx)
    target_boxes = keras.ops.take(target_boxes_all, safe_col_indices, axis=0)
    valid_masks_expanded = keras.ops.expand_dims(valid_masks_flat, axis=-1)
    valid_masks_expanded = keras.ops.cast(
        valid_masks_expanded, target_boxes.dtype
    )
    target_boxes = target_boxes * valid_masks_expanded
    l1_loss = keras.ops.sum(
        keras.ops.abs(src_boxes - target_boxes)
        * keras.ops.cast(valid_masks_expanded, src_boxes.dtype)
    )
    src_boxes_xyxy = keras.utils.bounding_boxes.convert_format(
        src_boxes,
        source="center_xywh",
        target="xyxy",
    )
    target_boxes_xyxy = keras.utils.bounding_boxes.convert_format(
        target_boxes,
        source="center_xywh",
        target="xyxy",
    )
    ciou = keras.utils.bounding_boxes.compute_ciou(
        src_boxes_xyxy,
        target_boxes_xyxy,
        bounding_box_format="xyxy",
    )
    ciou_loss = keras.ops.sum(
        (1.0 - ciou) * keras.ops.cast(valid_masks_flat, src_boxes.dtype)
    )
    return {
        "loss_bbox": l1_loss / num_boxes,
        "loss_ciou": ciou_loss / num_boxes,
    }


def compute_local_losses(
    outputs,
    targets,
    indices,
    num_boxes,
    backbone,
    ddf_temperature,
    compute_ddf=None,
):
    """Computes local refinement losses (FGL and DDF).

    This function calculates two advanced losses for fine-grained box
    and feature refinement:
    1.  **Focal Grid Loss (`loss_fgl`):** This loss operates on the
        integral-based representation of the bounding box corners. It is a
        focal loss applied to the distribution over discrete bins,
        encouraging the model to produce sharp, unimodal distributions
        around the true corner locations.
    2.  **Distribution-guided Denoising Focal Loss (`loss_ddf`):** This is
        a knowledge distillation loss used for auxiliary decoder layers. It
        minimizes the KL-divergence between the corner prediction
        distribution of an intermediate layer (student) and that of the
        final decoder layer (teacher). This guides the intermediate layers
        to learn features that are consistent with the final, most refined
        predictions.

    Args:
        outputs: dict, A dictionary of model predictions, including
            `"pred_corners"`, `"ref_points"`, and potentially teacher
            predictions like `"teacher_corners"` and `"teacher_logits"`.
        targets: list of dict, A list of dictionaries with ground truth
            `"boxes"`.
        indices: tuple of Tensors, The assignments from the Hungarian
        matcher.
        num_boxes: scalar Tensor, The total number of ground truth boxes for
        normalization.
        compute_ddf: bool, Indicates whether to compute the DDF loss.

    Returns:
        Dictionary: A dictionary containing the computed FGL and DDF losses.
    """
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
    batch_idx, src_idx = _get_source_permutation_idx(indices)
    col_indices_flat = keras.ops.reshape(col_indices, [-1])
    valid_masks_flat = keras.ops.reshape(valid_masks, [-1])
    target_boxes_all = targets[0]["boxes"]
    if keras.ops.ndim(target_boxes_all) == 3:
        target_boxes_all = keras.ops.squeeze(target_boxes_all, axis=0)
    max_box_idx = keras.ops.maximum(keras.ops.shape(target_boxes_all)[0] - 1, 0)
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

    pred_corners_matched_flat = gather_along_first_two_dims(
        outputs["pred_corners"], batch_idx, src_idx
    )
    pred_corners_matched = keras.ops.reshape(
        pred_corners_matched_flat,
        (-1, backbone.decoder.max_num_bins + 1),
    )
    ref_points_matched = gather_along_first_two_dims(
        outputs["ref_points"], batch_idx, src_idx
    )
    ref_points_matched = keras.ops.stop_gradient(ref_points_matched)
    target_boxes_corners_matched = keras.utils.bounding_boxes.convert_format(
        target_boxes_matched_center,
        source="center_xywh",
        target="xyxy",
    )
    reg_scale_tensor = backbone.decoder.reg_scale
    up_tensor = backbone.decoder.upsampling_factor
    target_corners_dist, weight_right, weight_left = bbox2distance(
        ref_points_matched,
        target_boxes_corners_matched,
        backbone.decoder.max_num_bins,
        reg_scale_tensor,
        up_tensor,
    )
    pred_boxes_matched_center = gather_along_first_two_dims(
        outputs["pred_boxes"], batch_idx, src_idx
    )
    pred_boxes_corners_matched = keras.utils.bounding_boxes.convert_format(
        pred_boxes_matched_center,
        source="center_xywh",
        target="xyxy",
    )
    ious_pairwise = keras.utils.bounding_boxes.compute_iou(
        pred_boxes_corners_matched,
        target_boxes_corners_matched,
        bounding_box_format="xyxy",
    )
    ious = keras.ops.diagonal(ious_pairwise)
    ious = ious * keras.ops.cast(valid_masks_flat, dtype=ious.dtype)
    weight_targets_fgl = keras.ops.reshape(
        keras.ops.tile(keras.ops.expand_dims(ious, 1), [1, 4]),
        [-1],
    )
    weight_targets_fgl = keras.ops.stop_gradient(weight_targets_fgl)
    losses["loss_fgl"] = unimodal_distribution_focal_loss(
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
            (-1, backbone.decoder.max_num_bins + 1),
        )
        target_corners_all = keras.ops.reshape(
            keras.ops.stop_gradient(outputs["teacher_corners"]),
            (-1, backbone.decoder.max_num_bins + 1),
        )

        def compute_ddf_loss_fn():
            weight_targets_local = keras.ops.max(
                keras.ops.sigmoid(outputs["teacher_logits"]), axis=-1
            )
            num_queries = keras.ops.cast(
                keras.ops.shape(weight_targets_local)[1],
                dtype=batch_idx.dtype,
            )
            flat_update_indices = batch_idx * num_queries + src_idx
            flat_update_indices = keras.ops.expand_dims(
                flat_update_indices, axis=-1
            )
            mask = keras.ops.zeros_like(weight_targets_local, dtype="bool")
            mask_flat = keras.ops.scatter_update(
                keras.ops.reshape(mask, (-1,)),
                flat_update_indices,
                keras.ops.ones_like(batch_idx, dtype="bool"),
            )
            mask = keras.ops.reshape(
                mask_flat, keras.ops.shape(weight_targets_local)
            )
            weight_targets_local_flat = keras.ops.reshape(
                weight_targets_local, (-1,)
            )
            weight_targets_local_matched_flat = keras.ops.scatter_update(
                weight_targets_local_flat,
                flat_update_indices,
                ious,
            )
            weight_targets_local = keras.ops.reshape(
                weight_targets_local_matched_flat,
                keras.ops.shape(weight_targets_local),
            )
            weight_targets_local_expanded = keras.ops.reshape(
                keras.ops.tile(
                    keras.ops.expand_dims(weight_targets_local, axis=-1),
                    [1, 1, 4],
                ),
                [-1],
            )
            weight_targets_local_expanded = keras.ops.stop_gradient(
                weight_targets_local_expanded
            )
            # NOTE: Original impl hardcodes `ddf_temperature` to 5.0 for
            # DDFL.
            # KerasHub lets users configure it if needed.
            # Ref: https://github.com/huggingface/transformers/blob/b374c3d12e8a42014b7911d1bddf598aeada1154/src/transformers/loss/loss_d_fine.py#L238
            pred_softmax = keras.ops.softmax(
                pred_corners_all / ddf_temperature, axis=-1
            )
            target_softmax = keras.ops.softmax(
                target_corners_all / ddf_temperature, axis=-1
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
                weight_targets_local_expanded * (ddf_temperature**2) * kl_div
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
                    * keras.ops.cast(neg_mask_flat, loss_match_local.dtype)
                )
                / keras.ops.sum(
                    keras.ops.cast(neg_mask_flat, loss_match_local.dtype)
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
                loss_match_local1 * num_pos + loss_match_local2 * num_neg
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
        return keras.ops.convert_to_tensor(0.0, dtype=keras.backend.floatx())

    losses["loss_ddf"] = keras.ops.cond(compute_ddf, ddf_true_fn, ddf_false_fn)
    return losses


def _translate_gt_valid_case(
    gt_flat, valid_idx_mask, function_values, max_num_bins, mask
):
    closest_left_indices = (
        keras.ops.sum(keras.ops.cast(mask, "int32"), axis=1) - 1
    )
    indices_float = keras.ops.cast(closest_left_indices, dtype=gt_flat.dtype)
    weight_right = keras.ops.zeros_like(indices_float)
    weight_left = keras.ops.zeros_like(indices_float)
    valid_indices_int = keras.ops.arange(keras.ops.shape(valid_idx_mask)[0])
    valid_indices_int = keras.ops.where(valid_idx_mask, valid_indices_int, -1)
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
    left_values = keras.ops.take(function_values, valid_indices_long, axis=0)
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


def translate_gt(gt, max_num_bins, reg_scale, up):
    gt_flat = keras.ops.reshape(gt, [-1])
    function_values = weighting_function(max_num_bins, up, reg_scale)
    diffs = keras.ops.expand_dims(
        function_values, axis=0
    ) - keras.ops.expand_dims(gt_flat, axis=1)
    mask = diffs <= 0
    closest_left_indices = (
        keras.ops.sum(keras.ops.cast(mask, "int32"), axis=1) - 1
    )
    indices_float = keras.ops.cast(closest_left_indices, dtype=gt_flat.dtype)
    weight_right = keras.ops.zeros_like(indices_float)
    weight_left = keras.ops.zeros_like(indices_float)
    valid_idx_mask = (indices_float >= 0) & (indices_float < max_num_bins)
    return keras.ops.cond(
        keras.ops.any(valid_idx_mask),
        lambda: _translate_gt_valid_case(
            gt_flat, valid_idx_mask, function_values, max_num_bins, mask
        ),
        lambda: (
            keras.ops.zeros_like(indices_float),
            keras.ops.zeros_like(weight_right),
            keras.ops.ones_like(weight_left),
        ),
    )


def _compute_bbox2distance(points, bbox, max_num_bins, reg_scale, up, eps=0.1):
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
    four_lens_translated, weight_right, weight_left = translate_gt(
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


def bbox2distance(points, bbox, max_num_bins, reg_scale, up, eps=0.1):
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
        lambda: _compute_bbox2distance(
            points, bbox, max_num_bins, reg_scale, up, eps
        ),
    )


def unimodal_distribution_focal_loss(
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


def _get_source_permutation_idx(indices):
    """Gathers the batch and source indices for matched predictions.

    This method is a JAX-compatible adaptation of the author's approach,
    which creates dynamically sized tensors by concatenating indices from a
    list, which is not traceable by a JIT compiler.

    To ensure JAX compatibility, this implementation uses a masking
    strategy. It returns fixed-size tensors where invalid positions are
    padded with `0`. The downstream loss functions then use the
    `valid_masks` tensor to ignore these padded entries during loss
    computation.
    """
    row_indices, _, valid_masks = indices
    batch_size = keras.ops.shape(row_indices)[0]
    max_matches = keras.ops.shape(row_indices)[1]
    batch_indices = keras.ops.arange(batch_size, dtype="int32")
    batch_indices = keras.ops.expand_dims(batch_indices, axis=1)
    batch_indices = keras.ops.tile(batch_indices, [1, max_matches])
    batch_indices_flat = keras.ops.reshape(batch_indices, (-1,))
    row_indices_flat = keras.ops.reshape(row_indices, (-1,))
    valid_masks_flat = keras.ops.reshape(valid_masks, (-1,))
    batch_idx = keras.ops.where(
        valid_masks_flat,
        keras.ops.cast(batch_indices_flat, "int64"),
        0,
    )
    src_idx = keras.ops.where(
        valid_masks_flat,
        keras.ops.cast(row_indices_flat, dtype="int64"),
        0,
    )
    return batch_idx, src_idx


def get_cdn_matched_indices(dn_meta):
    """Generates matched indices for contrastive denoising (CDN) training.

    This method is a JAX-compatible adaptation of the author's approach,
    which iterates through the batch to build a list of dynamically sized
    index tensors, which is not traceable by a JIT compiler.

    To ensure JAX compatibility, this implementation operates on the entire
    batch as a single tensor operation. It uses the pre-padded
    `dn_positive_idx` tensor (where -1 indicates padding) to generate
    fixed-size `row_indices`, `col_indices`, and a `valid_masks` tensor.
    """
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
