import keras
from keras import ops


class DiffBinLoss(keras.losses.Loss):
    def __init__(self, alpha=1.0, beta=10.0, name="diffbin_loss"):
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta
        self.bce = keras.losses.BinaryCrossentropy(
            from_logits=False,
            label_smoothing=0.0,
            axis=-1,
            reduction=None,
        )

    def call(self, y_true, y_pred):
        prob_map_true = y_true[..., 0:1]  # Channel 0
        binary_map_true = y_true[..., 1:2]  # Channel 1
        thresh_map_true = y_true[..., 2:3]  # Channel 2
        dilated_mask = y_true[..., 3:4]  # Channel 3

        prob_map_pred = y_pred[..., 0:1]  # Channel 0 - probability maps
        thresh_map_pred = y_pred[..., 1:2]  # Channel 1 - threshold maps
        binary_map_pred = y_pred[..., 2:3]

        ls = self.hard_negative_mining_bce(prob_map_true, prob_map_pred)
        lb = self.hard_negative_mining_bce(thresh_map_true, thresh_map_pred)
        lt = self.threshold_map_loss(
            binary_map_true, binary_map_pred, dilated_mask
        )
        total_loss = ls + (self.alpha * lb) + (self.beta * lt)
        return total_loss

    def hard_negative_mining_bce(self, y_true, y_pred):
        y_true_flat = ops.reshape(y_true, [-1])
        y_pred_flat = ops.reshape(y_pred, [-1])

        pixel_losses = self.bce(y_true_flat, y_pred_flat)

        # Identify positive and negative pixels
        positive_mask = ops.cast(y_true_flat > 0.5, dtype=y_pred_flat.dtype)
        negative_mask = 1 - positive_mask

        no_positives = ops.sum(positive_mask)
        no_negatives = ops.sum(negative_mask)
        k_neg = ops.cond(
            ops.equal(no_positives, 0),
            lambda: ops.cast(
                ops.minimum(ops.convert_to_tensor(32.0), no_negatives),
                dtype="int32",
            ),
            lambda: ops.cast(
                ops.minimum(no_positives * 3, no_negatives), dtype="int32"
            ),
        )
        positive_losses = ops.where(
            positive_mask > 0.5,
            pixel_losses,
            ops.convert_to_tensor(0.0, dtype=pixel_losses.dtype),
        )
        positive_indices = ops.where(positive_mask > 0.5)
        actual_positive_losses = ops.take(pixel_losses, 
                                          ops.reshape(positive_indices, [-1]))

        # Get hard negative losses
        # Sort indices based on masked_negative_losses in descending order
        # Select the top_k_neg indices
        # Gather the hard negative losses
        masked_negative_losses = ops.where(
            negative_mask > 0.5,
            pixel_losses,
            ops.convert_to_tensor(-1e9, dtype=pixel_losses.dtype),
        )
        top_k_values, top_k_indices = ops.top_k(masked_negative_losses, k=k_neg)
        hard_negative_losses = top_k_values
        sampled_losses = ops.cond(
            ops.equal(no_positives, 0),
            lambda: hard_negative_losses,
            lambda: ops.concatenate(
                [actual_positive_losses, hard_negative_losses], axis=0
            ),
        )
        return ops.cond(
            ops.size(sampled_losses) > 0,
            lambda: ops.mean(sampled_losses),
            lambda: ops.convert_to_tensor(0.0, dtype=y_pred_flat.dtype),
        )

    def threshold_map_loss(self, y_true, y_pred, dilated_mask):
        dilated_mask_float = ops.cast(dilated_mask, dtype=y_pred.dtype)
        l1_diff = ops.abs(y_true - y_pred)

        # Apply the mask: only count losses where mask is 1
        masked_l1_diff = l1_diff * dilated_mask_float

        num_active_pixels = ops.sum(dilated_mask_float)

        return ops.cond(
            ops.equal(num_active_pixels, 0),
            lambda: ops.convert_to_tensor(0.0, dtype=y_pred.dtype),
            lambda: ops.sum(masked_l1_diff) / num_active_pixels,
        )
