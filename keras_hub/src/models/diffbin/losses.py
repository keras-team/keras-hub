import keras
from keras import ops


class DiffBinLoss(keras.losses.Loss):
    def __init__(self, alpha=1.0, beta=10.0, name="diffbin_loss"):
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta
        self.bce = keras.losses.BinaryCrossentropy(
            from_logits=True, reduction=keras.losses.Reduction.NONE
        )

    def call(self, y_true, y_pred):
        prob_map_true, binary_map_true, thresh_map_true, dilated_mask = y_true
        prob_map_pred, binary_map_pred, thresh_map_pred = y_pred

        ls = self.hard_negative_mining_bce(prob_map_true, prob_map_pred)
        lb = self.hard_negative_mining_bce(thresh_map_true, thresh_map_pred)
        lt = self.threshold_map_loss(
            binary_map_true, binary_map_pred, dilated_mask
        )
        total_loss = ls + (self.alpha * lb) + (self.beta * lt)
        return total_loss

    def hard_negative_mining_bce(self, y_true, y_pred):
        y_true_flat = ops.flatten(y_true)
        y_pred_flat = ops.flatten(y_pred)

        pixel_losses = self.bce(y_true_flat, y_pred_flat)

        # Identify positive and negative pixels
        positive_mask = ops.cast(y_true_flat > 0.5, dype=y_pred_flat.dtype)
        negative_mask = 1 - positive_mask
        positive_mask = ops.cast(y_true_flat > 0.5, dtype=y_pred_flat.dtype)
        negative_mask = 1.0 - positive_mask

        # Get losses for positive pixels
        actual_positive_losses = ops.where(
            positive_mask > 0.5,
            pixel_losses,
            ops.convert_to_tensor(0.0, dtype=pixel_losses.dtype),
        )
        actual_positive_losses = ops.reshape(
            actual_positive_losses[actual_positive_losses > 0], [-1]
        )

        no_positives = ops.sum(positive_mask)
        no_negatives = ops.sum(negative_mask)
        if ops.equal(no_positives, 0):
            k_neg = ops.cast(
                ops.minimum(ops.convert_to_tensor(32.0), no_negatives),
                dtype="int32",
            )
        else:
            k_neg = ops.cast(no_positives * 3, dtype="int32")
            k_neg = ops.minimum(k_neg, ops.cast(no_negatives, dtype="int32"))

        # Get hard negative losses
        # Sort indices based on masked_negative_losses in descending order
        # Select the top_k_neg indices
        # Gather the hard negative losses

        masked_negative_losses = ops.where(
            negative_mask > 0.5, pixel_losses, ops.full_like(pixel_losses, -1e9)
        )
        sorted_indices = ops.argsort(masked_negative_losses, axis=-1)
        hard_negative_indices = sorted_indices[:k_neg]
        hard_negative_losses = ops.gather(pixel_losses, hard_negative_indices)
        # Combine positive losses and hard negative losses
        actual_positive_losses = ops.where(
            positive_mask > 0.5, pixel_losses, ops.full_like(pixel_losses, 0.0)
        )
        actual_positive_losses = ops.reshape(
            actual_positive_losses[actual_positive_losses > 0], [-1]
        )

        if ops.equal(no_negatives, 0):
            sampled_losses = hard_negative_losses
        else:
            sampled_losses = ops.concatenate(
                [actual_positive_losses, hard_negative_losses], axis=0
            )

        if ops.size(sampled_losses) > 0:
            return ops.mean(sampled_losses)
        else:
            return ops.convert_to_tensor(0.0, dtype=y_pred_flat.dtype)

    def threshold_map_loss(self, y_true, y_pred, dilated_mask):
        dilated_mask_float = ops.cast(dilated_mask, dtype=y_pred.dtype)
        l1_diff = ops.abs(y_true - y_pred)
        # Apply the mask: only count losses where mask is 1
        masked_l1_diff = l1_diff * dilated_mask_float

        num_active_pixels = ops.sum(dilated_mask_float)
        if ops.equal(num_active_pixels, 0):
            return ops.convert_to_tensor(0.0, dtype=y_pred.dtype)
        else:
            return ops.sum(masked_l1_diff) / num_active_pixels
