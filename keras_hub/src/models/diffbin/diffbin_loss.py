import keras
from keras import ops


class DiffBinLoss(keras.losses.Loss):
    def __init__(self, alpha=1.0, beta=10.0, name="diffbin_loss"):
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-7

    def call(self, y_true, y_pred):
        prob_map_true = y_true[..., 0:1]  # Channel 0
        thresh_map_true= y_true[..., 1:2]  # Channel 1
        binary_map_true = y_true[..., 2:3]  # Channel 2
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
        """
        Computes the hard negative mining binary cross-entropy loss.
        This function applies hard negative mining to the binary cross-entropy
        loss for the given true and predicted values.
        (https://medium.com/@sundardell955/hard-negative-mining-91b5792259c5)
        Args:
            y_true: Tensor, ground truth values.
            y_pred: Tensor, predicted values.
        Returns:
            A scalar tensor representing the hard negative mining binary
            cross-entropy loss.
        """
        y_pred_clipped = ops.clip(y_pred, self.eps, 1.0 - self.eps)
        pixel_losses = -(
            y_true * ops.log(y_pred_clipped)
            + (1.0 - y_true) * ops.log(1.0 - y_pred_clipped)
        )

        pixel_losses = ops.reshape(pixel_losses, [-1])
        y_true_flat = ops.reshape(y_true, [-1])

        pos_mask = ops.cast(y_true_flat > 0.5, dtype=pixel_losses.dtype)
        neg_mask = 1.0 - pos_mask

        num_pos = ops.sum(pos_mask)
        num_neg = ops.sum(neg_mask)

        k_neg = ops.cond(
            ops.equal(num_pos, 0),
            lambda: ops.cast(
                ops.minimum(ops.convert_to_tensor(32), num_neg), dtype="int32"
            ),
            lambda: ops.cast(ops.minimum(num_pos * 3, num_neg), dtype="int32"),
        )

        pos_idx = ops.reshape(ops.where(pos_mask > 0.5), [-1])
        pos_loss = ops.take(pixel_losses, pos_idx)

        neg_loss_masked = ops.where(
            neg_mask > 0.5,
            pixel_losses,
            ops.convert_to_tensor(-1e9, dtype=pixel_losses.dtype),
        )
        top_k_vals, _ = ops.top_k(neg_loss_masked, k=k_neg)

        sampled = ops.cond(
            ops.equal(num_pos, 0),
            lambda: top_k_vals,
            lambda: ops.concatenate([pos_loss, top_k_vals], axis=0),
        )

        return ops.cond(
            ops.size(sampled) > 0,
            lambda: ops.mean(sampled),
            lambda: ops.convert_to_tensor(0.0, dtype=pixel_losses.dtype),
        )

    def threshold_map_loss(self, y_true, y_pred, dilated_mask):
        """
        Computes the threshold map loss.
        This function calculates the pixel-wise L1 loss between the true and
        predicted values, weighted by a dilated mask.
        Args:
            y_true: Tensor, ground truth values.
            y_pred: Tensor, predicted values.
            dilated_mask: Tensor, a mask used to weight the loss.
        Returns:
            A scalar tensor representing the threshold map loss.
        """
        mask_f = ops.cast(dilated_mask, dtype=y_pred.dtype)
        l1 = ops.abs(y_true - y_pred) * mask_f
        n_pix = ops.sum(mask_f)

        return ops.cond(
            ops.equal(n_pix, 0),
            lambda: ops.convert_to_tensor(0.0, dtype=y_pred.dtype),
            lambda: ops.sum(l1) / n_pix,
        )

