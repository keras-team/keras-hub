import keras
from keras import ops


class DiceLoss:
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps

    def __call__(self, y_true, y_pred, mask, weights=None):
        if weights is not None:
            mask = weights * mask
        intersection = ops.sum((y_pred * y_true * mask))
        union = ops.sum((y_pred * mask)) + ops.sum(y_true * mask) + self.eps
        loss = 1 - 2.0 * intersection / union
        return loss


class MaskL1Loss:
    def __init__(self, **kwargs):
        pass

    def __call__(self, y_true, y_pred, mask):
        mask_sum = ops.sum(mask)
        loss = ops.where(
            mask_sum == 0.0,
            0.0,
            ops.sum(ops.absolute(y_pred - y_true) * mask) / mask_sum,
        )
        return loss


class BalanceCrossEntropyLoss:
    def __init__(self, negative_ratio=3.0, eps=1e-6, **kwargs):
        self.negative_ratio = negative_ratio
        self.eps = eps

    def __call__(self, y_true, y_pred, mask, return_origin=False):
        positive = ops.cast((y_true > 0.5) & ops.cast(mask, "bool"), "uint8")
        negative = ops.cast((y_true < 0.5) & ops.cast(mask, "bool"), "uint8")
        positive_count = ops.sum(ops.cast(positive, "int32"))
        negative_count = ops.sum(ops.cast(negative, "int32"))
        negative_count_max = ops.cast(
            ops.cast(positive_count, "float32") * self.negative_ratio, "int32"
        )

        negative_count = ops.where(
            negative_count > negative_count_max,
            negative_count_max,
            negative_count,
        )
        # Keras' losses reduce some axis. Since we don't want that here, we add
        # a dummy dimension to y_true and y_pred
        loss = keras.losses.BinaryCrossentropy(
            from_logits=False,
            label_smoothing=0.0,
            axis=-1,
            reduction=None,
        )(y_true=y_true[..., None], y_pred=y_pred[..., None])

        positive_loss = loss * ops.cast(positive, "float32")
        negative_loss = loss * ops.cast(negative, "float32")

        # hard negative mining, as suggested in the paper:
        # compute the threshold for hard negatives, and zero-out
        # negative losses below the threshold. using this approach,
        # we achieve efficient computation on GPUs

        # compute negative_count relative to the element count of y_pred
        negative_count_rel = ops.cast(negative_count, "float32") / ops.prod(
            ops.shape(negative_count)
        )
        # compute the threshold value for negative losses and zero neg. loss
        # values below this threshold
        negative_loss_thresh = ops.quantile(
            negative_loss, 1.0 - negative_count_rel
        )
        negative_loss = negative_loss * ops.cast(
            negative_loss > negative_loss_thresh, "float32"
        )

        balance_loss = (ops.sum(positive_loss) + ops.sum(negative_loss)) / (
            ops.cast(positive_count + negative_count, "float32") + self.eps
        )

        if return_origin:
            return balance_loss, loss
        return balance_loss


class DBLoss(keras.losses.Loss):
    def __init__(self, eps=1e-6, l1_scale=10.0, bce_scale=5.0, **kwargs):
        super().__init__(*kwargs)
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss()
        self.bce_loss = BalanceCrossEntropyLoss()

        self.l1_scale = l1_scale
        self.bce_scale = bce_scale

    def call(self, y_true, y_pred):
        p_map_pred, t_map_pred, b_map_pred = ops.unstack(y_pred, 3, axis=-1)
        shrink_map, shrink_mask, thresh_map, thresh_mask = ops.unstack(
            y_true, 4, axis=-1
        )

        # we here implement L1BalanceCELoss from PyTorch's
        # Differential Binarization implementation
        Ls = self.bce_loss(
            y_true=shrink_map,
            y_pred=p_map_pred,
            mask=shrink_mask,
            return_origin=False,
        )
        Lt = self.l1_loss(
            y_true=thresh_map,
            y_pred=t_map_pred,
            mask=thresh_mask,
        )
        dice_loss = self.dice_loss(
            y_true=shrink_map,
            y_pred=b_map_pred,
            mask=shrink_mask,
        )
        loss = dice_loss + self.l1_scale * Lt + Ls * self.bce_scale
        return loss
