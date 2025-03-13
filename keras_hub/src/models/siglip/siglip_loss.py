import keras
from keras import ops


class SigLIPLoss(keras.losses.Loss):
    """SigLIP Loss.

    SigLIP loss replaces the loss function used in CLIP by a simple pairwise
    sigmoid loss. Unlike standard contrastive learning with softmax
    normalization, the sigmoid loss operates solely on image-text pairs and does
    not require a global view of the pairwise similarities for normalization.
    The sigmoid loss simultaneously allows further scaling up the batch size,
    while also performing better at smaller batch sizes.

    References:
        - [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)
    """

    def call(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)

        # Ref: https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py
        logits = y_pred
        m1_diag1 = y_true

        # Standard sigmoid computes everything twice, once assuming positive
        # labels and once assuming negative ones. But here we know exactly where
        # to find positives (on "me" diagonal) and negatives (everywhere else),
        # so compute each one's loss only once:
        loglike = ops.nn.log_sigmoid(m1_diag1 * logits)

        # Normalize by npos per column, but that's one, so just sum.
        nll = -ops.sum(loglike, axis=-1)
        return nll
