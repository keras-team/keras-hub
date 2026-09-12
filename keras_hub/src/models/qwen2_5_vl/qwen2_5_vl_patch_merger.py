import keras
from keras import ops


@keras.saving.register_keras_serializable(package="keras_hub")
class Qwen2_5_VLPatchMerger(keras.layers.Layer):
    """
    Spatial 2x2 patch merger for Qwen2.5-VL.

    Input
    -----
    tokens : [B, N, C]
    T,H,W  : grid sizes

    Output
    ------
    [B, T*(H/2)*(W/2), 4C]
    """

    def call(self, tokens, *, T, H, W):

        B = ops.shape(tokens)[0]
        C = ops.shape(tokens)[-1]

        # restore grid
        x = ops.reshape(tokens, (B, T, H, W, C))

        # group 2×2 spatial patches
        x = ops.reshape(
            x,
            (
                B,
                T,
                H // 2,
                2,
                W // 2,
                2,
                C,
            ),
        )

        # reorder
        x = ops.transpose(x, (0, 1, 2, 4, 3, 5, 6))

        # merge channels
        x = ops.reshape(
            x,
            (
                B,
                T * (H // 2) * (W // 2),
                4 * C,
            ),
        )

        return x

    def get_config(self):
        return super().get_config()