import keras
from keras import ops

from qwen2_5_vl_vision_block import Qwen2_5_VLVisionBlock
from qwen2_5_vl_rms_norm import Qwen2_5_VLRMSNorm


@keras.saving.register_keras_serializable(package="keras_hub")
class Qwen2_5_VLVisionEncoder(keras.layers.Layer):
    """
    Qwen2.5-VL Vision Encoder

    Architecture:
        Conv3D Patch Embedding
        Hybrid Vision Transformer Blocks
        Final RMSNorm

    Returns
    -------
    dict:
        vision_tokens : [B, N, C]
        grid_thw : [3] tensor containing (T, H, W)
    """

    def __init__(
        self,
        hidden_size,
        num_layers,
        num_heads,
        intermediate_size,
        patch_size=14,
        temporal_patch_size=2,
        window_size=8,
        global_layers=None,
        theta=10000.0,
        rms_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.window_size = window_size
        self.theta = theta
        self.rms_eps = rms_eps

        if global_layers is None:
            global_layers = []
        self.global_layers = list(global_layers)

        # ---------------------------------------------------
        # Spatiotemporal Patch Embedding
        # ---------------------------------------------------
        self.patch_embed = keras.layers.Conv3D(
            filters=hidden_size,
            kernel_size=(
                temporal_patch_size,
                patch_size,
                patch_size,
            ),
            strides=(
                temporal_patch_size,
                patch_size,
                patch_size,
            ),
            padding="valid",
            use_bias=False,
            name="patch_embed",
        )

        # ---------------------------------------------------
        # Hybrid Vision Blocks
        # ---------------------------------------------------
        self.blocks = [
            Qwen2_5_VLVisionBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                window_size=window_size,
                use_global_attention=(i in self.global_layers),
                theta=theta,
                rms_eps=rms_eps,
                name=f"vision_block_{i}",
            )
            for i in range(num_layers)
        ]

        # ---------------------------------------------------
        # Final Norm
        # ---------------------------------------------------
        self.final_norm = Qwen2_5_VLRMSNorm(
            hidden_size,
            epsilon=rms_eps,
            name="vision_final_norm",
        )

    def call(self, images, training=False):
        """
        images: [B, T, H, W, 3]
        """

        # ---------------------------------------------------
        # Patch Embedding
        # ---------------------------------------------------
        x = self.patch_embed(images)

        shape = ops.shape(x)

        B = shape[0]
        T = shape[1]
        H = shape[2]
        W = shape[3]
        C = shape[4]

        # ---------------------------------------------------
        # Flatten patches → tokens
        # ---------------------------------------------------
        tokens = ops.reshape(x, (B, T * H * W, C))

        # ---------------------------------------------------
        # Hybrid Transformer Blocks
        # ---------------------------------------------------
        for block in self.blocks:
            tokens = block(
                tokens,
                T=T,
                H=H,
                W=W,
                training=training,
            )

        tokens = self.final_norm(tokens)

        # ---------------------------------------------------
        # Return tensor-safe outputs
        # ---------------------------------------------------
        grid = ops.stack([T, H, W])

        return {
            "vision_tokens": tokens,
            "grid_thw": grid,
        }

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "patch_size": self.patch_size,
                "temporal_patch_size": self.temporal_patch_size,
                "window_size": self.window_size,
                "global_layers": self.global_layers,
                "theta": self.theta,
                "rms_eps": self.rms_eps,
            }
        )

        return config