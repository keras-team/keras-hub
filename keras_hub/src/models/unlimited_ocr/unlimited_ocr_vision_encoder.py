import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.clip.clip_layers import CLIPEncoderLayer
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_layers import (
    UnlimitedOCRCLIPEmbedding,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_layers import (
    UnlimitedOCRSamPositionEmbedding,
)
from keras_hub.src.models.vit_det.vit_layers import ViTDetPatchingAndEmbedding
from keras_hub.src.models.vit_det.vit_layers import WindowedTransformerEncoder


@keras_hub_export("keras_hub.models.UnlimitedOCRVisionEncoder")
class UnlimitedOCRVisionEncoder(keras.Model):
    """The Unlimited-OCR vision tower: a SAM encoder followed by a CLIP encoder.

    The SAM encoder turns an image into a feature map. A neck and two strided
    convolutions reduce the map by a factor of `4 * sam_patch_size`. The CLIP
    encoder then takes this feature map as its patch tokens. The layer
    concatenates the CLIP tokens (without the class token) and the SAM
    features on the channel axis. The output has
    `clip_hidden_dim + sam_downsample_channels[-1]` channels.

    The layer accepts any image size that the SAM windows and grids allow.
    It resizes the absolute and relative position tables to the input grid.
    The model uses one global view and local tiles of different sizes.

    Args:
        sam_hidden_dim: int. The width of the SAM blocks.
        sam_num_layers: int. The number of SAM blocks.
        sam_num_heads: int. The number of SAM attention heads.
        sam_mlp_dim: int. The intermediate size of the SAM MLP.
        sam_output_channels: int. The channels of the SAM neck.
        sam_patch_size: int. The patch size of the SAM patch embedding.
        sam_image_size: int. The image size that sizes the SAM position
            tables.
        sam_window_size: int. The window size of the SAM windowed blocks.
        sam_global_attn_indexes: tuple of int. The indexes of the SAM blocks
            with global attention.
        sam_downsample_channels: tuple of two ints. The channels of the two
            strided convolutions after the neck.
        clip_hidden_dim: int. The width of the CLIP blocks. This value must
            equal `sam_downsample_channels[-1]`.
        clip_num_layers: int. The number of CLIP blocks.
        clip_num_heads: int. The number of CLIP attention heads.
        clip_intermediate_dim: int. The intermediate size of the CLIP MLP.
        clip_image_size: int. The image size that sizes the CLIP position
            table.
        clip_patch_size: int. The patch size that sizes the CLIP position
            table.

    Example:
    ```python
    encoder = keras_hub.models.UnlimitedOCRVisionEncoder(
        sam_hidden_dim=16,
        sam_num_layers=2,
        sam_num_heads=2,
        sam_mlp_dim=32,
        sam_output_channels=8,
        sam_patch_size=8,
        sam_image_size=128,
        sam_window_size=4,
        sam_global_attn_indexes=(1,),
        sam_downsample_channels=(12, 16),
        clip_hidden_dim=16,
        clip_num_layers=2,
        clip_num_heads=2,
        clip_intermediate_dim=32,
        clip_image_size=48,
        clip_patch_size=16,
    )
    # A 128 pixel image gives a 16x16 patch grid and a 4x4 feature grid.
    features = encoder(np.ones((1, 128, 128, 3), dtype="float32"))
    features.shape  # (1, 16, 32)
    ```
    """

    def __init__(
        self,
        sam_hidden_dim=768,
        sam_num_layers=12,
        sam_num_heads=12,
        sam_mlp_dim=3072,
        sam_output_channels=256,
        sam_patch_size=16,
        sam_image_size=1024,
        sam_window_size=14,
        sam_global_attn_indexes=(2, 5, 8, 11),
        sam_downsample_channels=(512, 1024),
        clip_hidden_dim=1024,
        clip_num_layers=24,
        clip_num_heads=16,
        clip_intermediate_dim=4096,
        clip_image_size=224,
        clip_patch_size=14,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if clip_hidden_dim != sam_downsample_channels[-1]:
            raise ValueError(
                "`clip_hidden_dim` must equal `sam_downsample_channels[-1]`. "
                "The CLIP encoder takes the SAM features as patch tokens. "
                f"Received: clip_hidden_dim={clip_hidden_dim}, "
                f"sam_downsample_channels={sam_downsample_channels}"
            )
        self.sam_hidden_dim = sam_hidden_dim
        self.sam_num_layers = sam_num_layers
        self.sam_num_heads = sam_num_heads
        self.sam_mlp_dim = sam_mlp_dim
        self.sam_output_channels = sam_output_channels
        self.sam_patch_size = sam_patch_size
        self.sam_image_size = sam_image_size
        self.sam_window_size = sam_window_size
        self.sam_global_attn_indexes = tuple(sam_global_attn_indexes)
        self.sam_downsample_channels = tuple(sam_downsample_channels)
        self.clip_hidden_dim = clip_hidden_dim
        self.clip_num_layers = clip_num_layers
        self.clip_num_heads = clip_num_heads
        self.clip_intermediate_dim = clip_intermediate_dim
        self.clip_image_size = clip_image_size
        self.clip_patch_size = clip_patch_size
        self.output_dim = clip_hidden_dim + self.sam_downsample_channels[-1]

        # === SAM encoder ===
        sam_grid_size = sam_image_size // sam_patch_size
        self.sam_patch_embedding = ViTDetPatchingAndEmbedding(
            kernel_size=(sam_patch_size, sam_patch_size),
            strides=(sam_patch_size, sam_patch_size),
            embed_dim=sam_hidden_dim,
            dtype=self.dtype_policy,
            name="sam_patch_embedding",
        )
        self.sam_position_embedding = UnlimitedOCRSamPositionEmbedding(
            grid_size=sam_grid_size,
            hidden_dim=sam_hidden_dim,
            dtype=self.dtype_policy,
            name="sam_position_embedding",
        )
        # The relative position tables of a global block have
        # `2 * sam_grid_size - 1` rows. The attention resizes them to the
        # grid of the input. A windowed block uses `sam_window_size`.
        self.sam_blocks = [
            WindowedTransformerEncoder(
                project_dim=sam_hidden_dim,
                intermediate_dim=sam_mlp_dim,
                num_heads=sam_num_heads,
                use_bias=True,
                use_rel_pos=True,
                window_size=(
                    0 if i in self.sam_global_attn_indexes else sam_window_size
                ),
                input_size=(sam_grid_size, sam_grid_size),
                layer_norm_epsilon=1e-6,
                dtype=self.dtype_policy,
                name=f"sam_block_{i}",
            )
            for i in range(sam_num_layers)
        ]
        self.sam_neck_conv_1 = keras.layers.Conv2D(
            sam_output_channels,
            kernel_size=1,
            use_bias=False,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
            dtype=self.dtype_policy,
            name="sam_neck_conv_1",
        )
        self.sam_neck_norm_1 = keras.layers.LayerNormalization(
            epsilon=1e-6, dtype=self.dtype_policy, name="sam_neck_norm_1"
        )
        self.sam_neck_conv_2 = keras.layers.Conv2D(
            sam_output_channels,
            kernel_size=3,
            padding="same",
            use_bias=False,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
            dtype=self.dtype_policy,
            name="sam_neck_conv_2",
        )
        self.sam_neck_norm_2 = keras.layers.LayerNormalization(
            epsilon=1e-6, dtype=self.dtype_policy, name="sam_neck_norm_2"
        )
        # `padding="same"` with a stride of 2 pads only after the input on an
        # even size. The source pads one pixel on every side
        # (modeling_unlimited_ocr.py:556). `ZeroPadding2D` plus a valid
        # convolution gives the same result.
        self.sam_net_2_padding = keras.layers.ZeroPadding2D(
            1, dtype=self.dtype_policy, name="sam_net_2_padding"
        )
        self.sam_net_2 = keras.layers.Conv2D(
            self.sam_downsample_channels[0],
            kernel_size=3,
            strides=2,
            use_bias=False,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
            dtype=self.dtype_policy,
            name="sam_net_2",
        )
        self.sam_net_3_padding = keras.layers.ZeroPadding2D(
            1, dtype=self.dtype_policy, name="sam_net_3_padding"
        )
        self.sam_net_3 = keras.layers.Conv2D(
            self.sam_downsample_channels[1],
            kernel_size=3,
            strides=2,
            use_bias=False,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
            dtype=self.dtype_policy,
            name="sam_net_3",
        )

        # === CLIP encoder ===
        self.clip_embedding = UnlimitedOCRCLIPEmbedding(
            hidden_dim=clip_hidden_dim,
            grid_size=clip_image_size // clip_patch_size,
            dtype=self.dtype_policy,
            name="clip_embedding",
        )
        self.clip_pre_layer_norm = keras.layers.LayerNormalization(
            epsilon=1e-5, dtype=self.dtype_policy, name="clip_pre_layer_norm"
        )
        self.clip_layers = [
            CLIPEncoderLayer(
                hidden_dim=clip_hidden_dim,
                num_heads=clip_num_heads,
                intermediate_dim=clip_intermediate_dim,
                intermediate_activation="quick_gelu",
                use_causal_mask=False,
                dtype=self.dtype_policy,
                name=f"clip_layer_{i}",
            )
            for i in range(clip_num_layers)
        ]

    def build(self, input_shape):
        # Trace the sublayers on a symbolic input to create all weights. The
        # smallest image gives a 1x1 feature grid.
        side = self.sam_patch_size * 4
        self.call(keras.KerasTensor((1, side, side, input_shape[-1])))
        self.built = True

    def call(self, x):
        # === SAM encoder ===
        x = self.sam_patch_embedding(x)
        x = self.sam_position_embedding(x)
        for block in self.sam_blocks:
            x = block(x)
        x = self.sam_neck_norm_1(self.sam_neck_conv_1(x))
        x = self.sam_neck_norm_2(self.sam_neck_conv_2(x))
        x = self.sam_net_2(self.sam_net_2_padding(x))
        sam_features = self.sam_net_3(self.sam_net_3_padding(x))

        # === CLIP encoder ===
        x = self.clip_embedding(sam_features)
        x = self.clip_pre_layer_norm(x)
        for layer in self.clip_layers:
            x = layer(x)

        batch = ops.shape(sam_features)[0]
        height, width = sam_features.shape[1], sam_features.shape[2]
        sam_features = ops.reshape(
            sam_features,
            (batch, height * width, self.sam_downsample_channels[-1]),
        )
        # The CLIP class token does not reach the output
        # (modeling_unlimited_ocr.py:968-969).
        return ops.concatenate([x[:, 1:, :], sam_features], axis=-1)

    def compute_output_spec(self, x):
        patch = self.sam_patch_size
        height, width = x.shape[1], x.shape[2]
        num_tokens = None
        if height is not None and width is not None:
            # Both strided convolutions use a kernel of 3, a stride of 2 and
            # a padding of 1.
            grid = [height // patch, width // patch]
            for _ in range(2):
                grid = [(size + 1) // 2 for size in grid]
            num_tokens = grid[0] * grid[1]
        return keras.KerasTensor(
            (x.shape[0], num_tokens, self.output_dim), dtype=self.compute_dtype
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sam_hidden_dim": self.sam_hidden_dim,
                "sam_num_layers": self.sam_num_layers,
                "sam_num_heads": self.sam_num_heads,
                "sam_mlp_dim": self.sam_mlp_dim,
                "sam_output_channels": self.sam_output_channels,
                "sam_patch_size": self.sam_patch_size,
                "sam_image_size": self.sam_image_size,
                "sam_window_size": self.sam_window_size,
                "sam_global_attn_indexes": self.sam_global_attn_indexes,
                "sam_downsample_channels": self.sam_downsample_channels,
                "clip_hidden_dim": self.clip_hidden_dim,
                "clip_num_layers": self.clip_num_layers,
                "clip_num_heads": self.clip_num_heads,
                "clip_intermediate_dim": self.clip_intermediate_dim,
                "clip_image_size": self.clip_image_size,
                "clip_patch_size": self.clip_patch_size,
            }
        )
        return config
