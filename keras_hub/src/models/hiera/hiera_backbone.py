"""`HieraBackbone` — hierarchical ViT image encoder used by SAM2."""

import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.hiera.hiera_layers import (
    HieraAbsolutePositionEmbedding,
)
from keras_hub.src.models.hiera.hiera_layers import HieraBlock
from keras_hub.src.models.hiera.hiera_layers import HieraPatchEmbedding


@keras_hub_export("keras_hub.models.HieraBackbone")
class HieraBackbone(Backbone):
    """Hiera hierarchical image encoder, as shipped in SAM2.

    This implementation matches the trunk of SAM2's image encoder
    (`image_encoder.trunk.*` in the HuggingFace `facebook/sam2-hiera-*-hf`
    safetensors). It is structured as four stages of transformer blocks with
    2x2 strided pooling and a channel-dimension doubling at every stage
    boundary, using mask-unit (windowed) attention inside each stage and
    global attention at a configurable set of block indices.

    The forward pass returns the feature map produced by the final stage as
    the default output, and exposes the multi-scale feature pyramid through
    `HieraBackbone.pyramid_outputs`. Pyramid keys follow the convention
    used elsewhere in the repo (`P2` through `P5` for strides 4, 8, 16, 32
    with a `patch_stride=4` and three pooling stages).

    Args:
        embed_dim: int. Channel dimension at stage 0. Doubles at each
            subsequent pooling stage.
        num_heads: int. Number of attention heads at stage 0. Doubles at
            each subsequent pooling stage.
        stages: tuple of four ints. Number of transformer blocks in each
            stage. Defaults to `(1, 2, 7, 2)` (Hiera tiny).
        global_attention_blocks: tuple of ints. Block indices (into the
            flattened list across all stages) that use full global
            attention. All other blocks use windowed (mask-unit) attention
            with the per-stage window size from `window_spec`.
        window_spec: tuple of four ints. Window size used by the windowed
            attention blocks in each stage.
        window_pos_embed_bkg_spatial_size: tuple `(h, w)`. Size of the
            learned background position-embedding grid; it is bicubically
            interpolated to the stage-0 feature map size.
        patch_kernel_size: tuple `(kh, kw)`. Kernel size of the patch
            embedding convolution.
        patch_stride: tuple `(sh, sw)`. Stride of the patch embedding
            convolution. Defines the coarsest-scale pyramid stride.
        q_stride: tuple `(sh, sw)`. Stride applied to the query when
            pooling between stages.
        mlp_ratio: float. Hidden-to-output ratio of the MLP in each block.
        layer_norm_epsilon: float. Epsilon for the LayerNorm layers inside
            each block.
        image_shape: tuple `(H, W, C)`. Input image shape. Defaults to
            `(1024, 1024, 3)` to match SAM2.
        dtype: Optional Keras dtype / DTypePolicy.

    Example:
    ```python
    backbone = keras_hub.models.HieraBackbone(
        embed_dim=96,
        num_heads=1,
        stages=(1, 2, 7, 2),
        global_attention_blocks=(5, 7, 9),
        window_spec=(8, 4, 14, 7),
        window_pos_embed_bkg_spatial_size=(7, 7),
    )
    pyramid = keras.Model(
        inputs=backbone.inputs, outputs=backbone.pyramid_outputs
    )
    feats = pyramid(keras.random.normal((1, 1024, 1024, 3)))
    # feats["P2"]: (1, 256, 256, 96)
    # feats["P3"]: (1, 128, 128, 192)
    # feats["P4"]: (1, 64, 64, 384)
    # feats["P5"]: (1, 32, 32, 768)
    ```
    """

    def __init__(
        self,
        embed_dim=96,
        num_heads=1,
        stages=(1, 2, 7, 2),
        global_attention_blocks=(5, 7, 9),
        window_spec=(8, 4, 14, 7),
        window_pos_embed_bkg_spatial_size=(7, 7),
        patch_kernel_size=(7, 7),
        patch_stride=(4, 4),
        q_stride=(2, 2),
        mlp_ratio=4.0,
        layer_norm_epsilon=1e-6,
        image_shape=(1024, 1024, 3),
        dtype=None,
        **kwargs,
    ):
        if len(stages) != 4:
            raise ValueError(
                "Hiera expects exactly four stages; got "
                f"len(stages)={len(stages)}."
            )
        if len(window_spec) != 4:
            raise ValueError(
                "`window_spec` must have one entry per stage; got "
                f"len(window_spec)={len(window_spec)}."
            )

        # === Layers ===
        self.patch_embed = HieraPatchEmbedding(
            embed_dim=embed_dim,
            kernel_size=patch_kernel_size,
            stride=patch_stride,
            dtype=dtype,
            name="patch_embed",
        )
        total_blocks = sum(stages)
        stage_starts = []
        cumulative = 0
        for count in stages:
            stage_starts.append(cumulative)
            cumulative += count
        global_attention_blocks = set(global_attention_blocks)

        self.blocks = []
        current_dim = embed_dim
        current_heads = num_heads
        for block_index in range(total_blocks):
            stage_index = next(
                (s for s in range(3, -1, -1) if block_index >= stage_starts[s]),
                0,
            )
            is_pool_block = (
                stage_index > 0 and block_index == stage_starts[stage_index]
            )
            if is_pool_block:
                block_q_stride = q_stride
                dim_out = current_dim * 2
                heads_out = current_heads * 2
            else:
                block_q_stride = None
                dim_out = current_dim
                heads_out = current_heads

            window_size = (
                0
                if block_index in global_attention_blocks
                else window_spec[stage_index]
            )

            block = HieraBlock(
                dim_in=current_dim,
                dim_out=dim_out,
                num_heads=heads_out if is_pool_block else current_heads,
                mlp_ratio=mlp_ratio,
                q_stride=block_q_stride,
                window_size=window_size,
                layer_norm_epsilon=layer_norm_epsilon,
                dtype=dtype,
                name=f"blocks_{block_index}",
            )
            self.blocks.append(block)
            current_dim = dim_out
            current_heads = heads_out

        stage0_height = image_shape[0] // patch_stride[0]
        stage0_width = image_shape[1] // patch_stride[1]
        self.pos_embed_layer = HieraAbsolutePositionEmbedding(
            embed_dim=embed_dim,
            background_size=window_pos_embed_bkg_spatial_size,
            window_size=window_spec[0],
            feature_map_size=(stage0_height, stage0_width),
            dtype=dtype,
            name="pos_embed",
        )

        # === Functional Model ===
        image_input = keras.Input(shape=image_shape, name="images")
        x = self.patch_embed(image_input)
        x = self.pos_embed_layer(x)

        pyramid_outputs = {}
        stage_cursor = 0
        stage_counter = 0
        for block_index, block in enumerate(self.blocks):
            x = block(x)
            stage_cursor += 1
            if stage_cursor == stages[stage_counter]:
                pyramid_outputs[f"P{stage_counter + 2}"] = x
                stage_counter += 1
                stage_cursor = 0

        super().__init__(
            inputs=image_input,
            outputs=x,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.stages = tuple(stages)
        self.global_attention_blocks = tuple(sorted(global_attention_blocks))
        self.window_spec = tuple(window_spec)
        self.window_pos_embed_bkg_spatial_size = tuple(
            window_pos_embed_bkg_spatial_size
        )
        self.patch_kernel_size = tuple(patch_kernel_size)
        self.patch_stride = tuple(patch_stride)
        self.q_stride = tuple(q_stride)
        self.mlp_ratio = mlp_ratio
        self.layer_norm_epsilon = layer_norm_epsilon
        self.image_shape = tuple(image_shape)
        self.pyramid_outputs = pyramid_outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "stages": self.stages,
                "global_attention_blocks": self.global_attention_blocks,
                "window_spec": self.window_spec,
                "window_pos_embed_bkg_spatial_size": (
                    self.window_pos_embed_bkg_spatial_size
                ),
                "patch_kernel_size": self.patch_kernel_size,
                "patch_stride": self.patch_stride,
                "q_stride": self.q_stride,
                "mlp_ratio": self.mlp_ratio,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "image_shape": self.image_shape,
            }
        )
        return config
