import keras

from keras_hub.src.models.flux.flux_layers import DoubleStreamBlock
from keras_hub.src.models.flux.flux_layers import EmbedND
from keras_hub.src.models.flux.flux_layers import LastLayer
from keras_hub.src.models.flux.flux_layers import MLPEmbedder
from keras_hub.src.models.flux.flux_layers import SingleStreamBlock
from keras_hub.src.models.flux.flux_maths import TimestepEmbedding


class Flux(keras.Model):
    """
    Transformer model for flow matching on sequences,
    utilizing a double-stream and single-stream block structure.

    The model processes image and text data with associated positional and timestep
    embeddings, and optionally applies guidance embedding. Double-stream blocks
    handle separate image and text streams, while single-stream blocks combine
    these streams. Ported from: https://github.com/black-forest-labs/flux

    Args:
        in_channels: int. The number of input channels.
        hidden_size: int. The hidden size of the transformer, must be divisible by `num_heads`.
        mlp_ratio: float. The ratio of the MLP dimension to the hidden size.
        num_heads: int. The number of attention heads.
        depth: int. The number of double-stream blocks.
        depth_single_blocks: int. The number of single-stream blocks.
        axes_dim: list[int]. A list of dimensions for the positional embedding axes.
        theta: int. The base frequency for positional embeddings.
        qkv_bias: bool. Whether to apply bias to the query, key, and value projections.
        guidance_embed: bool. If True, applies guidance embedding in the model.

    Raises:
        ValueError: If `hidden_size` is not divisible by `num_heads`, or if `sum(axes_dim)` is not equal to the
                    positional embedding dimension.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        mlp_ratio: float,
        num_heads: int,
        depth: int,
        depth_single_blocks: int,
        axes_dim: list[int],
        theta: int,
        qkv_bias: bool,
        guidance_embed: bool,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = self.in_channels
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        pe_dim = hidden_size // num_heads
        if sum(axes_dim) != pe_dim:
            raise ValueError(
                f"Got {axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)
        self.img_in = keras.layers.Dense(self.hidden_size, use_bias=True)
        self.time_in = MLPEmbedder(hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(hidden_dim=self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(hidden_dim=self.hidden_size)
            if guidance_embed
            else keras.layers.Identity()
        )
        self.txt_in = keras.layers.Dense(self.hidden_size)

        self.double_blocks = [
            DoubleStreamBlock(
                self.hidden_size,
                self.num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
            )
            for _ in range(depth)
        ]

        self.single_blocks = [
            SingleStreamBlock(
                self.hidden_size, self.num_heads, mlp_ratio=mlp_ratio
            )
            for _ in range(depth_single_blocks)
        ]

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)
        self.timestep_embedding = TimestepEmbedding()
        self.guidance_embed = guidance_embed

    def build(self, input_shape):
        (
            img_shape,
            img_ids_shape,
            txt_shape,
            txt_ids_shape,
            timestep_shape,
            y_shape,
            guidance_shape,
        ) = input_shape

        # Build input layers
        self.img_in.build(img_shape)
        self.txt_in.build(txt_shape)

        # Build timestep embedding and vector inputs
        self.timestep_embedding.build(timestep_shape)
        self.time_in.build((None, 256))  # timestep embedding size is 256
        self.vector_in.build(y_shape)

        if self.guidance_embed:
            if guidance_shape is None:
                raise ValueError(
                    "Guidance shape must be provided for guidance-distilled model."
                )
            self.guidance_in.build(
                (None, 256)
            )  # guidance embedding size is 256

        # Build positional embedder
        ids_shape = (
            None,
            img_ids_shape[1] + txt_ids_shape[1],
            img_ids_shape[2],
        )
        self.pe_embedder.build(ids_shape)

        # Build double stream blocks
        for block in self.double_blocks:
            block.build((img_shape, txt_shape, (None, 256), ids_shape))

        # Build single stream blocks
        concat_img_shape = (
            None,
            img_shape[1] + txt_shape[1],
            self.hidden_size,
        )  # Concatenated shape
        for block in self.single_blocks:
            block.build((concat_img_shape, (None, 256), ids_shape))

        # Build final layer
        # Adjusted to match expected input shape for the final layer
        self.final_layer.build(
            (None, img_shape[1] + txt_shape[1], self.hidden_size)
        )  # Concatenated shape

        self.built = True  # Mark as built

    def call(
        self,
        img,
        img_ids,
        txt,
        txt_ids,
        timesteps,
        y,
        guidance=None,
    ):
        """
        Forward pass through the Flux model.

        Args:
            img: KerasTensor. Image input tensor of shape (N, L, D) where N is the batch size,
                 L is the sequence length, and D is the feature dimension.
            img_ids: KerasTensor. Image ID input tensor of shape (N, L, D) corresponding
                 to the image sequences.
            txt: KerasTensor. Text input tensor of shape (N, L, D).
            txt_ids: KerasTensor. Text ID input tensor of shape (N, L, D) corresponding
                to the text sequences.
            timesteps: KerasTensor. Timestep tensor used to compute positional embeddings.
            y: KerasTensor. Additional vector input, such as target values.
            guidance: KerasTensor, optional. Guidance input tensor used
                in guidance-embedded models.

        Returns:
            KerasTensor: The output tensor of the model, processed through
            double and single stream blocks and the final layer.
        """
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError(
                "Input img and txt tensors must have 3 dimensions."
            )

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(self.timestep_embedding(timesteps, dim=256))
        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.guidance_in(
                self.timestep_embedding(guidance, dim=256)
            )
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = keras.ops.concatenate((txt_ids, img_ids), axis=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = keras.ops.concatenate((txt, img), axis=1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(
            img, vec
        )  # (N, T, patch_size ** 2 * out_channels)
        return img
