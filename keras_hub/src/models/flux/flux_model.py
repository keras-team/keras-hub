import keras

from keras_hub.src.models.flux.flux_layers import DoubleStreamBlock
from keras_hub.src.models.flux.flux_layers import EmbedND
from keras_hub.src.models.flux.flux_layers import LastLayer
from keras_hub.src.models.flux.flux_layers import MLPEmbedder
from keras_hub.src.models.flux.flux_layers import SingleStreamBlock
from keras_hub.src.models.flux.flux_maths import timestep_embedding


class Flux(keras.Model):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(
        self,
        in_channels: int,
        vec_in_dim: int,
        context_in_dim: int,
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
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
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
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError(
                "Input img and txt tensors must have 3 dimensions."
            )

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
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
