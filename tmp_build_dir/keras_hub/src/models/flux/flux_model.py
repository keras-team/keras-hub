import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.flux.flux_layers import DoubleStreamBlock
from keras_hub.src.models.flux.flux_layers import EmbedND
from keras_hub.src.models.flux.flux_layers import LastLayer
from keras_hub.src.models.flux.flux_layers import MLPEmbedder
from keras_hub.src.models.flux.flux_layers import SingleStreamBlock
from keras_hub.src.models.flux.flux_maths import TimestepEmbedding


@keras_hub_export("keras_hub.models.FluxBackbone")
class FluxBackbone(Backbone):
    """Transformer model for flow matching on sequences.

    The model processes image and text data with associated positional and
    timestep embeddings, and optionally applies guidance embedding.
    Double-stream blocks handle separate image and text streams, while
    single-stream blocks combine these streams. Ported from:
    https://github.com/black-forest-labs/flux


    Args:
        input_channels: int. The number of input channels.
        hidden_size: int. The hidden size of the transformer, must be divisible
            by `num_heads`.
        mlp_ratio: float. The ratio of the MLP dimension to the hidden size.
        num_heads: int. The number of attention heads.
        depth: int. The number of double-stream blocks.
        depth_single_blocks: int. The number of single-stream blocks.
        axes_dim: list[int]. A list of dimensions for the positional embedding
            axes.
        theta: int. The base frequency for positional embeddings.
        use_bias: bool. Whether to apply bias to the query, key, and value
            projections.
        guidance_embed: bool. If True, applies guidance embedding in the model.

    Call arguments:
        image: KerasTensor. Image input tensor of shape (N, L, D) where N is the
            batch size, L is the sequence length, and D is the feature
            dimension.
        image_ids: KerasTensor. Image ID input tensor of shape (N, L, D)
            corresponding to the image sequences.
        text: KerasTensor. Text input tensor of shape (N, L, D).
        text_ids: KerasTensor. Text ID input tensor of shape (N, L, D)
            corresponding to the text sequences.
        timesteps: KerasTensor. Timestep tensor used to compute positional
            embeddings.
        y: KerasTensor. Additional vector input, such as target values.
        guidance: KerasTensor, optional. Guidance input tensor used
            in guidance-embedded models.
    Raises:
        ValueError: If `hidden_size` is not divisible by `num_heads`, or if
            `sum(axes_dim)` is not equal to the positional embedding dimension.
    """

    def __init__(
        self,
        input_channels,
        hidden_size,
        mlp_ratio,
        num_heads,
        depth,
        depth_single_blocks,
        axes_dim,
        theta,
        use_bias,
        guidance_embed=False,
        # These will be inferred from the CLIP/T5 encoders later
        image_shape=(None, 768, 3072),
        text_shape=(None, 768, 3072),
        image_ids_shape=(None, 768, 3072),
        text_ids_shape=(None, 768, 3072),
        y_shape=(None, 128),
        **kwargs,
    ):
        # === Layers ===
        self.positional_embedder = EmbedND(theta=theta, axes_dim=axes_dim)
        self.image_input_embedder = keras.layers.Dense(
            hidden_size, use_bias=True
        )
        self.time_input_embedder = MLPEmbedder(hidden_dim=hidden_size)
        self.vector_embedder = MLPEmbedder(hidden_dim=hidden_size)
        self.guidance_input_embedder = (
            MLPEmbedder(hidden_dim=hidden_size)
            if guidance_embed
            else keras.layers.Identity()
        )
        self.text_input_embedder = keras.layers.Dense(hidden_size)

        self.double_blocks = [
            DoubleStreamBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                use_bias=use_bias,
            )
            for _ in range(depth)
        ]

        self.single_blocks = [
            SingleStreamBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth_single_blocks)
        ]

        self.final_layer = LastLayer(hidden_size, 1, input_channels)
        self.timestep_embedding = TimestepEmbedding()
        self.guidance_embed = guidance_embed

        # === Functional Model ===
        image_input = keras.Input(shape=image_shape, name="image")
        image_ids = keras.Input(shape=image_ids_shape, name="image_ids")
        text_input = keras.Input(shape=text_shape, name="text")
        text_ids = keras.Input(shape=text_ids_shape, name="text_ids")
        y = keras.Input(shape=y_shape, name="y")
        timesteps_input = keras.Input(shape=(), name="timesteps")
        guidance_input = keras.Input(shape=(), name="guidance")

        # running on sequences image
        image = self.image_input_embedder(image_input)
        modulation_encoding = self.time_input_embedder(
            self.timestep_embedding(timesteps_input, dim=256)
        )
        if self.guidance_embed:
            if guidance_input is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            modulation_encoding = (
                modulation_encoding
                + self.guidance_input_embedder(
                    self.timestep_embedding(guidance_input, dim=256)
                )
            )

        modulation_encoding = modulation_encoding + self.vector_embedder(y)
        text = self.text_input_embedder(text_input)

        ids = keras.ops.concatenate((text_ids, image_ids), axis=1)
        positional_encoding = self.positional_embedder(ids)

        for block in self.double_blocks:
            image, text = block(
                image=image,
                text=text,
                modulation_encoding=modulation_encoding,
                positional_encoding=positional_encoding,
            )

        image = keras.ops.concatenate((text, image), axis=1)
        for block in self.single_blocks:
            image = block(
                image,
                modulation_encoding=modulation_encoding,
                positional_encoding=positional_encoding,
            )
        image = image[:, text.shape[1] :, ...]

        image = self.final_layer(
            image, modulation_encoding
        )  # (N, T, patch_size ** 2 * output_channels)

        super().__init__(
            inputs={
                "image": image_input,
                "image_ids": image_ids,
                "text": text_input,
                "text_ids": text_ids,
                "y": y,
                "timesteps": timesteps_input,
                "guidance": guidance_input,
            },
            outputs=image,
            **kwargs,
        )

        # === Config ===
        self.input_channels = input_channels
        self.output_channels = self.input_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.image_shape = image_shape
        self.text_shape = text_shape
        self.image_ids_shape = image_ids_shape
        self.text_ids_shape = text_ids_shape
        self.y_shape = y_shape
        self.mlp_ratio = mlp_ratio
        self.depth = depth
        self.depth_single_blocks = depth_single_blocks
        self.axes_dim = axes_dim
        self.theta = theta
        self.use_bias = use_bias

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_channels": self.input_channels,
                "hidden_size": self.hidden_size,
                "mlp_ratio": self.mlp_ratio,
                "num_heads": self.num_heads,
                "depth": self.depth,
                "depth_single_blocks": self.depth_single_blocks,
                "axes_dim": self.axes_dim,
                "theta": self.theta,
                "use_bias": self.use_bias,
                "guidance_embed": self.guidance_embed,
                "image_shape": self.image_shape,
                "text_shape": self.text_shape,
                "image_ids_shape": self.image_ids_shape,
                "text_ids_shape": self.text_ids_shape,
                "y_shape": self.y_shape,
            }
        )
        return config

    def encode_text_step(self, token_ids, negative_token_ids):
        raise NotImplementedError("Not implemented yet")

        def encode(token_ids):
            raise NotImplementedError("Not implemented yet")

    def encode_image_step(self, images):
        raise NotImplementedError("Not implemented yet")

    def add_noise_step(self, latents, noises, step, num_steps):
        raise NotImplementedError("Not implemented yet")

    def denoise_step(
        self,
    ):
        raise NotImplementedError("Not implemented yet")

    def decode_step(self, latents):
        raise NotImplementedError("Not implemented yet")
