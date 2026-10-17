import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.flux.flux_layers import EmbedND
from keras_hub.src.models.flux.flux_layers import StripTextTokens
from keras_hub.src.models.flux.flux_maths import TimestepEmbedding
from keras_hub.src.models.flux2.flux2_layers import Flux2DoubleStreamBlock
from keras_hub.src.models.flux2.flux2_layers import Flux2LastLayer
from keras_hub.src.models.flux2.flux2_layers import Flux2MLPEmbedder
from keras_hub.src.models.flux2.flux2_layers import Flux2Modulation
from keras_hub.src.models.flux2.flux2_layers import Flux2SingleStreamBlock


@keras_hub_export("keras_hub.models.Flux2Backbone")
class Flux2Backbone(Backbone):
    """FLUX.2 rectified flow transformer.

    The FLUX.2 MMDiT denoiser: double-stream blocks process the image and
    text token streams with separate weights and joint attention, then
    single-stream blocks process the concatenated `[text, image]` sequence.
    Compared to `keras_hub.models.FluxBackbone` (FLUX.1), FLUX.2:

    - shares one AdaLN modulation projection across all double-stream blocks
      (per stream) and one across all single-stream blocks, instead of
      giving every block its own;
    - uses SwiGLU instead of GELU feed-forward layers;
    - has no bias in any linear layer;
    - drops the pooled CLIP vector (`y`): conditioning is the timestep (and,
      for guidance-distilled variants, the guidance strength) only;
    - uses four RoPE position axes, `(t, h, w, l)`, instead of three.

    The text conditioning is the stack of several hidden layers of the text
    encoder concatenated along the feature axis. For FLUX.2 [klein] 4B this
    is the output of Qwen3-4B layers 9, 18 and 27, so `text_shape[-1]` is
    `3 * 2560 = 7680`.

    This backbone covers the transformer only; the text encoder, the VAE and
    the sampler are separate. Ported from
    https://github.com/black-forest-labs/flux2.

    Args:
        input_channels: int. The number of channels of each packed latent
            token (and of the output). `128` for FLUX.2: 32 VAE latent
            channels packed into 2x2 patches.
        hidden_size: int. The hidden size of the transformer. Must be
            divisible by `num_heads`.
        mlp_ratio: float. The ratio of the MLP hidden size to `hidden_size`.
        num_heads: int. The number of attention heads.
        depth: int. The number of double-stream blocks.
        depth_single_blocks: int. The number of single-stream blocks.
        axes_dim: list of int. The RoPE dimension of each position axis.
            `sum(axes_dim)` must equal the head dimension,
            `hidden_size // num_heads`.
        theta: int. The RoPE base frequency.
        guidance_embed: bool. Whether the model is guidance-distilled and
            takes a `guidance` input. Defaults to `False`.
        image_shape: tuple. The shape of the `image` input, excluding the
            batch axis. Defaults to `(None, 128)`.
        text_shape: tuple. The shape of the `text` input, excluding the batch
            axis. Defaults to `(None, 7680)`.
        image_ids_shape: tuple. The shape of the `image_ids` input,
            excluding the batch axis. Defaults to `(None, 4)`.
        text_ids_shape: tuple. The shape of the `text_ids` input, excluding
            the batch axis. Defaults to `(None, 4)`.
        **kwargs: other keyword arguments passed to `keras.Model`, including
            `name`, `dtype` etc.

    Call arguments:
        image: Tensor of shape `(batch_size, image_length, input_channels)`.
            The packed noisy latents.
        image_ids: Tensor of shape `(batch_size, image_length, 4)`. The
            `(t, h, w, l)` RoPE position of every latent token, i.e.
            `(0, row, column, 0)` for text-to-image.
        text: Tensor of shape `(batch_size, text_length, text_shape[-1])`.
            The text encoder hidden states.
        text_ids: Tensor of shape `(batch_size, text_length, 4)`. The RoPE
            position of every text token, i.e. `(0, 0, 0, index)`.
        timesteps: Tensor of shape `(batch_size,)`. The flow-matching time
            in `[0, 1]` (it is scaled by 1000 internally).
        guidance: Tensor of shape `(batch_size,)`. The guidance strength.
            Only present when `guidance_embed=True`.

    Example:
    ```python
    import numpy as np
    import keras_hub

    # Load a pretrained FLUX.2 [klein] 4B backbone.
    backbone = keras_hub.models.Flux2Backbone.from_preset("flux2_klein_4b")

    # A randomly initialized, tiny FLUX.2 backbone.
    backbone = keras_hub.models.Flux2Backbone(
        input_channels=16,
        hidden_size=64,
        mlp_ratio=3.0,
        num_heads=2,
        depth=1,
        depth_single_blocks=2,
        axes_dim=[8, 8, 8, 8],
        theta=2000,
        image_shape=(None, 16),
        text_shape=(None, 24),
    )
    outputs = backbone(
        {
            "image": np.ones((1, 12, 16)),
            "image_ids": np.zeros((1, 12, 4)),
            "text": np.ones((1, 5, 24)),
            "text_ids": np.zeros((1, 5, 4)),
            "timesteps": np.ones((1,)),
        }
    )
    ```
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
        guidance_embed=False,
        image_shape=(None, 128),
        text_shape=(None, 7680),
        image_ids_shape=(None, 4),
        text_ids_shape=(None, 4),
        **kwargs,
    ):
        if hidden_size % num_heads != 0:
            raise ValueError(
                "`hidden_size` must be divisible by `num_heads`. Received: "
                f"hidden_size={hidden_size}, num_heads={num_heads}"
            )
        head_dim = hidden_size // num_heads
        if sum(axes_dim) != head_dim:
            raise ValueError(
                "`sum(axes_dim)` must equal the head dimension "
                f"(`hidden_size // num_heads` = {head_dim}). Received: "
                f"axes_dim={axes_dim} (sum {sum(axes_dim)})"
            )
        for name, shape in (
            ("image_ids_shape", image_ids_shape),
            ("text_ids_shape", text_ids_shape),
        ):
            if shape[-1] != len(axes_dim):
                raise ValueError(
                    f"The last axis of `{name}` must equal "
                    f"`len(axes_dim)` = {len(axes_dim)}, one position per "
                    f"RoPE axis. Received: {name}={shape}"
                )

        # === Layers ===
        self.positional_embedder = EmbedND(
            theta=theta, axes_dim=axes_dim, name="positional_embedder"
        )
        self.image_input_embedder = keras.layers.Dense(
            hidden_size, use_bias=False, name="image_input_embedder"
        )
        self.text_input_embedder = keras.layers.Dense(
            hidden_size, use_bias=False, name="text_input_embedder"
        )
        self.timestep_embedding = TimestepEmbedding(
            dim=256, name="timestep_embedding"
        )
        self.time_input_embedder = Flux2MLPEmbedder(
            hidden_size, name="time_input_embedder"
        )
        self.guidance_input_embedder = (
            Flux2MLPEmbedder(hidden_size, name="guidance_input_embedder")
            if guidance_embed
            else None
        )
        self.double_stream_modulation_image = Flux2Modulation(
            hidden_size, double=True, name="double_stream_modulation_image"
        )
        self.double_stream_modulation_text = Flux2Modulation(
            hidden_size, double=True, name="double_stream_modulation_text"
        )
        self.single_stream_modulation = Flux2Modulation(
            hidden_size, double=False, name="single_stream_modulation"
        )
        self.double_blocks = [
            Flux2DoubleStreamBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                name=f"double_block_{i}",
            )
            for i in range(depth)
        ]
        self.single_blocks = [
            Flux2SingleStreamBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                name=f"single_block_{i}",
            )
            for i in range(depth_single_blocks)
        ]
        self.strip_text_tokens = StripTextTokens(name="strip_text_tokens")
        self.final_layer = Flux2LastLayer(
            hidden_size, input_channels, name="final_layer"
        )

        # === Functional Model ===
        image_input = keras.Input(shape=image_shape, name="image")
        image_ids = keras.Input(shape=image_ids_shape, name="image_ids")
        text_input = keras.Input(shape=text_shape, name="text")
        text_ids = keras.Input(shape=text_ids_shape, name="text_ids")
        timesteps_input = keras.Input(shape=(), name="timesteps")
        guidance_input = (
            keras.Input(shape=(), name="guidance") if guidance_embed else None
        )

        image = self.image_input_embedder(image_input)
        text = self.text_input_embedder(text_input)

        modulation_encoding = self.time_input_embedder(
            self.timestep_embedding(timesteps_input)
        )
        if guidance_embed:
            modulation_encoding = modulation_encoding + (
                self.guidance_input_embedder(
                    self.timestep_embedding(guidance_input)
                )
            )

        # Text first: every block assembles its attention sequence as
        # `[text, image]`, and the RoPE positions must follow the same order.
        ids = ops.concatenate((text_ids, image_ids), axis=1)
        positional_encoding = self.positional_embedder(ids)

        # Computed once and shared by every block of the same kind.
        image_modulation = self.double_stream_modulation_image(
            modulation_encoding
        )
        text_modulation = self.double_stream_modulation_text(
            modulation_encoding
        )
        single_modulation = self.single_stream_modulation(modulation_encoding)

        for block in self.double_blocks:
            image, text = block(
                image=image,
                text=text,
                image_modulation=image_modulation,
                text_modulation=text_modulation,
                positional_encoding=positional_encoding,
            )

        sequence = ops.concatenate((text, image), axis=1)
        for block in self.single_blocks:
            sequence = block(
                sequence,
                modulation=single_modulation,
                positional_encoding=positional_encoding,
            )
        image = self.strip_text_tokens(sequence, text)
        outputs = self.final_layer(image, modulation_encoding)

        inputs = {
            "image": image_input,
            "image_ids": image_ids,
            "text": text_input,
            "text_ids": text_ids,
            "timesteps": timesteps_input,
        }
        if guidance_embed:
            inputs["guidance"] = guidance_input

        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        # === Config ===
        self.input_channels = input_channels
        self.output_channels = input_channels
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.depth = depth
        self.depth_single_blocks = depth_single_blocks
        self.axes_dim = axes_dim
        self.theta = theta
        self.guidance_embed = guidance_embed
        self.image_shape = image_shape
        self.text_shape = text_shape
        self.image_ids_shape = image_ids_shape
        self.text_ids_shape = text_ids_shape

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
                "guidance_embed": self.guidance_embed,
                "image_shape": self.image_shape,
                "text_shape": self.text_shape,
                "image_ids_shape": self.image_ids_shape,
                "text_ids_shape": self.text_ids_shape,
            }
        )
        return config
