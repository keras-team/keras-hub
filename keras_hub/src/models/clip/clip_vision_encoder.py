from keras import layers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.clip.clip_encoder_block import CLIPEncoderBlock
from keras_hub.src.models.clip.clip_vision_embedding import CLIPVisionEmbedding
from keras_hub.src.utils.keras_utils import standardize_data_format


@keras_hub_export("keras_hub.models.CLIPVisionEncoder")
class CLIPVisionEncoder(Backbone):
    """CLIP vision core network with hyperparameters.

    Args:
        patch_size: int. The size of each square patch in the input image.
        hidden_dim: int. The size of the transformer hidden state at the end
            of each transformer layer.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads for each transformer.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        intermediate_activation: activation function. The activation that
            is used for the first Dense layer in a two-layer feedforward network
            for each transformer.
        intermediate_output_index: optional int. The index of the intermediate
            output. If specified, the output will become a dictionary with two
            keys `"sequence_output"` and `"intermediate_output"`.
        image_shape: tuple. The input shape without the batch size. Defaults to
            `(224, 224, 3)`.
        data_format: `None` or str. If specified, either `"channels_last"` or
            `"channels_first"`. The ordering of the dimensions in the
            inputs. `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the models computations and weights. Note that some
            computations, such as softmax and layer normalization will always
            be done a float32 precision regardless of dtype.
    """

    def __init__(
        self,
        patch_size,
        hidden_dim,
        num_layers,
        num_heads,
        intermediate_dim,
        intermediate_activation="quick_gelu",
        intermediate_output_index=None,
        image_shape=(224, 224, 3),
        data_format=None,
        dtype=None,
        name=None,
        **kwargs,
    ):
        data_format = standardize_data_format(data_format)
        if data_format == "channels_last":
            height, width = image_shape[0], image_shape[1]
        else:
            height, width = image_shape[1], image_shape[2]
        if height != width:
            raise ValueError(
                "`CLIPVisionEncoder` expects the height and width to be the "
                f"same in `image_shape`. Received: image_shape={image_shape}"
            )

        if (
            intermediate_output_index is not None
            and intermediate_output_index < 0
        ):
            intermediate_output_index += num_layers

        # `prefix` is used to prevent duplicate name when utilizing multiple
        # CLIP models within a single model, such as in StableDiffusion3.
        prefix = str(name) + "_" if name is not None else ""

        # === Layers ===
        self.embedding = CLIPVisionEmbedding(
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            image_size=height,
            data_format=data_format,
            dtype=dtype,
            name=f"{prefix}embedding",
        )
        self.pre_layer_norm = layers.LayerNormalization(
            epsilon=1e-5, dtype=dtype, name=f"{prefix}pre_layer_norm"
        )
        self.encoder_layers = [
            CLIPEncoderBlock(
                hidden_dim,
                num_heads,
                intermediate_dim,
                intermediate_activation,
                use_causal_mask=False,  # `False` in the vision encoder.
                dtype=dtype,
                name=f"{prefix}encoder_block_{i}",
            )
            for i in range(num_layers)
        ]
        self.layer_norm = layers.LayerNormalization(
            epsilon=1e-5, dtype=dtype, name=f"{prefix}layer_norm"
        )

        # === Functional Model ===
        image_input = layers.Input(shape=image_shape, name="images")
        x = self.embedding(image_input)
        x = self.pre_layer_norm(x)
        intermediate_output = None
        for i, block in enumerate(self.encoder_layers):
            x = block(x)
            if i == intermediate_output_index:
                intermediate_output = x
        sequence_output = self.layer_norm(x)

        if intermediate_output_index is not None:
            outputs = {
                "sequence_output": sequence_output,
                "intermediate_output": intermediate_output,
            }
        else:
            outputs = sequence_output
        super().__init__(
            inputs={"images": image_input},
            outputs=outputs,
            dtype=dtype,
            name=name,
            **kwargs,
        )

        # === Config ===
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.intermediate_activation = intermediate_activation
        self.intermediate_output_index = intermediate_output_index
        self.image_shape = image_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "intermediate_activation": self.intermediate_activation,
                "intermediate_output_index": self.intermediate_output_index,
                "image_shape": self.image_shape,
            }
        )
        return config
