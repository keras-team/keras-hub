from keras import layers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.siglip.siglip_layers import SigLIPEncoderLayer
from keras_hub.src.models.siglip.siglip_layers import (
    SigLIPMultiHeadAttentionPooling,
)
from keras_hub.src.models.siglip.siglip_layers import SigLIPVisionEmbedding
from keras_hub.src.utils.keras_utils import standardize_data_format


@keras_hub_export("keras_hub.models.SigLIPVisionEncoder")
class SigLIPVisionEncoder(Backbone):
    """SigLIP vision core network with hyperparameters.

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
            for each transformer. Defaults to `"gelu_approximate"`.
        layer_norm_epsilon: float. The epsilon for the layer normalization.
            Defaults to `1e-6`.
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
        intermediate_activation="gelu_approximate",
        layer_norm_epsilon=1e-6,
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
                "`SigLIPVisionEncoder` expects the height and width to be the "
                f"same in `image_shape`. Received: image_shape={image_shape}"
            )

        # `prefix` is used to prevent duplicate name when utilizing multiple
        # SigLIP encoders within a single model.
        prefix = str(name) + "_" if name is not None else ""

        # === Layers ===
        self.embedding = SigLIPVisionEmbedding(
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            image_size=height,
            data_format=data_format,
            dtype=dtype,
            name=f"{prefix}embedding",
        )
        self.encoder_layers = [
            SigLIPEncoderLayer(
                hidden_dim,
                num_heads,
                intermediate_dim,
                intermediate_activation,
                layer_norm_epsilon=layer_norm_epsilon,
                dtype=dtype,
                name=f"{prefix}encoder_block_{i}",
            )
            for i in range(num_layers)
        ]
        self.post_layer_norm = layers.LayerNormalization(
            epsilon=1e-6, dtype=dtype, name=f"{prefix}post_layer_norm"
        )
        self.head = SigLIPMultiHeadAttentionPooling(
            hidden_dim,
            intermediate_dim,
            num_heads,
            intermediate_activation,
            layer_norm_epsilon,
            dtype=dtype,
            name=f"{prefix}head",
        )

        # === Functional Model ===
        image_input = layers.Input(shape=image_shape, name="images")
        x = self.embedding(image_input)
        for _, block in enumerate(self.encoder_layers):
            x = block(x)
        x = self.post_layer_norm(x)
        x = self.head(x)
        outputs = x
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
        self.layer_norm_epsilon = layer_norm_epsilon
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
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "image_shape": self.image_shape,
            }
        )
        return config
