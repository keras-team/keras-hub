import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.vit.vit_layers import ViTEncoder
from keras_hub.src.models.vit.vit_layers import ViTPatchingAndEmbedding
from keras_hub.src.utils.keras_utils import standardize_data_format


@keras_hub_export("keras_hub.models.ViTBackbone")
class ViTBackbone(Backbone):
    """Vision Transformer (ViT) backbone.

    This backbone implements the Vision Transformer architecture as described in
    [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).
    It transforms the input image into a sequence of patches, embeds them, and
    then processes them through a series of Transformer encoder layers.

    Args:
        image_shape: A tuple or list of 3 integers representing the shape of the
            input image `(height, width, channels)`, `height` and `width` must
            be equal.
        patch_size: int. The size of each image patch, the input image will be
            divided into patches of shape `(patch_size, patch_size)`.
        num_layers: int. The number of transformer encoder layers.
        num_heads: int. specifying the number of attention heads in each
            Transformer encoder layer.
        hidden_dim: int. The dimensionality of the hidden representations.
        mlp_dim: int. The dimensionality of the intermediate MLP layer in
            each Transformer encoder layer.
        dropout_rate: float. The dropout rate for the Transformer encoder
            layers.
        attention_dropout: float. The dropout rate for the attention mechanism
            in each Transformer encoder layer.
        layer_norm_epsilon: float. Value used for numerical stability in
            layer normalization.
        use_mha_bias: bool. Whether to use bias in the multi-head
            attention layers.
        use_mlp_bias: bool. Whether to use bias in the MLP layers.
        data_format: str.  `"channels_last"` or `"channels_first"`, specifying
            the data format for the input image. If `None`, defaults to
            `"channels_last"`.
        dtype: The dtype of the layer weights. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the parent
            `Backbone` class.
    """

    def __init__(
        self,
        image_shape,
        patch_size,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout_rate=0.0,
        attention_dropout=0.0,
        layer_norm_epsilon=1e-6,
        use_mha_bias=True,
        use_mlp_bias=True,
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        # === Laters ===
        data_format = standardize_data_format(data_format)
        h_axis, w_axis, channels_axis = (
            (-3, -2, -1) if data_format == "channels_last" else (-2, -1, -3)
        )
        # Check that the input image is well specified.
        if image_shape[h_axis] is None or image_shape[w_axis] is None:
            raise ValueError(
                f"Image shape must have defined height and width. Found `None` "
                f"at index {h_axis} (height) or {w_axis} (width). "
                f"Image shape: {image_shape}"
            )
        if image_shape[h_axis] != image_shape[w_axis]:
            raise ValueError(
                f"Image height and width must be equal. Found height: "
                f"{image_shape[h_axis]}, width: {image_shape[w_axis]} at "
                f"indices {h_axis} and {w_axis} respectively. Image shape: "
                f"{image_shape}"
            )

        num_channels = image_shape[channels_axis]

        # === Functional Model ===
        inputs = keras.layers.Input(shape=image_shape)

        x = ViTPatchingAndEmbedding(
            image_size=image_shape[h_axis],
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            num_channels=num_channels,
            data_format=data_format,
            dtype=dtype,
            name="vit_patching_and_embedding",
        )(inputs)

        output = ViTEncoder(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            layer_norm_epsilon=layer_norm_epsilon,
            use_mha_bias=use_mha_bias,
            use_mlp_bias=use_mlp_bias,
            dtype=dtype,
            name="vit_encoder",
        )(x)

        super().__init__(
            inputs=inputs,
            outputs=output,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.image_shape = image_shape
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_mha_bias = use_mha_bias
        self.use_mlp_bias = use_mlp_bias
        self.data_format = data_format

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_shape": self.image_shape,
                "patch_size": self.patch_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "mlp_dim": self.mlp_dim,
                "dropout_rate": self.dropout_rate,
                "attention_dropout": self.attention_dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "use_mha_bias": self.use_mha_bias,
                "use_mlp_bias": self.use_mlp_bias,
            }
        )
        return config
