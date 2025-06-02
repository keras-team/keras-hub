import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.deit.deit_layers import DeiTEmbeddings
from keras_hub.src.models.deit.deit_layers import DeiTEncoder
from keras_hub.src.utils.keras_utils import standardize_data_format


@keras_hub_export("keras_hub.models.DeiTBackbone")
class DeiTBackbone(Backbone):
    """DeiT backbone.

    This backbone implements the Data-efficient Image Transformer (DeiT)
    architecture as described in [Training data-efficient image
    transformers & distillation through attention]
    (https://arxiv.org/abs/2012.12877).

    Args:
        image_shape: A tuple or list of 3 integers representing the shape of the
            input image `(height, width, channels)`.
        patch_size: tuple or int. The size of each image patch. If an int is
            provided, it will be used for both height and width. The input image
            will be split into patches of shape `(patch_size_h, patch_size_w)`.
        num_layers: int. The number of transformer encoder layers.
        num_heads: int. The number of attention heads in each Transformer
            encoder layer.
        hidden_dim: int. The dimensionality of the hidden representations.
        intermediate_dim: int. The dimensionality of the intermediate MLP layer
            in each Transformer encoder layer.
        dropout_rate: float. The dropout rate for the Transformer encoder
            layers.
        attention_dropout: float. The dropout rate for the attention mechanism
            in each Transformer encoder layer.
        layer_norm_epsilon: float. Value used for numerical stability in layer
            normalization.
        use_mha_bias: bool. Whether to use bias in the multi-head attention
            layers.
        data_format: str. `"channels_last"` or `"channels_first"`, specifying
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
        intermediate_dim,
        dropout_rate=0.0,
        attention_dropout=0.0,
        layer_norm_epsilon=1e-6,
        use_mha_bias=True,
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        # === Laters ===
        data_format = standardize_data_format(data_format)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
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
        # Check that image dimensions be divisible by patch size
        if image_shape[h_axis] % patch_size[0] != 0:
            raise ValueError(
                f"Input height {image_shape[h_axis]} should be divisible by "
                f"patch size {patch_size}."
            )
        if image_shape[w_axis] % patch_size[1] != 0:
            raise ValueError(
                f"Input height {image_shape[w_axis]} should be divisible by "
                f"patch size {patch_size}."
            )

        num_channels = image_shape[channels_axis]

        # === Functional Model ===
        inputs = keras.layers.Input(shape=image_shape)

        x = DeiTEmbeddings(
            image_size=(image_shape[h_axis], image_shape[w_axis]),
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            num_channels=num_channels,
            data_format=data_format,
            dropout_rate=dropout_rate,
            dtype=dtype,
            name="deit_patching_and_embedding",
        )(inputs)

        output, _, _ = DeiTEncoder(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            use_mha_bias=use_mha_bias,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            layer_norm_epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="deit_encoder",
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
        self.intermediate_dim = intermediate_dim
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_mha_bias = use_mha_bias
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
                "intermediate_dim": self.intermediate_dim,
                "dropout_rate": self.dropout_rate,
                "attention_dropout": self.attention_dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "use_mha_bias": self.use_mha_bias,
            }
        )
        return config
