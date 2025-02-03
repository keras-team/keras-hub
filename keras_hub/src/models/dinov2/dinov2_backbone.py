import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.dinov2.dinov2_layers import DinoV2Encoder
from keras_hub.src.models.dinov2.dinov2_layers import Dinov2PatchAndEmbeddings
from keras_hub.src.utils.keras_utils import standardize_data_format


@keras_hub_export("keras_hub.models.DinoV2Backbone")
class DinoV2Backbone(Backbone):
    """DinoV2 backbone.

    This backbone implements the Vision Transformer architecture as described in
    [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).
    It transforms the input image into a sequence of patches, embeds them, and
    then processes them through a series of Transformer encoder layers.

    Args:
        image_shape: A tuple or list of 3 integers representing the shape of the
            input image `(height, width, channels)`, `height` and `width` must
            be equal.
        patch_size: (int, int). The size of each image patch, the input image
            will be divided into patches of shape
            `(patch_size_h, patch_size_w)`.
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
        drop_path_rate=0.0,
        attention_dropout=0.0,
        layer_norm_epsilon=1e-6,
        layer_scale_value=1.0,
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

        if image_shape[h_axis] % patch_size[0] != 0:
            raise ValueError(
                f"Input height {image_shape[h_axis]} should be divisible by "
                f"patch size {patch_size[0]}."
            )

        if image_shape[w_axis] % patch_size[1] != 0:
            raise ValueError(
                f"Input width {image_shape[h_axis]} should be divisible by "
                f"patch size {patch_size[1]}."
            )

        num_channels = image_shape[channels_axis]

        # === Functional Model ===
        inputs = keras.layers.Input(shape=image_shape)

        x = Dinov2PatchAndEmbeddings(
            image_size=(image_shape[h_axis], image_shape[w_axis]),
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            num_channels=num_channels,
            data_format=data_format,
            dropout_rate=dropout_rate,
            dtype=dtype,
            name="dinov2_patching_and_embedding",
        )(inputs)

        output, all_hidden_states, all_attention_scores = DinoV2Encoder(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            use_mha_bias=use_mha_bias,
            use_mlp_bias=use_mlp_bias,
            dropout_rate=dropout_rate,
            drop_path_rate=drop_path_rate,
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
        self.layer_scale_value = layer_scale_value
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
                "layer_scale_value": self.layer_scale_value,
                "use_mha_bias": self.use_mha_bias,
                "use_mlp_bias": self.use_mlp_bias,
            }
        )
        return config
