import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.vit.vit_layers import ViTEncoder
from keras_hub.src.models.vit.vit_layers import ViTPatchingAndEmbedding


@keras_hub_export("keras_hub.models.Blip2VisionEncoder")
class Blip2VisionEncoder(keras.Model):
    """EVA-CLIP vision encoder for BLIP-2.

    A Vision Transformer (ViT) that encodes images into patch-level feature
    sequences consumed by the Q-Former.

    Args:
        image_size: int. Height and width of the input image (must be square).
        patch_size: int. Side length of each square patch. Must evenly divide
            `image_size`.
        num_layers: int. Number of transformer encoder layers.
        num_heads: int. Number of self-attention heads per layer.
        hidden_dim: int. Dimensionality of patch embeddings and transformer
            hidden states.
        intermediate_dim: int. Inner dimensionality of the MLP block inside
            each transformer layer.
        use_patch_bias: bool. Whether the patch-embedding Conv2D has a bias.
        use_class_token: bool. Whether to prepend a learnable [CLS] token.
        use_mha_bias: bool. Whether multi-head attention projections have
            biases.
        use_mlp_bias: bool. Whether MLP dense layers have biases.
        dropout_rate: float. Dropout probability applied inside each
            transformer layer.
        layer_norm_epsilon: float. Epsilon for layer normalisation.
        initializer_range: float. Std-dev of the truncated-normal initialiser
            used for all kernel weights. Defaults to `0.02`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. Compute and
            weight dtype for the model.

    Example:
    ```python
    encoder = Blip2VisionEncoder(
        image_size=224, patch_size=14, num_layers=39, num_heads=16,
        hidden_dim=1408, intermediate_dim=6144,
        use_patch_bias=True, use_class_token=True,
        use_mha_bias=True, use_mlp_bias=True,
        dropout_rate=0.0, layer_norm_epsilon=1e-6,
    )
    images = np.random.rand(2, 224, 224, 3).astype("float32")
    features = encoder(images)  # (2, 257, 1408)
    ```
    """

    def __init__(
        self,
        image_size,
        patch_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        use_patch_bias,
        use_class_token,
        use_mha_bias,
        use_mlp_bias,
        dropout_rate,
        layer_norm_epsilon,
        initializer_range,
        dtype=None,
        **kwargs,
    ):
        h, w = (image_size, image_size) if isinstance(image_size, int) else image_size
        ph, pw = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        # === Functional graph ===
        image_input = keras.Input(
            shape=(h, w, 3), name="images"
        )

        x = ViTPatchingAndEmbedding(
            image_size=(h, w),
            patch_size=(ph, pw),
            hidden_dim=hidden_dim,
            num_channels=3,
            use_class_token=use_class_token,
            use_patch_bias=use_patch_bias,
            dtype=dtype,
            name="patching_and_embedding",
        )(image_input)

        x = ViTEncoder(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=intermediate_dim,
            dropout_rate=dropout_rate,
            layer_norm_epsilon=layer_norm_epsilon,
            use_mha_bias=use_mha_bias,
            use_mlp_bias=use_mlp_bias,
            dtype=dtype,
            name="encoder",
        )(x)

        super().__init__(
            inputs=image_input,
            outputs=x,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.use_patch_bias = use_patch_bias
        self.use_class_token = use_class_token
        self.use_mha_bias = use_mha_bias
        self.use_mlp_bias = use_mlp_bias
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        # works for both square and non-square
        image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        self.num_vision_tokens_per_image = (
            (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        ) + (1 if use_class_token else 0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "use_patch_bias": self.use_patch_bias,
                "use_class_token": self.use_class_token,
                "use_mha_bias": self.use_mha_bias,
                "use_mlp_bias": self.use_mlp_bias,
                "dropout_rate": self.dropout_rate,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "initializer_range": self.initializer_range,
            }
        )
        return config
