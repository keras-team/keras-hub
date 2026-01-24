from keras import layers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.dinov2.dinov2_layers import DINOV2Embedding
from keras_hub.src.models.dinov2.dinov2_layers import DINOV2Encoder
from keras_hub.src.models.feature_pyramid_backbone import FeaturePyramidBackbone
from keras_hub.src.utils.keras_utils import standardize_data_format


@keras_hub_export("keras_hub.models.DINOV2Backbone")
class DINOV2Backbone(FeaturePyramidBackbone):
    """DINOV2 core network with hyperparameters.

    DINOV2 offers a powerful, generalist visual backbone learned entirely from
    unlabeled images as described in [DINOv2: Learning Robust Visual Features
    without Supervision](https://arxiv.org/abs/2304.07193)

    The default constructor gives a fully customizable, randomly initialized
    DINOV2 model with any number of layers, heads, and embedding dimensions. To
    load preset architectures and weights, use the `from_preset` constructor.

    Note that this backbone is a Feature Pyramid Backbone that can output
    intermediate feature maps from different stages of the model. See the
    example below for how to access these feature pyramid outputs.

    Note that this backbone supports interpolation of the position embeddings
    to the input image shape. This is useful when the input image shape is
    different from the shape used to train the position embeddings. The
    `position_embedding_shape` argument is used to specify the original shape
    used to train the position embeddings.

    Args:
        patch_size: int. The size of each square patch in the input image.
        num_layers: int. The number of transformer layers.
        hidden_dim: int. The size of the transformer hidden state at the end
            of each transformer layer.
        num_heads: int. The number of attention heads for each transformer.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        layer_scale_init_value: float. The initial value for the layer scale in
            the transformer layers. Defaults to `1.0`.
        num_register_tokens: int. The number of register tokens to use in the
            embedding layer. Defaults to `0`.
        use_mask_token: bool. Whether to use a mask token in the embedding
            layer. Defaults to `True`.
        use_swiglu_ffn: bool. Whether to use SwigLU activation in the MLP
            layers. Defaults to `False`.
        dropout_rate: float. The dropout rate to use. Defaults to `0.0`.
        drop_path_rate: float. The drop path rate to use. Defaults to `0.0`.
        image_shape: tuple. The input shape without the batch size. Defaults to
            `(224, 224, 3)`.
        position_embedding_shape: tuple. The original shape used to train the
            position embeddings. This is used to interpolate the position
            embeddings to the actual input shape. Defaults to `(518, 518)`.
        antialias_in_interpolation: bool. Whether to use antialiasing in the
            interpolation of the position embeddings. Defaults to `False`.
        apply_layernorm: bool. Whether to apply layer normalization to the
            outputs of each stage in the feature pyramid. Defaults to `False`.
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

    Example:
    ```python
    # Pretrained DINOV2 model.
    input_data = {
        "images": np.ones(shape=(1, 518, 518, 3), dtype="float32"),
    }
    model = keras_hub.models.DINOV2Backbone.from_preset(
        "dinov2_base"
    )
    model(input_data)

    # Pretrained DINOV2 model with custom image shape.
    input_data = {
        "images": np.ones(shape=(1, 224, 224, 3), dtype="float32"),
    }
    model = keras_hub.models.DINOV2Backbone.from_preset(
        "dinov2_base", image_shape=(224, 224, 3)
    )
    model(input_data)

    # Randomly initialized DINOV2 model with custom config.
    model = keras_hub.models.DINOV2Backbone(
        patch_size=14,
        num_layers=2,
        hidden_dim=32,
        num_heads=2,
        intermediate_dim=128,
        image_shape=(224, 224, 3),
        position_embedding_shape=(518, 518),
    )
    model(input_data)

    # Accessing feature pyramid outputs.
    backbone = keras_hub.models.DINOV2Backbone.from_preset(
        "dinov2_base", image_shape=(224, 224, 3)
    )
    model = keras.Model(
        inputs=backbone.inputs,
        outputs=backbone.pyramid_outputs,
    )
    features = model(input_data)
    ```
    """

    def __init__(
        self,
        patch_size,
        num_layers,
        hidden_dim,
        num_heads,
        intermediate_dim,
        layer_scale_init_value=1.0,
        num_register_tokens=0,
        use_mask_token=True,
        use_swiglu_ffn=False,
        dropout_rate=0.0,
        drop_path_rate=0.0,
        image_shape=(224, 224, 3),
        position_embedding_shape=(518, 518, 3),
        antialias_in_interpolation=False,
        apply_layernorm=False,
        data_format=None,
        dtype=None,
        name=None,
        **kwargs,
    ):
        data_format = standardize_data_format(data_format)
        if data_format == "channels_last":
            height, width = image_shape[0], image_shape[1]
            position_embedding_height, position_embedding_width = (
                position_embedding_shape[0],
                position_embedding_shape[1],
            )
        else:
            height, width = image_shape[1], image_shape[2]
            position_embedding_height, position_embedding_width = (
                position_embedding_shape[1],
                position_embedding_shape[2],
            )
        if height != width:
            raise ValueError(
                "`DINOV2Backbone` expects the height and width to be the "
                f"same in `image_shape`. Received: image_shape={image_shape}"
            )

        # `prefix` is used to prevent duplicate name when utilizing multiple
        # DINOV2Backbone encoders within a single model.
        prefix = str(name) + "_" if name is not None else ""

        # === Layers ===
        self.embeddings = DINOV2Embedding(
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            image_shape=(height, width),
            num_register_tokens=num_register_tokens,
            use_mask_token=use_mask_token,
            dropout_rate=dropout_rate,
            position_embedding_shape=(
                position_embedding_height,
                position_embedding_width,
            ),
            antialias_in_interpolation=antialias_in_interpolation,
            data_format=data_format,
            dtype=dtype,
            name=f"{prefix}embeddings",
        )
        self.encoder = DINOV2Encoder(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            intermediate_dim=intermediate_dim,
            layer_scale_init_value=layer_scale_init_value,
            use_swiglu_ffn=use_swiglu_ffn,
            dropout_rate=dropout_rate,
            drop_path_rate=drop_path_rate,
            dtype=dtype,
            name=f"{prefix}encoder",
        )
        self.layernorm = layers.LayerNormalization(
            epsilon=1e-6, dtype=dtype, name=f"{prefix}layernorm"
        )

        # === Functional Model ===
        pyramid_outputs = {}
        image_input = layers.Input(shape=image_shape, name="images")
        x = self.embeddings(image_input)
        pyramid_outputs["stem"] = x
        x, encoder_pyramid_outputs = self.encoder(x)
        pyramid_outputs.update(encoder_pyramid_outputs)
        x = self.layernorm(x)
        if apply_layernorm:
            for key in pyramid_outputs:
                pyramid_outputs[key] = self.layernorm(pyramid_outputs[key])
        outputs = x
        super().__init__(
            inputs={"images": image_input},
            outputs=outputs,
            dtype=dtype,
            name=name,
            **kwargs,
        )

        # === Config ===
        self.patch_size = int(patch_size)
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.intermediate_dim = int(intermediate_dim)
        self.layer_scale_init_value = float(layer_scale_init_value)
        self.num_register_tokens = int(num_register_tokens)
        self.use_mask_token = bool(use_mask_token)
        self.use_swiglu_ffn = bool(use_swiglu_ffn)
        self.dropout_rate = float(dropout_rate)
        self.drop_path_rate = float(drop_path_rate)
        self.image_shape = image_shape
        self.position_embedding_shape = position_embedding_shape
        self.antialias_in_interpolation = bool(antialias_in_interpolation)
        self.apply_layernorm = apply_layernorm
        self.pyramid_outputs = pyramid_outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "num_layers": self.num_layers,
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "layer_scale_init_value": self.layer_scale_init_value,
                "num_register_tokens": self.num_register_tokens,
                "use_mask_token": self.use_mask_token,
                "use_swiglu_ffn": self.use_swiglu_ffn,
                "dropout_rate": self.dropout_rate,
                "drop_path_rate": self.drop_path_rate,
                "image_shape": self.image_shape,
                "position_embedding_shape": self.position_embedding_shape,
                "antialias_in_interpolation": self.antialias_in_interpolation,
                "apply_layernorm": self.apply_layernorm,
            }
        )
        return config
