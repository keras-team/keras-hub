from keras import layers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.dinov3.dinov3_layers import DINOV3Embedding
from keras_hub.src.models.dinov3.dinov3_layers import DINOV3Encoder
from keras_hub.src.models.dinov3.dinov3_layers import (
    DINOV3RopePositionEmbedding,
)
from keras_hub.src.models.feature_pyramid_backbone import FeaturePyramidBackbone
from keras_hub.src.utils.keras_utils import standardize_data_format


@keras_hub_export("keras_hub.models.DINOV3Backbone")
class DINOV3Backbone(FeaturePyramidBackbone):
    """DINOV3 core network with hyperparameters.

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
        hidden_activation: str or callable. Activation to use in the MLP.
            Defaults to `"gelu"`.
        use_gated_mlp: bool. Whether to use Gated MLP layers. Defaults to
            `False`.
        use_query_bias: bool. Whether to use a bias for the query projection.
            Defaults to `True`.
        use_key_bias: bool. Whether to use a bias for the key projection.
            Defaults to `True`.
        use_value_bias: bool. Whether to use a bias for the value projection.
            Defaults to `True`.
        use_proj_bias: bool. Whether to use a bias for the output projection.
            Defaults to `True`.
        use_mlp_bias: bool. Whether to use a bias for the dense layers in MLP.
            Defaults to `True`.
        attention_dropout: float. The dropout rate for the attention
            probabilities. Defaults to `0.0`.
        drop_path_rate: float. The drop path rate to use. Defaults to `0.0`.
        image_shape: tuple. The input shape without the batch size. Defaults to
            `(518, 518, 3)`.
        rope_theta: float. The base period of the rotary position embeddings.
            Defaults to `100.0`.
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
    # Pretrained DINOV3 model.
    input_data = {
        "images": np.ones(shape=(1, 518, 518, 3), dtype="float32"),
    }
    model = keras_hub.models.DINOV3Backbone.from_preset(
        "dinov3_vit_small_lvd1689m"
    )
    model(input_data)

    # Pretrained DINOV3 model with custom image shape.
    input_data = {
        "images": np.ones(shape=(1, 224, 224, 3), dtype="float32"),
    }
    model = keras_hub.models.DINOV3Backbone.from_preset(
        "dinov3_vit_small_lvd1689m", image_shape=(224, 224, 3)
    )
    model(input_data)

    # Randomly initialized DINOV3 model with custom config.
    model = keras_hub.models.DINOV3Backbone(
        patch_size=14,
        num_layers=2,
        hidden_dim=32,
        num_heads=2,
        intermediate_dim=128,
        image_shape=(224, 224, 3),
    )
    model(input_data)

    # Accessing feature pyramid outputs.
    backbone = keras_hub.models.DINOV3Backbone.from_preset(
        "dinov3_vit_small_lvd1689m", image_shape=(224, 224, 3)
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
        num_register_tokens=4,
        use_mask_token=True,
        hidden_activation="gelu",
        use_gated_mlp=False,
        use_query_bias=True,
        use_key_bias=True,
        use_value_bias=True,
        use_proj_bias=True,
        use_mlp_bias=True,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        layer_norm_eps=1e-5,
        image_shape=(518, 518, 3),
        rope_theta=100.0,
        apply_layernorm=False,
        data_format=None,
        dtype=None,
        name=None,
        **kwargs,
    ):
        data_format = standardize_data_format(data_format)

        prefix = str(name) + "_" if name is not None else ""

        # === Layers ===
        self.embeddings = DINOV3Embedding(
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            use_mask_token=use_mask_token,
            data_format=data_format,
            dtype=dtype,
            name=f"{prefix}embeddings",
        )
        self.rope_embedding = DINOV3RopePositionEmbedding(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            rope_theta=rope_theta,
            patch_size=patch_size,
            dtype=dtype,
            name=f"{prefix}rope_embedding",
        )
        self.encoder = DINOV3Encoder(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            intermediate_dim=intermediate_dim,
            layer_scale_init_value=layer_scale_init_value,
            hidden_activation=hidden_activation,
            use_gated_mlp=use_gated_mlp,
            use_query_bias=use_query_bias,
            use_key_bias=use_key_bias,
            use_value_bias=use_value_bias,
            use_proj_bias=use_proj_bias,
            use_mlp_bias=use_mlp_bias,
            attention_dropout=attention_dropout,
            drop_path_rate=drop_path_rate,
            layer_norm_eps=layer_norm_eps,
            dtype=dtype,
            name=f"{prefix}encoder",
        )
        self.layernorm = layers.LayerNormalization(
            epsilon=layer_norm_eps, dtype=dtype, name=f"{prefix}layernorm"
        )

        # === Functional Model ===
        pyramid_outputs = {}
        image_input = layers.Input(shape=image_shape, name="pixel_values")
        x = self.embeddings(image_input)
        pyramid_outputs["stem"] = x

        position_embeddings = self.rope_embedding(image_input)
        num_prefix_tokens = 1 + num_register_tokens

        x, encoder_pyramid_outputs = self.encoder(
            x,
            position_embeddings=position_embeddings,
            num_prefix_tokens=num_prefix_tokens,
        )
        pyramid_outputs.update(encoder_pyramid_outputs)
        x = self.layernorm(x)
        if apply_layernorm:
            for key in pyramid_outputs:
                pyramid_outputs[key] = self.layernorm(pyramid_outputs[key])
        outputs = x
        super().__init__(
            inputs={"pixel_values": image_input},
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
        self.hidden_activation = hidden_activation
        self.use_gated_mlp = bool(use_gated_mlp)
        self.use_query_bias = bool(use_query_bias)
        self.use_key_bias = bool(use_key_bias)
        self.use_value_bias = bool(use_value_bias)
        self.use_proj_bias = bool(use_proj_bias)
        self.use_mlp_bias = bool(use_mlp_bias)
        self.attention_dropout = float(attention_dropout)
        self.drop_path_rate = float(drop_path_rate)
        self.layer_norm_eps = float(layer_norm_eps)
        self.image_shape = image_shape
        self.rope_theta = rope_theta
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
                "num_register_tokens": self.num_register_tokens,
                "use_mask_token": self.use_mask_token,
                "layer_scale_init_value": self.layer_scale_init_value,
                "hidden_activation": self.hidden_activation,
                "use_gated_mlp": self.use_gated_mlp,
                "use_query_bias": self.use_query_bias,
                "use_key_bias": self.use_key_bias,
                "use_value_bias": self.use_value_bias,
                "use_proj_bias": self.use_proj_bias,
                "use_mlp_bias": self.use_mlp_bias,
                "attention_dropout": self.attention_dropout,
                "drop_path_rate": self.drop_path_rate,
                "layer_norm_eps": self.layer_norm_eps,
                "image_shape": self.image_shape,
                "rope_theta": self.rope_theta,
                "apply_layernorm": self.apply_layernorm,
            }
        )
        return config
