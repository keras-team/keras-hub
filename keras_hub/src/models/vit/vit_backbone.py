import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.vit.vit_layers import ViTEncoder
from keras_hub.src.models.vit.vit_layers import ViTPatchingAndEmbedding
from keras_hub.src.utils.keras_utils import standardize_data_format


@keras_hub_export("keras_hub.models.ViTBackbone")
class ViTBackbone(Backbone):
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
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        data_format = standardize_data_format(data_format)
        h_axis, w_axis = (
            (-3, -2) if data_format == "channels_last" else (-2, -1)
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

        # === Functional Model ===
        inputs = keras.layers.Input(shape=image_shape)

        x = ViTPatchingAndEmbedding(
            image_size=image_shape[h_axis],
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            dtype=dtype,
        )(inputs)

        output = ViTEncoder(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            layer_norm_epsilon=layer_norm_epsilon,
            dtype=dtype,
        )(x)

        super().__init__(
            inputs=inputs,
            outputs=output,
            **kwargs,
        )
