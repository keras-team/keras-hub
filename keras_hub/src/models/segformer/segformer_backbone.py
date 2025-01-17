import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone


@keras_hub_export("keras_hub.models.SegFormerBackbone")
class SegFormerBackbone(Backbone):
    """A Keras model implementing SegFormer for semantic segmentation.

    This class implements the majority of the SegFormer architecture described
    in [SegFormer: Simple and Efficient Design for Semantic Segmentation](https://arxiv.org/abs/2105.15203)
    and based on the TensorFlow implementation
    [from DeepVision](https://github.com/DavidLandup0/deepvision/tree/main/deepvision/models/segmentation/segformer).

    SegFormers are meant to be used with the MixTransformer (MiT) encoder
    family, and use a very lightweight all-MLP decoder head.

    The MiT encoder uses a hierarchical transformer which outputs features at
    multiple scales, similar to that of the hierarchical outputs typically
    associated with CNNs.

    Args:
        image_encoder: `keras.Model`. The backbone network for the model that is
            used as a feature extractor for the SegFormer encoder.
            Should be used with the MiT backbone model
            (`keras_hub.models.MiTBackbone`) which was created
            specifically for SegFormers.
        num_classes: int, the number of classes for the detection model,
            including the background class.
        projection_filters: int, number of filters in the
            convolution layer projecting the concatenated features into
            a segmentation map. Defaults to 256`.

    Example:

    Using the class with a custom `backbone`:

    ```python
    import keras_hub

    backbone = keras_hub.models.MiTBackbone(
        depths=[2, 2, 2, 2],
        image_shape=(224, 224, 3),
        hidden_dims=[32, 64, 160, 256],
        num_layers=4,
        blockwise_num_heads=[1, 2, 5, 8],
        blockwise_sr_ratios=[8, 4, 2, 1],
        max_drop_path_rate=0.1,
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
    )

    segformer_backbone = keras_hub.models.SegFormerBackbone(
        image_encoder=backbone, projection_filters=256)
    ```

    Using the class with a preset `backbone`:

    ```python
    import keras_hub

    backbone = keras_hub.models.MiTBackbone.from_preset("mit_b0_ade20k_512")
    segformer_backbone = keras_hub.models.SegFormerBackbone(
        image_encoder=backbone, projection_filters=256)
    ```

    """

    def __init__(
        self,
        image_encoder,
        projection_filters,
        **kwargs,
    ):
        if not isinstance(image_encoder, keras.layers.Layer) or not isinstance(
            image_encoder, keras.Model
        ):
            raise ValueError(
                "Argument `image_encoder` must be a `keras.layers.Layer` "
                f"instance or `keras.Model`. Received instead "
                f"image_encoder={image_encoder} "
                f"(of type {type(image_encoder)})."
            )

        # === Layers ===
        inputs = keras.layers.Input(shape=image_encoder.input.shape[1:])

        self.feature_extractor = keras.Model(
            image_encoder.inputs, image_encoder.pyramid_outputs
        )

        features = self.feature_extractor(inputs)
        # Get height and width of level one output
        _, height, width, _ = features["P1"].shape

        self.mlp_blocks = []

        for feature_dim, feature in zip(image_encoder.hidden_dims, features):
            self.mlp_blocks.append(
                keras.layers.Dense(
                    projection_filters, name=f"linear_{feature_dim}"
                )
            )

        self.resizing = keras.layers.Resizing(
            height, width, interpolation="bilinear"
        )
        self.concat = keras.layers.Concatenate(axis=-1)
        self.linear_fuse = keras.Sequential(
            [
                keras.layers.Conv2D(
                    filters=projection_filters, kernel_size=1, use_bias=False
                ),
                keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9),
                keras.layers.Activation("relu"),
            ]
        )

        # === Functional Model ===
        # Project all multi-level outputs onto
        # the same dimensionality and feature map shape
        multi_layer_outs = []
        for index, (feature_dim, feature) in enumerate(
            zip(image_encoder.hidden_dims, features)
        ):
            out = self.mlp_blocks[index](features[feature])
            out = self.resizing(out)
            multi_layer_outs.append(out)

        # Concat now-equal feature maps
        concatenated_outs = self.concat(multi_layer_outs[::-1])

        # Fuse concatenated features into a segmentation map
        seg = self.linear_fuse(concatenated_outs)

        super().__init__(
            inputs=inputs,
            outputs=seg,
            **kwargs,
        )

        # === Config ===
        self.projection_filters = projection_filters
        self.image_encoder = image_encoder

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "projection_filters": self.projection_filters,
                "image_encoder": keras.saving.serialize_keras_object(
                    self.image_encoder
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if "image_encoder" in config and isinstance(
            config["image_encoder"], dict
        ):
            config["image_encoder"] = keras.layers.deserialize(
                config["image_encoder"]
            )
        return super().from_config(config)
