import keras
from keras import layers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone


@keras_hub_export("keras_hub.models.DiffBinBackbone")
class DiffBinBackbone(Backbone):
    """Differentiable Binarization architecture for scene text detection.

    This class implements the Differentiable Binarization architecture for
    detecting text in natural images, described in
    [Real-time Scene Text Detection with Differentiable Binarization](
    https://arxiv.org/abs/1911.08947).

    The backbone architecture in this class contains the feature pyramid
    network and model heads.

    Args:
        image_encoder: A `keras_hub.models.ResNetBackbone` instance.
        fpn_channels: int. The number of channels to output by the feature
            pyramid network. Defaults to 256.
        head_kernel_list: list of ints. The kernel sizes of probability map and
            threshold map heads. Defaults to [3, 2, 2].
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the model's computations and weights.
    """

    def __init__(
        self,
        image_encoder,
        fpn_channels=256,
        head_kernel_list=[3, 2, 2],
        dtype=None,
        **kwargs,
    ):
        # === Functional Model ===
        inputs = image_encoder.input
        x = image_encoder.pyramid_outputs
        x = diffbin_fpn_model(x, out_channels=fpn_channels, dtype=dtype)

        probability_maps = diffbin_head(
            x,
            in_channels=fpn_channels,
            kernel_list=head_kernel_list,
            name="head_prob",
        )
        threshold_maps = diffbin_head(
            x,
            in_channels=fpn_channels,
            kernel_list=head_kernel_list,
            name="head_thresh",
        )

        outputs = {
            "probability_maps": probability_maps,
            "threshold_maps": threshold_maps,
        }

        super().__init__(inputs=inputs, outputs=outputs, dtype=dtype, **kwargs)

        # === Config ===
        self.image_encoder = image_encoder
        self.fpn_channels = fpn_channels
        self.head_kernel_list = head_kernel_list

    def get_config(self):
        config = super().get_config()
        config["fpn_channels"] = self.fpn_channels
        config["head_kernel_list"] = self.head_kernel_list
        config["image_encoder"] = keras.layers.serialize(self.image_encoder)
        return config

    @classmethod
    def from_config(cls, config):
        config["image_encoder"] = keras.layers.deserialize(
            config["image_encoder"]
        )
        return cls(**config)


def diffbin_fpn_model(inputs, out_channels, dtype=None):
    # lateral layers composing the FPN's bottom-up pathway using
    # pointwise convolutions of ResNet's pyramid outputs
    lateral_p2 = layers.Conv2D(
        out_channels,
        kernel_size=1,
        use_bias=False,
        name="neck_lateral_p2",
        dtype=dtype,
    )(inputs["P2"])
    lateral_p3 = layers.Conv2D(
        out_channels,
        kernel_size=1,
        use_bias=False,
        name="neck_lateral_p3",
        dtype=dtype,
    )(inputs["P3"])
    lateral_p4 = layers.Conv2D(
        out_channels,
        kernel_size=1,
        use_bias=False,
        name="neck_lateral_p4",
        dtype=dtype,
    )(inputs["P4"])
    lateral_p5 = layers.Conv2D(
        out_channels,
        kernel_size=1,
        use_bias=False,
        name="neck_lateral_p5",
        dtype=dtype,
    )(inputs["P5"])
    # top-down fusion pathway consisting of upsampling layers with
    # skip connections
    topdown_p5 = lateral_p5
    topdown_p4 = layers.Add(name="neck_topdown_p4")(
        [
            layers.UpSampling2D(dtype=dtype)(topdown_p5),
            lateral_p4,
        ]
    )
    topdown_p3 = layers.Add(name="neck_topdown_p3")(
        [
            layers.UpSampling2D(dtype=dtype)(topdown_p4),
            lateral_p3,
        ]
    )
    topdown_p2 = layers.Add(name="neck_topdown_p2")(
        [
            layers.UpSampling2D(dtype=dtype)(topdown_p3),
            lateral_p2,
        ]
    )
    # construct merged feature maps for each pyramid level
    featuremap_p5 = layers.Conv2D(
        out_channels // 4,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="neck_featuremap_p5",
        dtype=dtype,
    )(topdown_p5)
    featuremap_p4 = layers.Conv2D(
        out_channels // 4,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="neck_featuremap_p4",
        dtype=dtype,
    )(topdown_p4)
    featuremap_p3 = layers.Conv2D(
        out_channels // 4,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="neck_featuremap_p3",
        dtype=dtype,
    )(topdown_p3)
    featuremap_p2 = layers.Conv2D(
        out_channels // 4,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="neck_featuremap_p2",
        dtype=dtype,
    )(topdown_p2)
    featuremap_p5 = layers.UpSampling2D((8, 8), dtype=dtype)(featuremap_p5)
    featuremap_p4 = layers.UpSampling2D((4, 4), dtype=dtype)(featuremap_p4)
    featuremap_p3 = layers.UpSampling2D((2, 2), dtype=dtype)(featuremap_p3)
    featuremap = layers.Concatenate(axis=-1, dtype=dtype)(
        [featuremap_p5, featuremap_p4, featuremap_p3, featuremap_p2]
    )
    return featuremap


def diffbin_head(inputs, in_channels, kernel_list, name):
    x = layers.Conv2D(
        in_channels // 4,
        kernel_size=kernel_list[0],
        padding="same",
        use_bias=False,
        name=f"{name}_conv0_weights",
    )(inputs)
    x = layers.BatchNormalization(
        beta_initializer=keras.initializers.Constant(1e-4),
        gamma_initializer=keras.initializers.Constant(1.0),
        name=f"{name}_conv0_bn",
    )(x)
    x = layers.ReLU(name=f"{name}_conv0_relu")(x)
    x = layers.Conv2DTranspose(
        in_channels // 4,
        kernel_size=kernel_list[1],
        strides=2,
        padding="valid",
        bias_initializer=keras.initializers.RandomUniform(
            minval=-1.0 / (in_channels // 4 * 1.0) ** 0.5,
            maxval=1.0 / (in_channels // 4 * 1.0) ** 0.5,
        ),
        name=f"{name}_conv1_weights",
    )(x)
    x = layers.BatchNormalization(
        beta_initializer=keras.initializers.Constant(1e-4),
        gamma_initializer=keras.initializers.Constant(1.0),
        name=f"{name}_conv1_bn",
    )(x)
    x = layers.ReLU(name=f"{name}_conv1_relu")(x)
    x = layers.Conv2DTranspose(
        1,
        kernel_size=kernel_list[2],
        strides=2,
        padding="valid",
        activation="sigmoid",
        bias_initializer=keras.initializers.RandomUniform(
            minval=-1.0 / (in_channels // 4 * 1.0) ** 0.5,
            maxval=1.0 / (in_channels // 4 * 1.0) ** 0.5,
        ),
        name=f"{name}_conv2_weights",
    )(x)
    return x
