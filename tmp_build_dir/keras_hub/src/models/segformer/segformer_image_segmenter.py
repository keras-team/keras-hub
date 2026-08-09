import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_segmenter import ImageSegmenter
from keras_hub.src.models.segformer.segformer_backbone import SegFormerBackbone
from keras_hub.src.models.segformer.segformer_image_segmenter_preprocessor import (  # noqa: E501
    SegFormerImageSegmenterPreprocessor,
)


@keras_hub_export("keras_hub.models.SegFormerImageSegmenter")
class SegFormerImageSegmenter(ImageSegmenter):
    """A Keras model implementing SegFormer for semantic segmentation.

    This class implements the segmentation head of the SegFormer architecture
    described in [SegFormer: Simple and Efficient Design for Semantic
    Segmentation with Transformers] (https://arxiv.org/abs/2105.15203) and
    [based on the TensorFlow implementation from DeepVision]
    (https://github.com/DavidLandup0/deepvision/tree/main/deepvision/models/segmentation/segformer).

    SegFormers are meant to be used with the MixTransformer (MiT) encoder
    family, and and use a very lightweight all-MLP decoder head.

    The MiT encoder uses a hierarchical transformer which outputs features at
    multiple scales, similar to that of the hierarchical outputs typically
    associated with CNNs.

    Args:
        image_encoder: `keras.Model`. The backbone network for the model that is
            used as a feature extractor for the SegFormer encoder. It is
            *intended* to be used only with the MiT backbone model
            (`keras_hub.models.MiTBackbone`) which was created specifically for
            SegFormers. Alternatively, can be a `keras_hub.models.Backbone` a
            model subclassing `keras_hub.models.FeaturePyramidBackbone`, or a
            `keras.Model` that has a `pyramid_outputs` property which is a
            dictionary with keys "P2", "P3", "P4", and "P5" and layer names as
            values.
        num_classes: int, the number of classes for the detection model,
            including the background class.
        projection_filters: int, number of filters in the
            convolution layer projecting the concatenated features into a
            segmentation map. Defaults to 256`.


    Example:

    Using presets:

    ```python
    segmenter = keras_hub.models.SegFormerImageSegmenter.from_preset(
        "segformer_b0_ade20k_512"
    )

    images = np.random.rand(1, 512, 512, 3)
    segformer(images)
    ```

    Using the SegFormer backbone:

    ```python
    encoder = keras_hub.models.MiTBackbone.from_preset(
        "mit_b0_ade20k_512"
    )
    backbone = keras_hub.models.SegFormerBackbone(
        image_encoder=encoder,
        projection_filters=256,
    )
    ```

    Using the SegFormer backbone with a custom encoder:

    ```python
    images = np.ones(shape=(1, 96, 96, 3))
    labels = np.zeros(shape=(1, 96, 96, 1))

    encoder = keras_hub.models.MiTBackbone(
        depths=[2, 2, 2, 2],
        image_shape=(96, 96, 3),
        hidden_dims=[32, 64, 160, 256],
        num_layers=4,
        blockwise_num_heads=[1, 2, 5, 8],
        blockwise_sr_ratios=[8, 4, 2, 1],
        max_drop_path_rate=0.1,
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
    )

    backbone = keras_hub.models.SegFormerBackbone(
        image_encoder=encoder,
        projection_filters=256,
    )
    segformer = keras_hub.models.SegFormerImageSegmenter(
        backbone=backbone,
        num_classes=4,
    )
    segformer(images
    ```

    Using the segmentor class with a preset backbone:

    ```python
    image_encoder = keras_hub.models.MiTBackbone.from_preset(
        "mit_b0_ade20k_512"
    )
    backbone = keras_hub.models.SegFormerBackbone(
        image_encoder=encoder,
        projection_filters=256,
    )
    segformer = keras_hub.models.SegFormerImageSegmenter(
        backbone=backbone,
        num_classes=4,
    )
    ```
    """

    backbone_cls = SegFormerBackbone
    preprocessor_cls = SegFormerImageSegmenterPreprocessor

    def __init__(
        self,
        backbone,
        num_classes,
        preprocessor=None,
        **kwargs,
    ):
        if not isinstance(backbone, keras.layers.Layer) or not isinstance(
            backbone, keras.Model
        ):
            raise ValueError(
                "Argument `backbone` must be a `keras.layers.Layer` instance "
                f" or `keras.Model`. Received instead "
                f"backbone={backbone} (of type {type(backbone)})."
            )

        # === Layers ===
        inputs = backbone.input

        self.backbone = backbone
        self.preprocessor = preprocessor
        self.dropout = keras.layers.Dropout(0.1)
        self.output_segmentation_head = keras.layers.Conv2D(
            filters=num_classes, kernel_size=1, strides=1
        )
        self.resizing = keras.layers.Resizing(
            height=inputs.shape[1],
            width=inputs.shape[2],
            interpolation="bilinear",
        )

        # === Functional Model ===
        x = self.backbone(inputs)
        x = self.dropout(x)
        x = self.output_segmentation_head(x)
        output = self.resizing(x)

        super().__init__(
            inputs=inputs,
            outputs=output,
            **kwargs,
        )

        # === Config ===
        self.num_classes = num_classes
        self.backbone = backbone

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "backbone": keras.saving.serialize_keras_object(self.backbone),
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
