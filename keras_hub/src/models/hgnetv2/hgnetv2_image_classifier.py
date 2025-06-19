import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.hgnetv2.hgnetv2_backbone import HGNetV2Backbone
from keras_hub.src.models.hgnetv2.hgnetv2_image_classifier_preprocessor import (
    HGNetV2ImageClassifierPreprocessor,
)
from keras_hub.src.models.hgnetv2.hgnetv2_layers import HGNetV2ConvLayer
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.task import Task


@keras_hub_export("keras_hub.models.HGNetV2ImageClassifier")
class HGNetV2ImageClassifier(ImageClassifier):
    backbone_cls = HGNetV2Backbone
    preprocessor_cls = HGNetV2ImageClassifierPreprocessor

    def __init__(
        self,
        backbone,
        preprocessor,
        num_classes,
        head_filters,
        pooling="avg",
        activation=None,
        dropout=0.0,
        head_dtype=None,
        use_learnable_affine_block_head=False,
        **kwargs,
    ):
        name = kwargs.get("name", "hgnetv2_image_classifier")
        head_dtype = head_dtype or backbone.dtype_policy
        data_format = getattr(backbone, "data_format", "channels_last")
        channel_axis = -1 if data_format == "channels_last" else 1

        # NOTE: This isn't in the usual order because the config is needed
        # before layer initialization and the functional model.
        # === Config ===
        self.num_classes = num_classes
        self.pooling = pooling
        self.activation = activation
        self.dropout = dropout
        self.head_filters = head_filters

        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor
        self.last_conv = HGNetV2ConvLayer(
            in_channels=backbone.hidden_sizes[-1],
            out_channels=self.head_filters,
            kernel_size=1,
            stride=1,
            groups=1,
            activation="relu",
            use_learnable_affine_block=use_learnable_affine_block_head,
            data_format=data_format,
            channel_axis=channel_axis,
            name="head_last",
            dtype=head_dtype,
        )
        if self.pooling == "avg":
            self.pooler = keras.layers.GlobalAveragePooling2D(
                data_format=data_format,
                dtype=head_dtype,
                name=f"{name}_avg_pool" if name else "avg_pool",
            )
        elif self.pooling == "max":
            self.pooler = keras.layers.GlobalMaxPooling2D(
                data_format=data_format,
                dtype=head_dtype,
                name=f"{name}_max_pool" if name else "max_pool",
            )
        # Check valid pooling.
        else:
            raise ValueError(
                "Unknown `pooling` type. Polling should be either `'avg'` or "
                f"`'max'`. Received: pooling={pooling}."
            )

        self.flatten_layer = keras.layers.Flatten(
            dtype=head_dtype,
            name=f"{name}_flatten" if name else "flatten",
        )
        self.output_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=head_dtype,
            name=f"{name}_output_dropout" if name else "output_dropout",
        )
        if self.num_classes > 0:
            self.output_dense = keras.layers.Dense(
                units=num_classes,
                activation=activation,
                dtype=head_dtype,
                name="predictions",
            )
        else:
            self.output_dense = keras.layers.Identity(name="predictions")

        # === Functional Model ===
        inputs = backbone.input
        feature_maps = backbone(inputs)
        last_stage_name = backbone.stage_names[-1]
        last_hidden_state_for_pooling = feature_maps[last_stage_name]
        x = self.last_conv(last_hidden_state_for_pooling)
        x = self.pooler(x)
        x = self.flatten_layer(x)
        x = self.output_dropout(x)
        outputs = self.output_dense(x)
        Task.__init__(
            self,
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

    def get_config(self):
        config = Task.get_config(self)
        config.update(
            {
                "num_classes": self.num_classes,
                "pooling": self.pooling,
                "activation": self.activation,
                "dropout": self.dropout,
                "head_filters": self.head_filters,
            }
        )
        return config
