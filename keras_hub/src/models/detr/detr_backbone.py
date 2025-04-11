from keras import layers
from keras import ops
from src.models.detr.detr_layers import DETRTransformer
from src.models.detr.detr_layers import position_embedding_sine

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone


def _freeze_batch_norm(model):
    """DETR uses "frozen" batch norm, i.e. batch normalization
    with zeros and ones as the parameters, and they don't get adjusted
    during training. This was done through a custom class.

    Since it's tricky to exchange all BatchNormalization layers
    in an existing model with FrozenBatchNormalization, we just
    make them untrainable and assign the "frozen" parameters.
    """
    for layer in model.layers:
        if isinstance(layer, layers.BatchNormalization):
            # Disable training of the layer
            layer.trainable = False
            # Set the layer to inference mode
            layer._trainable = False
            # Manually freeze weights and stats
            layer.gamma.assign(ops.ones_like(layer.gamma))
            layer.beta.assign(ops.zeros_like(layer.beta))
            layer.moving_mean.assign(ops.zeros_like(layer.moving_mean))
            layer.moving_variance.assign(ops.ones_like(layer.moving_variance))

    return model


@keras_hub_export("keras_hub.models.DETR")
class DETR(Backbone):
    """A Keras model implementing DETR for object detection.

    This class implements the majority of the DETR architecture described
    in [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
    and based on the [TensorFlow implementation]
    (https://github.com/tensorflow/models/tree/master/official/projects/detr).

    DETR is meant to be used with a modified ResNet50 backbone/encoder.

    Args:
        image_encoder: `keras.Model`. The backbone network for the model that is
            used as a feature extractor for the SegFormer encoder.
            Should be used with
            `keras_hub.models.ResNetBackbone.from_preset("resnet_50_imagenet")`.
        ...

    Examples:

    ```
    # todo
    ```

    """

    def __init__(
        self,
        backbone,
        num_queries,
        hidden_size,
        num_classes,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dropout_rate=0.1,
        **kwargs,
    ):
        # === Layers ===
        inputs = layers.Input(shape=backbone.input.shape[1:])

        input_proj = layers.Conv2D(hidden_size, 1, name="conv2d")
        transformer = DETRTransformer(
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout_rate=dropout_rate,
        )
        # query_embeddings = self.add_weight(
        #    shape=[num_queries, hidden_size],
        # )
        # cannot call self.add_weight before super()
        # TODO: look into how to work around this.
        # for the time being, initialize query_embeddings
        # as a static vector
        query_embeddings = ops.ones([num_queries, hidden_size])

        class_embed = layers.Dense(num_classes, name="cls_dense")
        bbox_embed = [
            layers.Dense(hidden_size, activation="relu", name="box_dense_0"),
            layers.Dense(hidden_size, activation="relu", name="box_dense_1"),
            layers.Dense(4, name="box_dense_2"),
        ]

        # === Functional Model ===
        batch_size = ops.shape(inputs)[0]
        features = backbone(inputs)
        shape = ops.shape(features)
        mask = self._generate_image_mask(inputs, shape[1:3])

        pos_embed = position_embedding_sine(
            mask[:, :, :, 0], num_pos_features=hidden_size
        )
        pos_embed = ops.reshape(pos_embed, [batch_size, -1, hidden_size])

        features = ops.reshape(
            input_proj(features), [batch_size, -1, hidden_size]
        )
        mask = ops.reshape(mask, [batch_size, -1])

        decoded_list = transformer(
            {
                "inputs": features,
                "targets": ops.tile(
                    ops.expand_dims(query_embeddings, axis=0),
                    (batch_size, 1, 1),
                ),
                "pos_embed": pos_embed,
                "mask": mask,
            }
        )
        out_list = []
        for decoded in decoded_list:
            decoded = ops.stack(decoded)
            output_class = class_embed(decoded)
            box_out = decoded
            for layer in bbox_embed:
                box_out = layer(box_out)
            output_coord = layers.Activation("sigmoid")(box_out)
            out = {"cls_outputs": output_class, "box_outputs": output_coord}
            out_list.append(out)

        super().__init__(
            inputs=inputs,
            outputs=out_list,
            **kwargs,
        )

        # === Config ===
        self.num_queries = num_queries
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout_rate = dropout_rate
        if hidden_size % 2 != 0:
            raise ValueError("hidden_size must be a multiple of 2.")
        self.backbone = backbone

    def get_config(self):
        return {
            "backbone": self.backbone,
            "num_queries": self.num_queries,
            "hidden_size": self.hidden_size,
            "num_classes": self.num_classes,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "dropout_rate": self.dropout_rate,
        }

    @property
    def backbone(self):
        return self.backbone

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape=None):
        self.build_detection_decoder()
        super().build(input_shape)

    def _generate_image_mask(self, inputs, target_shape):
        """Generates image mask from input image."""
        mask = ops.expand_dims(
            ops.cast(ops.not_equal(ops.sum(inputs, axis=-1), 0), inputs.dtype),
            axis=-1,
        )
        mask = ops.image.resize(mask, target_shape, interpolation="nearest")
        return mask
