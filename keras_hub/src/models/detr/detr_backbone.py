import math

from keras import Model
from keras import layers
from keras import ops
from src.models.detr.detr_layers import DetrSinePositionEmbedding
from src.models.detr.detr_layers import DETRTransformer


class DETR(Model):
    """DETR Model.

    Includes a backbone (ResNet50), query embedding,
    DETRTransformer (DetrTransformerEncoder + DetrTransformerDecoder)
    class and box heads.
    """

    def __init__(
        self,
        backbone,
        backbone_endpoint_name,
        num_queries,
        hidden_size,
        num_classes,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dropout_rate=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._num_queries = num_queries
        self._hidden_size = hidden_size
        self._num_classes = num_classes
        self._num_encoder_layers = num_encoder_layers
        self._num_decoder_layers = num_decoder_layers
        self._dropout_rate = dropout_rate
        if hidden_size % 2 != 0:
            raise ValueError("hidden_size must be a multiple of 2.")
        self._backbone = backbone
        self._backbone_endpoint_name = backbone_endpoint_name

    def build(self, input_shape=None):
        self._input_proj = layers.Conv2D(
            self._hidden_size, 1, name="detr/conv2d"
        )
        self._build_detection_decoder()
        super().build(input_shape)

    def _build_detection_decoder(self):
        """Builds detection decoder."""
        self._transformer = DETRTransformer(
            num_encoder_layers=self._num_encoder_layers,
            num_decoder_layers=self._num_decoder_layers,
            dropout_rate=self._dropout_rate,
        )
        self._query_embeddings = self.add_weight(
            "detr/query_embeddings",
            shape=[self._num_queries, self._hidden_size],
        )
        sqrt_k = math.sqrt(1.0 / self._hidden_size)
        self._class_embed = layers.layers.Dense(
            self._num_classes, name="detr/cls_dense"
        )
        self._bbox_embed = [
            layers.Dense(
                self._hidden_size, activation="relu", name="detr/box_dense_0"
            ),
            layers.Dense(
                self._hidden_size, activation="relu", name="detr/box_dense_1"
            ),
            layers.Dense(4, name="detr/box_dense_2"),
        ]
        self._sigmoid = layers.Activation("sigmoid")

    @property
    def backbone(self):
        return self._backbone

    def get_config(self):
        return {
            "backbone": self._backbone,
            "backbone_endpoint_name": self._backbone_endpoint_name,
            "num_queries": self._num_queries,
            "hidden_size": self._hidden_size,
            "num_classes": self._num_classes,
            "num_encoder_layers": self._num_encoder_layers,
            "num_decoder_layers": self._num_decoder_layers,
            "dropout_rate": self._dropout_rate,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def _generate_image_mask(self, inputs, target_shape):
        """Generates image mask from input image."""
        mask = ops.expand_dims(
            ops.cast(ops.not_equal(ops.sum(inputs, axis=-1), 0), inputs.dtype),
            axis=-1,
        )
        mask = tf.image.resize(
            mask, target_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        return mask

    def call(self, inputs, training=None):
        batch_size = ops.shape(inputs)[0]
        features = self._backbone(inputs)[self._backbone_endpoint_name]
        shape = ops.shape(features)
        mask = self._generate_image_mask(inputs, shape[1:3])

        pos_embed = DetrSinePositionEmbedding(embedding_dim=self._hidden_size)(
            mask[:, :, :, 0]
        )
        pos_embed = ops.reshape(pos_embed, [batch_size, -1, self._hidden_size])

        features = ops.reshape(
            self._input_proj(features), [batch_size, -1, self._hidden_size]
        )
        mask = ops.reshape(mask, [batch_size, -1])

        decoded_list = self._transformer(
            {
                "inputs": features,
                "targets": ops.tile(
                    ops.expand_dims(self._query_embeddings, axis=0),
                    (batch_size, 1, 1),
                ),
                "pos_embed": pos_embed,
                "mask": mask,
            }
        )
        out_list = []
        for decoded in decoded_list:
            decoded = ops.stack(decoded)
            output_class = self._class_embed(decoded)
            box_out = decoded
            for layer in self._bbox_embed:
                box_out = layer(box_out)
            output_coord = self._sigmoid(box_out)
            out = {"cls_outputs": output_class, "box_outputs": output_coord}
            out_list.append(out)

        return out_list
