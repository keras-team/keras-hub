from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.detr.detr_backbone import DETRBackbone
from keras_hub.src.models.detr.detr_layers import CreateCrossAttentionMask
from keras_hub.src.models.detr.detr_layers import CreateSelfAttentionMask
from keras_hub.src.models.detr.detr_layers import DETRMLPPredictionHead
from keras_hub.src.models.detr.detr_layers import DetrQueryEmbedding
from keras_hub.src.models.detr.detr_layers import DetrTransformerDecoder
from keras_hub.src.models.object_detector import ObjectDetector
from keras_hub.src.models.task import Task


@keras_hub_export("keras_hub.models.DETRObjectDetector")
class DETRObjectDetector(ObjectDetector):
    """DETR (DEtection TRansformer) object detection model.

    This class implements the DETR architecture from the paper
    "End-to-End Object Detection with Transformers" (https://arxiv.org/abs/2005.12872).

    DETR treats object detection as a direct set prediction problem using a
    transformer encoder-decoder architecture with learnable object queries.
    Unlike traditional detectors, it requires no hand-designed components like
    anchor generation or non-maximum suppression.

    Args:
        backbone: `DETRBackbone`. The backbone network that extracts features
            and encodes them with a transformer encoder.
        num_queries: int. Number of learnable object queries. Each query
            will predict one object (or no-object). Default: 100.
        num_classes: int. Number of object classes to detect (excluding background).
        num_decoder_layers: int. Number of transformer decoder layers. Default: 6.
        num_heads: int. Number of attention heads in decoder. Default: 8.
        intermediate_size: int. FFN intermediate dimension in decoder. Default: 2048.
        dropout: float. Dropout rate. Default: 0.1.
        activation: str. Activation function. Default: "relu".
        bounding_box_format: str. Format for bounding boxes (e.g., "xyxy", "yxyx").
            Default: "yxyx".
        preprocessor: Optional. A preprocessor instance for image preprocessing.
        dtype: Optional. Data type for the model.
        **kwargs: Additional arguments passed to the parent class.

    Example:
    ```python
    # Create DETR model
    import keras_hub

    # Load ResNet-50 backbone
    resnet = keras_hub.models.ResNetBackbone.from_preset("resnet_50_imagenet")

    # Create DETR backbone
    backbone = keras_hub.models.DETRBackbone(
        image_encoder=resnet,
        hidden_dim=256,
        num_encoder_layers=6,
    )

    # Create object detector
    model = keras_hub.models.DETRObjectDetector(
        backbone=backbone,
        num_queries=100,
        num_classes=91,  # COCO classes
    )

    # Compile and train
    model.compile(optimizer="adam")
    model.fit(dataset, epochs=10)
    ```

    References:
    - [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
    - [DETR GitHub](https://github.com/facebookresearch/detr)
    """

    backbone_cls = DETRBackbone

    def __init__(
        self,
        backbone,
        num_queries=100,
        num_classes=91,
        num_decoder_layers=6,
        num_heads=8,
        intermediate_size=2048,
        dropout=0.1,
        activation="relu",
        bounding_box_format="yxyx",
        preprocessor=None,
        dtype=None,
        **kwargs,
    ):
        hidden_dim = backbone.hidden_dim
        head_dtype = dtype or backbone.dtype_policy

        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

        self.query_embed = DetrQueryEmbedding(
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            name="query_embed",
        )

        self.decoder = DetrTransformerDecoder(
            num_layers=num_decoder_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            activation=activation,
            dropout_rate=dropout,
            attentiondropout_rate=0.0,
            use_bias=True,
            norm_first=False,
            norm_epsilon=1e-5,
            intermediate_dropout=0.0,
            name="decoder",
        )

        self.class_head = layers.Dense(
            num_classes + 1,  # +1 for no-object class
            name="class_embed",
            dtype=head_dtype,
        )

        self.bbox_head = DETRMLPPredictionHead(
            hidden_dim=hidden_dim,
            output_dim=4,
            num_layers=3,
            name="bbox_embed",
        )

        # === Functional Model ===
        inputs = self.backbone.input

        # Get encoded features from backbone
        backbone_outputs = self.backbone(inputs)
        encoded_features = backbone_outputs["encoded_features"]
        pos_embed = backbone_outputs["pos_embed"]
        mask = backbone_outputs["mask"]

        # Get learnable object queries (tiled for batch)
        query_embeds = self.query_embed(inputs)

        # Create attention masks
        self_attn_mask_layer = CreateSelfAttentionMask(
            num_queries=num_queries, name="self_attention_mask"
        )
        cross_attn_mask_layer = CreateCrossAttentionMask(
            num_queries=num_queries, name="cross_attention_mask"
        )

        self_attn_mask = self_attn_mask_layer(query_embeds)
        cross_attn_mask = cross_attn_mask_layer(mask)

        decoder_target = ops.zeros_like(query_embeds)

        # Decode
        decoder_outputs = self.decoder(
            target=decoder_target,
            memory=encoded_features,
            self_attention_mask=self_attn_mask,
            cross_attention_mask=cross_attn_mask,
            input_pos_embed=query_embeds,
            memory_pos_embed=pos_embed,
        )

        # Predictions
        class_logits = self.class_head(decoder_outputs)
        bbox_preds = self.bbox_head(decoder_outputs)

        # Apply sigmoid to bbox predictions to get normalized coordinates [0, 1]
        bbox_preds = layers.Activation("sigmoid", name="bbox_sigmoid")(
            bbox_preds
        )

        # Outputs
        outputs = {
            "cls_logits": class_logits,  # (batch, num_queries, num_classes+1)
            "bbox_regression": bbox_preds,  # (batch, num_queries, 4)
        }

        Task.__init__(
            self,
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

        # === Config ===
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.activation = activation
        self.bounding_box_format = bounding_box_format

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_queries": self.num_queries,
                "num_classes": self.num_classes,
                "num_decoder_layers": self.num_decoder_layers,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "dropout": self.dropout,
                "activation": self.activation,
                "bounding_box_format": self.bounding_box_format,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)
