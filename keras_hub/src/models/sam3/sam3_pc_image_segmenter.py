import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_segmenter import ImageSegmenter
from keras_hub.src.models.sam3.sam3_pc_backbone import (
    SAM3PromptableConceptBackbone,
)
from keras_hub.src.models.sam3.sam3_pc_image_segmenter_preprocessor import (
    SAM3PromptableConceptImageSegmenterPreprocessor,
)


@keras_hub_export("keras_hub.models.SAM3PromptableConceptImageSegmenter")
class SAM3PromptableConceptImageSegmenter(ImageSegmenter):
    """The Segment Anything 3 (SAM3) promptable concept image segmenter Model.

    SAM3 promptable concept segmentation (PCS) segments objects in images based
    on concept prompts, which could be short noun phrases
    (e.g., “yellow school bus”), image exemplars, or a combination of both.
    SAM3 PCS takes such prompts and returns segmentation masks and unique
    identities for all matching object instances.

    There are two ways to prompt:
    1. Text prompt: A short noun phrase describing the concept to segment.
    2. Box prompt: A box tells the model which part/crop of the image to
        segment.

    These prompts can be used individually or together, but at least one of the
    prompts must be present. To turn off a particular prompt, simply exclude it
    from the inputs to the model.

    Args:
        backbone: A `keras_hub.models.SAM3PromptableConceptBackbone` instance.
        preprocessor: Optional. An instance of
            `SAM3PromptableConceptImageSegmenterPreprocessor` for input data
            preprocessing.

    Example:

    Load pretrained model using `from_preset`.

    ```python
    image_size = 128
    batch_size = 2
    input_data = {
        "images": np.ones(
            (batch_size, image_size, image_size, 3), dtype="float32",
        ),
        "prompts": ["ear", "head"],
        "boxes": np.ones((batch_size, 1, 4), dtype="float32"),  # XYXY format.
        "box_labels": np.ones((batch_size, 1), dtype="float32"),
    }
    sam3_pcs = keras_hub.models.SAM3PromptableConceptImageSegmenter.from_preset(
        "sam3_pcs"
    )
    outputs = sam3_pcs.predict(input_data)
    scores = outputs["scores"]  # [B, num_queries]
    boxes = outputs["boxes"]  # [B, num_queries, 4]
    masks = outputs["masks"]  # [B, num_queries, H, W]
    ```

    Load pretrained model with custom image shape.

    ```python
    input_image_size = 128
    batch_size = 1
    model_image_size = 336
    input_data = {
        "images": np.ones(
            (batch_size, input_image_size, input_image_size, 3),
            dtype="float32",
        ),
        "prompts": ["ear", "head"],
        "boxes": np.ones((batch_size, 1, 4), dtype="float32"),  # XYXY format.
        "box_labels": np.ones((batch_size, 1), dtype="float32"),
    }
    sam3_backbone = keras_hub.models.SAM3PromptableConceptBackbone.from_preset(
        "sam3_pcs", image_shape=(model_image_size, model_image_size, 3)
    )
    sam3_preprocessor = keras_hub.models.SAM3PromptableConceptImageSegmenterPreprocessor.from_preset(
        "sam3_pcs"
    )
    sam3_preprocessor.image_size = (model_image_size, model_image_size)
    sam3_pcs = keras_hub.models.SAM3PromptableConceptImageSegmenter(
        backbone=sam3_backbone, preprocessor=sam3_preprocessor
    )
    outputs = sam3_pcs.predict(input_data)
    scores = outputs["scores"]  # [B, num_queries]
    boxes = outputs["boxes"]  # [B, num_queries, 4]
    masks = outputs["masks"]  # [B, num_queries, H, W]
    ```

    Load SAM3PromptableConceptImageSegmenter with custom backbone

    ```python
    vision_encoder = keras_hub.layers.SAM3VisionEncoder(
        image_shape=(224, 224, 3),
        patch_size=14,
        num_layers=2,
        hidden_dim=32,
        intermediate_dim=128,
        num_heads=2,
        fpn_hidden_dim=32,
        fpn_scale_factors=[4.0, 2.0, 1.0, 0.5],
        pretrain_image_shape=(112, 112, 3),
        window_size=2,
        global_attn_indexes=[1, 2],
    )
    text_encoder = keras_hub.layers.SAM3TextEncoder(
        vocabulary_size=1024,
        embedding_dim=32,
        hidden_dim=32,
        num_layers=2,
        num_heads=2,
        intermediate_dim=128,
    )
    geometry_encoder = keras_hub.layers.SAM3GeometryEncoder(
        num_layers=3,
        hidden_dim=32,
        intermediate_dim=128,
        num_heads=2,
        roi_size=7,
    )
    detr_encoder = keras_hub.layers.SAM3DetrEncoder(
        num_layers=3,
        hidden_dim=32,
        intermediate_dim=128,
        num_heads=2,
    )
    detr_decoder = keras_hub.layers.SAM3DetrDecoder(
        image_shape=(224, 224, 3),
        patch_size=14,
        num_layers=2,
        hidden_dim=32,
        intermediate_dim=128,
        num_heads=2,
        num_queries=100,
    )
    mask_decoder = keras_hub.layers.SAM3MaskDecoder(
        num_upsampling_stages=3,
        hidden_dim=32,
        num_heads=2,
    )
    backbone = keras_hub.models.SAM3PromptableConceptBackbone(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        geometry_encoder=geometry_encoder,
        detr_encoder=detr_encoder,
        detr_decoder=detr_decoder,
        mask_decoder=mask_decoder,
    )
    preprocessor = keras_hub.models.SAM3PromptableConceptImageSegmenterPreprocessor.from_preset(
        "sam3_pcs"
    )
    sam3_pcs = keras_hub.models.SAM3PromptableConceptImageSegmenter(
        backbone=backbone, preprocessor=preprocessor
    )
    ```

    For example, to pass in all the prompts, do:

    ```python
    image_size = 128
    batch_size = 2
    images = np.ones(
        (batch_size, image_size, image_size, 3), dtype="float32",
    )
    prompts = ["ear", "head"]
    # Box prompt in XYXY format
    boxes = np.array(
        [[[100.0, 100.0, 150.0, 150.0]], [[50.0, 50.0, 80.0, 80.0]]],
        dtype="float32",
    )
    # Box labels: 1 means positive box, 0 means negative box, -10 is for
    # padding boxes.
    box_labels = np.array([[1], [1]], dtype="int32")
    # Prepare an input dictionary:
    inputs = {
        "images": images,
        "prompts": prompts,
        "boxes": boxes,
        "box_labels": box_labels,
    }
    outputs = sam3_pcs.predict(inputs)
    scores = outputs["scores"]  # [B, num_queries]
    boxes = outputs["boxes"]  # [B, num_queries, 4]
    masks = outputs["masks"]  # [B, num_queries, H, W]
    ```

    Now, in case of only text prompts, simply exclude the box prompts:

    ```python
    inputs = {
        "images": images,
        "prompts": prompts,
    }
    outputs = sam3_pcs.predict(inputs)
    scores = outputs["scores"]  # [B, num_queries]
    boxes = outputs["boxes"]  # [B, num_queries, 4]
    masks = outputs["masks"]  # [B, num_queries, H, W]
    ```
    """  # noqa: E501

    backbone_cls = SAM3PromptableConceptBackbone
    preprocessor_cls = SAM3PromptableConceptImageSegmenterPreprocessor

    def __init__(
        self,
        backbone,
        preprocessor=None,
        **kwargs,
    ):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

        # === Functional Model ===
        inputs = self.backbone.input
        outputs = self.backbone(inputs)
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

    def fit(self, *args, **kwargs):
        raise NotImplementedError(
            "SAM3PromptableConceptImageSegmenter only supports inference for "
            "now. Training the model isn't supported yet."
        )

    def post_process_prediction(self, predictions):
        """Post-processes the raw model predictions.

        This method converts the raw model preditions into the scores, boxes and
        masks.

        The output format is as follows:
        - scores: A float tensor of shape `[batch_size, num_queries]`
            representing the confidence score of each object instance. The score
            is in the range [0, 1].
        - boxes: A float tensor of shape `[batch_size, num_queries, 4]`
            representing the bounding boxes of each object instance in
            `[x_min, y_min, x_max, y_max]` format. The box coordinates are
            normalized to the range [0, 1].
        - masks: A boolean tensor of shape
            `[batch_size, num_queries, height, width]` representing the binary
            masks of each object instance.
        """
        pred_logits = predictions["pred_logits"]
        pred_boxes = predictions["pred_boxes"]
        pred_masks = predictions["pred_masks"]
        presence_logits = predictions["presence_logits"]

        pred_scores = keras.ops.sigmoid(pred_logits)
        presence_scores = keras.ops.sigmoid(presence_logits)
        scores = keras.ops.multiply(pred_scores, presence_scores)

        masks = keras.ops.sigmoid(pred_masks)
        masks = keras.ops.transpose(masks, [0, 3, 1, 2])
        return {
            "scores": scores,
            "boxes": pred_boxes,
            "masks": masks,
        }

    def predict_step(self, *args):
        predictions = super().predict_step(*args)
        if isinstance(predictions, tuple):
            return self.post_process_prediction(predictions[0]), predictions[1]
        return self.post_process_prediction(predictions)

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        if "backbone" in config and isinstance(config["backbone"], dict):
            config["backbone"] = keras.saving.deserialize_keras_object(
                config["backbone"]
            )
        if "preprocessor" in config and isinstance(
            config["preprocessor"], dict
        ):
            config["preprocessor"] = keras.saving.deserialize_keras_object(
                config["preprocessor"]
            )
        return cls(**config)
