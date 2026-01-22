from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_segmenter import ImageSegmenter
from keras_hub.src.models.sam3.sam3_backbone import SAM3Backbone
from keras_hub.src.models.sam3.sam3_image_segmenter_preprocessor import (
    SAM3ImageSegmenterPreprocessor,
)


@keras_hub_export("keras_hub.models.SAM3ImageSegmenter")
class SAM3ImageSegmenter(ImageSegmenter):
    """The Segment Anything 3 (SAM3) image segmenter Model.


    Args:
      backbone: A `keras_hub.models.SAM3Backbone` instance.

    Example:
    Load pretrained model using `from_preset`.

    ```python
    image_size=128
    batch_size=2
    input_data = {
        "images": np.ones(
            (batch_size, image_size, image_size, 3),
            dtype="float32",
        ),
        "points": np.ones((batch_size, 1, 2), dtype="float32"),
        "labels": np.ones((batch_size, 1), dtype="float32"),
        "boxes": np.ones((batch_size, 1, 2, 2), dtype="float32"),
        "masks": np.zeros(
            (batch_size, 0, image_size, image_size, 1)
        ),
    }
    sam = keras_hub.models.SAMImageSegmenter.from_preset('sam_base_sa1b')
    outputs = sam.predict(input_data)
    masks, iou_pred = outputs["masks"], outputs["iou_pred"]
    ```

    Load segment anything image segmenter with custom backbone

    ```python
    image_size = 128
    batch_size = 2
    images = np.ones(
        (batch_size, image_size, image_size, 3),
        dtype="float32",
    )
    image_encoder = keras_hub.models.ViTDetBackbone(
        hidden_size=16,
        num_layers=16,
        intermediate_dim=16 * 4,
        num_heads=16,
        global_attention_layer_indices=[2, 5, 8, 11],
        patch_size=16,
        num_output_channels=8,
        window_size=2,
        image_shape=(image_size, image_size, 3),
    )
    prompt_encoder = keras_hub.layers.SAMPromptEncoder(
        hidden_size=8,
        image_embedding_size=(8, 8),
        input_image_size=(
            image_size,
            image_size,
        ),
        mask_in_channels=16,
    )
    mask_decoder = keras_hub.layers.SAMMaskDecoder(
        num_layers=2,
        hidden_size=8,
        intermediate_dim=32,
        num_heads=8,
        embedding_dim=8,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=8,
    )
    backbone = keras_hub.models.SAMBackbone(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
    )
    sam = keras_hub.models.SAMImageSegmenter(
        backbone=backbone
    )
    ```

    For example, to pass in all the prompts, do:

    ```python

    points = np.array([[[512., 512.], [100., 100.]]])
    # For labels: 1 means foreground point, 0 means background
    labels = np.array([[1., 0.]])
    box = np.array([[[[384., 384.], [640., 640.]]]])
    input_mask = np.ones((1, 1, 256, 256, 1))
    # Prepare an input dictionary:
    inputs = {
        "images": image,
        "points": points,
        "labels": labels,
        "boxes": box,
        "masks": input_mask
    }
    outputs = sam.predict(inputs)
    masks, iou_pred = outputs["masks"], outputs["iou_pred"]
    ```

    The first mask in the output `masks` (i.e. `masks[:, 0, ...]`) is the best
    mask predicted by the model based on the prompts. Other `masks`
    (i.e. `masks[:, 1:, ...]`) are alternate predictions that can be used if
    they are desired over the first one.
    Now, in case of only points and box prompts, simply exclude the masks:

    ```python
    inputs = {
        "images": image,
        "points": points,
        "labels": labels,
        "boxes": box,
    }

    outputs = sam.predict(inputs)
    masks, iou_pred = outputs["masks"], outputs["iou_pred"]
    ```

    Another example is that only points prompts are present.
    Note that if point prompts are present but no box prompt is present, the
    points must be padded using a zero point and -1 label:

    ```python
    padded_points = np.concatenate(
        [points, np.zeros((1, 1, 2))], axis=1
    )

    padded_labels = np.concatenate(
        [labels, -np.ones((1, 1))], axis=1
    )
    inputs = {
        "images": image,
        "points": padded_points,
        "labels": padded_labels,
    }
    outputs = sam.predict(inputs)
    masks, iou_pred = outputs["masks"], outputs["iou_pred"]
    ```
    """

    backbone_cls = SAM3Backbone
    preprocessor_cls = SAM3ImageSegmenterPreprocessor

    def __init__(self, backbone, preprocessor=None, **kwargs):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

        # === Functional Model ===
        inputs = self.backbone.input
        outputs = self.backbone(inputs)
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

    def fit(self, *args, **kwargs):
        raise NotImplementedError(
            "SAM3ImageSegmenter only supports inference for now. Training"
            " the model isn't supported yet."
        )
