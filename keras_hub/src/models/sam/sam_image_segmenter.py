import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_segmenter import ImageSegmenter
from keras_hub.src.models.sam.sam_backbone import SAMBackbone
from keras_hub.src.models.sam.sam_image_segmenter_preprocessor import (
    SAMImageSegmenterPreprocessor,
)


@keras_hub_export("keras_hub.models.SAMImageSegmenter")
class SAMImageSegmenter(ImageSegmenter):
    """The Segment Anything (SAM) image segmenter Model.

    SAM works by prompting the input images. There are three ways to prompt:
    (1) Labelled Points: Foreground points (points with label 1) are encoded
        such that the output masks generated by the mask decoder contain them
        and background points (points with label 0) are encoded such that the
        generated masks don't contain them.
    (2) Box: A box tells the model which part/crop of the image to segment.
    (3) Mask: An input mask can be used to refine the output of the mask
        decoder.
    These prompts can be mixed and matched but at least one of the prompts
    must be present. To turn off a particular prompt, simply exclude it from
    the inputs to the model.
    (1) For points prompts, the expected shape is `(batch, num_points, 2)`.
        The labels must have a corresponding shape of `(batch, num_points)`.
    (2) For box prompt, the expected shape is `(batch, 1, 2, 2)`.
    (3) Similarly, mask prompts have shape `(batch, 1, H, W, 1)`.


    Args:
      backbone: A `keras_hub.models.SAMBackbone` instance.

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
    # todo: update preset name
    sam = keras_hub.models.SAMImageSegmenter.from_preset(`sam_base`)
    sam(input_data)
    ```

    Load segment anything image segmenter with custom backbone

    ```python
    image_size = 128
    batch_size = 2
    images = np.ones(
        (batch_size, image_size, image_size, 3),
        dtype="float32",
    )
    image_encoder = ViTDetBackbone(
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
    prompt_encoder = SAMPromptEncoder(
        hidden_size=8,
        image_embedding_size=(8, 8),
        input_image_size=(
            image_size,
            image_size,
        ),
        mask_in_channels=16,
    )
    mask_decoder = SAMMaskDecoder(
        num_layers=2,
        hidden_size=8,
        intermediate_dim=32,
        num_heads=8,
        embedding_dim=8,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=8,
    )
    backbone = SAMBackbone(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        image_shape=(image_size, image_size, 3),
    )
    sam = SAMImageSegmenter(
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
    Prepare an input dictionary:
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

    backbone_cls = SAMBackbone
    preprocessor_cls = SAMImageSegmenterPreprocessor

    def __init__(self, backbone, preprocessor=None, **kwargs):
        # The implementation has been adapted form [Segment Anything
        # paper](https://arxiv.org/abs/2304.02643) and [Segment Anything
        # GitHub](https://github.com/facebookresearch/segment-anything) and
        # [Detectron2](https://github.com/facebookresearch/detectron2).
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor
        # === Functional Model ===
        inputs = self.backbone.input
        x = self.backbone(inputs)
        outputs = self.backbone.mask_decoder(**x)
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

    def predict_step(self, *args, **kwargs):
        if len(args) == 2:
            args = (args[0], self._add_placeholder_prompts(args[-1]))
        else:
            args = (self._add_placeholder_prompts(args[0]),)

        return super().predict_step(*args, **kwargs)

    def fit(self, *args, **kwargs):
        raise NotImplementedError(
            "Segment Anything Model only supports inference for now. Training"
            " the model isn't supported yet."
        )

    def _add_placeholder_prompts(self, inputs):
        """Adds placeholder prompt inputs for a call to SAM.

        Because SAM is a functional subclass model, all inputs must be specified in
        calls to the model. However, prompt inputs are all optional, so we have to
        add placeholders when they're not specified by the user.
        """
        inputs = inputs.copy()

        # Get the batch shape based on the image input
        batch_size = ops.shape(inputs["images"])[0]

        # The type of the placeholders must match the existing inputs with respect
        # to whether or not they are tensors (as opposed to Numpy arrays).
        zeros = ops.zeros if ops.is_tensor(inputs["images"]) else np.zeros

        # Fill in missing inputs.
        if "points" not in inputs:
            inputs["points"] = zeros((batch_size, 0, 2))
        if "labels" not in inputs:
            inputs["labels"] = zeros((batch_size, 0))
        if "boxes" not in inputs:
            inputs["boxes"] = zeros((batch_size, 0, 2, 2))
        if "masks" not in inputs:
            inputs["masks"] = zeros((batch_size, 0, 256, 256, 1))

        return inputs
