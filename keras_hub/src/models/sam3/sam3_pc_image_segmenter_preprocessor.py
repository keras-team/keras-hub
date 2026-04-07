import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.models.sam3.sam3_image_converter import SAM3ImageConverter
from keras_hub.src.models.sam3.sam3_pc_backbone import (
    SAM3PromptableConceptBackbone,
)
from keras_hub.src.models.sam3.sam3_tokenizer import SAM3Tokenizer
from keras_hub.src.utils.tensor_utils import preprocessing_function

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export(
    "keras_hub.models.SAM3PromptableConceptImageSegmenterPreprocessor"
)
class SAM3PromptableConceptImageSegmenterPreprocessor(Preprocessor):
    """SAM3 Promptable Concept Image Segmenter preprocessor.

    This preprocessing layer is meant for use with
    `keras_hub.models.SAM3PromptableConceptImageSegmenter`.

    Args:
        tokenizer: A `keras_hub.models.SAM3Tokenizer` instance.
        image_converter: A `keras_hub.layers.SAM3ImageConverter` instance.
        sequence_length: The length of the packed token_ids. Defaults to `32`.
        add_start_token: If `True`, the preprocessor will prepend the tokenizer
            start token to each input sequence. Defaults to `True`.
        add_end_token: If `True`, the preprocessor will append the tokenizer
            end token to each input sequence. Defaults to `True`.
        point_pad_value: int. The padding value for box prompts. Defaults to
            `-10`.

    Call arguments:
        x: A dictionary with the following keys:
            - images: A single image or a batch of images, of shape
              `(height, width, 3)` or `(batch_size, height, width, 3)`.
            - prompts: (optional) A string or a batch of strings containing the
              text prompts. If not provided, a default prompt will be used.
            - boxes: (optional) A tensor of shape `(num_boxes, 4)` or
              `(batch_size, num_boxes, 4)` containing box coordinates in
                `(x_min, y_min, x_max, y_max)` format. Coordinates should be in
                absolute pixel values. If not provided, no box prompts will be
                used. `-10` is used as the padding value.
            - box_labels: (optional) A tensor of shape `(num_boxes,)` or
              `(batch_size, num_boxes)` containing box labels. If not provided,
              no box labels will be used. `-10` is used as the padding value.

    Examples:

    ```python
    # Load the preprocessor from a preset.
    preprocessor = keras_hub.models.SAM3PromptableConceptImageSegmenterPreprocessor.from_preset(
        "sam3_pcs"
    )

    # Unbatched inputs, with one image and one text prompt.
    preprocessor(
        {
            "prompts": "ear",
            "images": np.ones((896, 896, 3), dtype="float32")
        }
    )

    # Unbatched inputs, with one image and one box prompt.
    preprocessor(
        {
            "boxes": [[0, 0, 300, 300]],
            "box_labels": [1],
            "images": np.ones((896, 896, 3), dtype="float32")
        }
    )

    # Batched inputs, one image per text prompt.
    preprocessor(
        {
            "prompts": [
                "ear",
                "head"
            ],
            "images": [
                np.ones((896, 896, 3), dtype="float32"),
                np.ones((896, 896, 3), dtype="float32")
            ]
        }
    )

    # Batched inputs, one image per box prompt.
    preprocessor(
        {
            "boxes": [
                [[0, 0, 300, 300]],
                [[50, 50, 100, 100]]
            ],
            "box_labels": [
                [1],
                [1]
            ],
            "images": [
                np.ones((896, 896, 3), dtype="float32"),
                np.ones((896, 896, 3), dtype="float32")
            ]
        }
    )

    # Different number of box prompts in every sample.
    preprocessor(
        {
            "boxes": [
                [[0, 0, 300, 300]],
                [[50, 50, 100, 100], [150, 150, 200, 200]]
            ],
            "box_labels": [
                [1],
                [1, 1]
            ],
            "images": [
                np.ones((896, 896, 3), dtype="float32"),
                np.ones((896, 896, 3), dtype="float32")
            ]
        }
    )

    # Apply preprocessing to a `tf.data.Dataset`.
    inputs = {
        "prompts": [
            "ear",
            "head",
        ],
        "images": np.ones((2, 896, 896, 3), dtype="float32")
    }
    ds = tf.data.Dataset.from_tensor_slices(inputs)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    ```
    """  # noqa: E501

    backbone_cls = SAM3PromptableConceptBackbone
    tokenizer_cls = SAM3Tokenizer
    image_converter_cls = SAM3ImageConverter

    def __init__(
        self,
        tokenizer,
        image_converter,
        sequence_length=32,
        add_start_token=True,
        add_end_token=True,
        point_pad_value=-10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.packer = None
        self.image_converter = image_converter
        self.sequence_length = sequence_length
        self.add_start_token = add_start_token
        self.add_end_token = add_end_token
        self.point_pad_value = point_pad_value

    def build(self, input_shape):
        # Defer packer creation to `build()` so that we can be sure tokenizer
        # assets have loaded when restoring a saved model.
        self.packer = StartEndPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sequence_length=self.sequence_length,
            return_padding_mask=True,
        )
        self.built = True

    def _preprocess_boxes(self, boxes, box_labels, height, width):
        if isinstance(boxes, tf.RaggedTensor):
            max_num_boxes = tf.reduce_max(boxes.row_lengths(axis=1))
            boxes = boxes.to_tensor(
                shape=[None, max_num_boxes, 4],
                default_value=self.point_pad_value,
            )
            box_labels = box_labels.to_tensor(
                shape=[None, max_num_boxes],
                default_value=self.point_pad_value,
            )
        box_dtype = keras.backend.standardize_dtype(boxes.dtype)
        normalized_boxes = tf.stack(
            [
                boxes[..., 0] / tf.cast(width, box_dtype),
                boxes[..., 1] / tf.cast(height, box_dtype),
                boxes[..., 2] / tf.cast(width, box_dtype),
                boxes[..., 3] / tf.cast(height, box_dtype),
            ],
            axis=-1,
        )
        boxes = tf.where(
            tf.equal(tf.expand_dims(box_labels, axis=-1), self.point_pad_value),
            tf.fill(
                tf.shape(normalized_boxes),
                tf.cast(self.point_pad_value, normalized_boxes.dtype),
            ),
            normalized_boxes,
        )
        # XYXY to CXCYWH.
        boxes = tf.stack(
            [
                (boxes[..., 0] + boxes[..., 2]) / 2.0,
                (boxes[..., 1] + boxes[..., 3]) / 2.0,
                boxes[..., 2] - boxes[..., 0],
                boxes[..., 3] - boxes[..., 1],
            ],
            axis=-1,
        )
        # Add batch indices.
        batch_size = tf.shape(boxes)[0]
        batch_indices = tf.range(batch_size, dtype=boxes.dtype)
        batch_indices = tf.reshape(batch_indices, (batch_size, 1, 1))
        batch_indices = tf.tile(batch_indices, (1, tf.shape(boxes)[1], 1))
        boxes = tf.concat([batch_indices, boxes], axis=-1)
        return boxes, box_labels

    @preprocessing_function
    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
    ):
        sequence_length = sequence_length or self.sequence_length

        images = x["images"]
        prompts = x.get("prompts", None)
        boxes, box_labels = x.get("boxes", None), x.get("box_labels", None)

        # Convert to batched inputs.
        if len(images.shape) == 3:
            is_batched = False
            images = tf.expand_dims(images, axis=0)
            if prompts is not None and len(prompts.shape) == 0:
                prompts = tf.expand_dims(prompts, axis=0)
            if boxes is not None and len(boxes.shape) == 2:
                boxes = tf.expand_dims(boxes, axis=0)
                box_labels = tf.expand_dims(box_labels, axis=0)
        else:
            is_batched = True

        batch_size = tf.shape(images)[0]
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]

        # Add placeholders if not provided.
        if prompts is None:
            prompts = tf.convert_to_tensor("visual")
            prompts = tf.tile(prompts[None], [batch_size])
        if boxes is None:
            boxes = tf.zeros((batch_size, 0, 4), dtype="float32")
            box_labels = tf.zeros((batch_size, 0), dtype="int32")

        # Tokenise the prompts.
        prompts = self.tokenizer(prompts)
        token_ids, padding_mask = self.packer(
            prompts,
            sequence_length=sequence_length + 1,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )

        # Resize and normalize the images.
        pixel_values = self.image_converter(images)
        if keras.config.backend() == "torch" and not isinstance(
            images, tf.Tensor
        ):
            images = images.cpu()

        # Normalize the boxes.
        boxes, box_labels = self._preprocess_boxes(
            boxes, box_labels, height, width
        )

        if not is_batched:
            token_ids = tf.squeeze(token_ids, axis=0)
            padding_mask = tf.squeeze(padding_mask, axis=0)
            pixel_values = tf.squeeze(pixel_values, axis=0)
            boxes = tf.squeeze(boxes, axis=0)
            box_labels = tf.squeeze(box_labels, axis=0)

        x = {
            "pixel_values": pixel_values,
            "token_ids": token_ids[..., :-1],
            "padding_mask": padding_mask[..., :-1],
            "boxes": boxes,
            "box_labels": box_labels,
        }
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "add_start_token": self.add_start_token,
                "add_end_token": self.add_end_token,
            }
        )
        return config

    @property
    def sequence_length(self):
        """The padded length of model input sequences."""
        return self._sequence_length

    @sequence_length.setter
    def sequence_length(self, value):
        self._sequence_length = value
        if self.packer is not None:
            self.packer.sequence_length = value

    @property
    def image_size(self):
        """Settable tuple of `(height, width)` ints. The output image shape."""
        if self.image_converter.resizing.height is None:
            return None
        return (
            self.image_converter.resizing.height,
            self.image_converter.resizing.width,
        )

    @image_size.setter
    def image_size(self, value):
        if value is None:
            value = (None, None)
        self.image_converter.resizing.height = value[0]
        self.image_converter.resizing.width = value[1]
