import keras

import keras_hub.src.bounding_box.validate_format as validate_format
from keras_hub.src.api_export import keras_hub_export

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.bounding_box.to_ragged")
def to_ragged(bounding_boxes, sentinel=-1, dtype="float32"):
    """converts a Dense padded bounding box `tf.Tensor` to a `tf.RaggedTensor`.

    Bounding boxes are ragged tensors in most use cases. Converting them to a
    dense tensor makes it easier to work with Tensorflow ecosystem.
    This function can be used to filter out the masked out bounding boxes by
    checking for padded sentinel value of the class_id axis of the
    bounding_boxes.

    Example:
    ```python
    bounding_boxes = {
        "boxes": tf.constant([[2, 3, 4, 5], [0, 1, 2, 3]]),
        "classes": tf.constant([[-1, 1]]),
    }
    bounding_boxes = bounding_box.to_ragged(bounding_boxes)
    print(bounding_boxes)
    # {
    #     "boxes": [[0, 1, 2, 3]],
    #     "classes": [[1]]
    # }
    ```

    Args:
        bounding_boxes: a Tensor of bounding boxes. May be batched, or
            unbatched.
        sentinel: The value indicating that a bounding box does not exist at the
            current index, and the corresponding box is padding, defaults to -1.
        dtype: the data type to use for the underlying Tensors.
    Returns:
        dictionary of `tf.RaggedTensor` or 'tf.Tensor' containing the filtered
        bounding boxes.
    """
    if keras.config.backend() != "tensorflow":
        raise NotImplementedError(
            "`bounding_box.to_ragged` was called using a backend which does "
            "not support ragged tensors. "
            f"Current backend: {keras.backend.backend()}."
        )

    info = validate_format.validate_format(bounding_boxes)

    if info["ragged"]:
        return bounding_boxes

    boxes = bounding_boxes.get("boxes")
    classes = bounding_boxes.get("classes")
    confidence = bounding_boxes.get("confidence", None)

    mask = classes != sentinel

    boxes = tf.ragged.boolean_mask(boxes, mask)
    classes = tf.ragged.boolean_mask(classes, mask)
    if confidence is not None:
        confidence = tf.ragged.boolean_mask(confidence, mask)

    if isinstance(boxes, tf.Tensor):
        boxes = tf.RaggedTensor.from_tensor(boxes)

    if isinstance(classes, tf.Tensor) and len(classes.shape) > 1:
        classes = tf.RaggedTensor.from_tensor(classes)

    if confidence is not None:
        if isinstance(confidence, tf.Tensor) and len(confidence.shape) > 1:
            confidence = tf.RaggedTensor.from_tensor(confidence)

    result = bounding_boxes.copy()
    result["boxes"] = tf.cast(boxes, dtype)
    result["classes"] = tf.cast(classes, dtype)

    if confidence is not None:
        result["confidence"] = tf.cast(confidence, dtype)

    return result
