import keras
from keras import layers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone


@keras_hub_export("keras_hub.models.VGGBackbone")
class VGGBackbone(Backbone):
    """This class represents Keras Backbone of VGG model.

    This class implements a VGG backbone as described in [Very Deep
    Convolutional Networks for Large-Scale Image Recognition](
    https://arxiv.org/abs/1409.1556)(ICLR 2015).

    Args:
      stackwise_num_repeats: list of ints, number of repeated convolutional
            blocks per VGG block. For VGG16 this is [2, 2, 3, 3, 3] and for
            VGG19 this is [2, 2, 4, 4, 4].
      stackwise_num_filters: list of ints, filter size for convolutional
            blocks per VGG block. For both VGG16 and VGG19 this is [
            64, 128, 256, 512, 512].
      image_shape: tuple, optional shape tuple, defaults to (None, None, 3).

    Examples:
    ```python
    input_data = np.ones((2, 224, 224, 3), dtype="float32")

    # Pretrained VGG backbone.
    model = keras_hub.models.VGGBackbone.from_preset("vgg_16_imagenet")
    model(input_data)

    # Randomly initialized VGG backbone with a custom config.
    model = keras_hub.models.VGGBackbone(
        stackwise_num_repeats = [2, 2, 3, 3, 3],
        stackwise_num_filters = [64, 128, 256, 512, 512],
        image_shape = (224, 224, 3),
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        stackwise_num_repeats,
        stackwise_num_filters,
        image_shape=(None, None, 3),
        **kwargs,
    ):
        # === Functional Model ===
        img_input = keras.layers.Input(shape=image_shape)
        x = img_input

        for stack_index in range(len(stackwise_num_repeats)):
            x = apply_vgg_block(
                x=x,
                num_layers=stackwise_num_repeats[stack_index],
                filters=stackwise_num_filters[stack_index],
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
                max_pool=True,
                name=f"block{stack_index + 1}",
            )

        super().__init__(inputs=img_input, outputs=x, **kwargs)

        # === Config ===
        self.stackwise_num_repeats = stackwise_num_repeats
        self.stackwise_num_filters = stackwise_num_filters
        self.image_shape = image_shape

    def get_config(self):
        return {
            "stackwise_num_repeats": self.stackwise_num_repeats,
            "stackwise_num_filters": self.stackwise_num_filters,
            "image_shape": self.image_shape,
        }


def apply_vgg_block(
    x,
    num_layers,
    filters,
    kernel_size,
    activation,
    padding,
    max_pool,
    name,
):
    """
    Applies VGG block
    Args:
        x: Tensor, input tensor to pass through network
        num_layers: int, number of CNN layers in the block
        filters: int, filter size of each CNN layer in block
        kernel_size: int (or) tuple, kernel size for CNN layer in block
        activation: str (or) callable, activation function for each CNN layer in
            block
        padding: str (or) callable, padding function for each CNN layer in block
        max_pool: bool, whether to add MaxPooling2D layer at end of block
        name: str, name of the block

    Returns:
        keras.KerasTensor
    """
    for num in range(1, num_layers + 1):
        x = layers.Conv2D(
            filters,
            kernel_size,
            activation=activation,
            padding=padding,
            name=f"{name}_conv{num}",
        )(x)
    if max_pool:
        x = layers.MaxPooling2D((2, 2), (2, 2), name=f"{name}_pool")(x)
    return x
