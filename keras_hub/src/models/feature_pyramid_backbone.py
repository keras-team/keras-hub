import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone


@keras_hub_export("keras_hub.models.FeaturePyramidBackbone")
class FeaturePyramidBackbone(Backbone):
    """A backbone with feature pyramid outputs.

    `FeaturePyramidBackbone` extends `Backbone` with a single `pyramid_outputs`
    property for accessing the feature pyramid outputs of the model. Subclassers
    should set the `pyramid_outputs` property during the model constructor.

    Example:

    ```python
    input_data = np.random.uniform(0, 256, size=(2, 224, 224, 3))

    # Convert to feature pyramid output format using ResNet.
    backbone = ResNetBackbone.from_preset("resnet50")
    model = keras.Model(
        inputs=backbone.inputs, outputs=backbone.pyramid_outputs
    )
    model(input_data)  # A dict containing the keys ["P2", "P3", "P4", "P5"]
    ```
    """

    @property
    def pyramid_outputs(self):
        """A dict for feature pyramid outputs.

        The key is a string represents the name of the feature output and the
        value is a `keras.KerasTensor`. A typical feature pyramid has multiple
        levels corresponding to scales such as `["P2", "P3", "P4", "P5"]`. Scale
        `Pn` represents a feature map `2^n` times smaller in width and height
        than the inputs.
        """
        return getattr(self, "_pyramid_outputs", {})

    @pyramid_outputs.setter
    def pyramid_outputs(self, value):
        if not isinstance(value, dict):
            raise TypeError(
                "`pyramid_outputs` must be a dictionary. "
                f"Received: value={value} of type {type(value)}"
            )
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(
                    "The key of `pyramid_outputs` must be a string. "
                    f"Received: key={k} of type {type(k)}"
                )
            if not isinstance(v, keras.KerasTensor):
                raise TypeError(
                    "The value of `pyramid_outputs` must be a "
                    "`keras.KerasTensor`. "
                    f"Received: value={v} of type {type(v)}"
                )
        self._pyramid_outputs = value
