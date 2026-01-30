import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.moondream.moondream_backbone import MoondreamBackbone
from keras_hub.src.models.moondream.moondream_preprocessor import \
    MoondreamPreprocessor


@keras_hub_export("keras_hub.models.MoondreamCausalLM")
class MoondreamCausalLM(CausalLM):
    backbone_cls = MoondreamBackbone
    preprocessor_cls = MoondreamPreprocessor

    def __init__(
        self,
        backbone,
        preprocessor=None,
        **kwargs,
    ):
        inputs = getattr(backbone, "input", None)

        super().__init__(**kwargs)

        # Manually set the attributes
        self.backbone = backbone
        self.preprocessor = preprocessor

        # Set tensor spec if available
        if inputs is not None:
            self.input_tensor_spec = inputs

    def call(self, inputs, training=False):
        if self.backbone is None:
            raise ValueError("Backbone not initialized")
        x = self.backbone(inputs)
        return x
