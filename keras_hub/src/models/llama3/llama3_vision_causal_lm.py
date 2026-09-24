from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.llama3.llama3_vision_backbone import (
    Llama3VisionBackbone,
)
from keras_hub.src.models.llama3.llama3_vision_preprocessor import (
    Llama3VisionPreprocessor,
)


@keras_hub_export("keras_hub.models.Llama3VisionCausalLM")
class Llama3VisionCausalLM(CausalLM):
    """End-to-end Llama 3.2 Vision model for causal language modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens. This model combines vision and language understanding to generate
    text based on image and text inputs.

    This model has a `generate()` method for text generation. The generation
    strategy is controlled by the `sampler` argument on `compile()`.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to inputs during
    `fit()`, `predict()`, `evaluate()` and `generate()`.

    Args:
        backbone: A `keras_hub.models.Llama3VisionBackbone` instance.
        preprocessor: A `keras_hub.models.Llama3VisionPreprocessor` or `None`.
            If `None`, inputs should be preprocessed before calling the model.

    Example:
    ```python
    # Load from preset.
    causal_lm = keras_hub.models.Llama3VisionCausalLM.from_preset(
        "llama3_2_vision_11b"
    )

    # Generate with image and text.
    output = causal_lm.generate({
        "images": image,
        "prompts": "Describe this image:",
    })
    ```
    """

    backbone_cls = Llama3VisionBackbone
    preprocessor_cls = Llama3VisionPreprocessor

    def __init__(self, backbone, preprocessor=None, **kwargs):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

        # === Functional Model ===
        inputs = backbone.input
        hidden_states = backbone(inputs)
        outputs = backbone.text_backbone.token_embedding(
            hidden_states, reverse=True
        )
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )
