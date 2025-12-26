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
    """Llama 3 Vision Causal LM model.

    This model combines the Llama 3 Vision Backbone with a causal language
    modeling head. It is capable of generating text based on input images
    and text prompts.

    A causal language model (LM) predicts the next token based on previous
    tokens. This task setup can be used to train the model unsupervised on
    plain text input, or to autoregressively generate plain text similar to
    the data used for training.

    This model has a `generate()` method, which generates text based on a
    prompt. The generation strategy used is controlled by an additional
    `sampler` argument on `compile()`. You can recompile the model with
    different `keras_hub.samplers` objects to control the generation. By
    default, `"top_k"` sampling will be used.

    Args:
        backbone: A `Llama3VisionBackbone` instance.
        preprocessor: A `Llama3VisionPreprocessor` instance or `None`.
            If `None`, this model will not apply preprocessing, and inputs
            should be preprocessed before calling the model.
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
