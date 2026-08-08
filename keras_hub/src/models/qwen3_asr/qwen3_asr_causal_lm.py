import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.qwen3_asr.qwen3_asr_backbone import Qwen3ASRBackbone
from keras_hub.src.models.qwen3_asr.qwen3_asr_causal_lm_preprocessor import (
    Qwen3ASRCausalLMPreprocessor,
)


@keras_hub_export("keras_hub.models.Qwen3ASRCausalLM")
class Qwen3ASRCausalLM(CausalLM):
    """An end-to-end Qwen3-ASR model for causal language modeling.

    Args:
        backbone: A `keras_hub.models.Qwen3ASRBackbone` instance.
        preprocessor: A `keras_hub.models.Qwen3ASRCausalLMPreprocessor` or
            `None`.
    """

    backbone_cls = Qwen3ASRBackbone
    preprocessor_cls = Qwen3ASRCausalLMPreprocessor

    def __init__(self, backbone, preprocessor=None, **kwargs):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

        # === Functional Model ===
        inputs = backbone.input
        hidden_states = backbone(inputs)
        outputs = backbone.token_embedding(hidden_states, reverse=True)
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )
