from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.rwkv7.rwkv7_backbone import RWKV7Backbone
from keras_hub.src.models.rwkv7.rwkv7_causal_lm_preprocessor import (
    RWKV7CausalLMPreprocessor,
)


@keras_hub_export("keras_hub.models.RWKV7CausalLM")
class RWKV7CausalLM(CausalLM):
    backbone_cls = RWKV7Backbone
    preprocessor_cls = RWKV7CausalLMPreprocessor

    def __init__(self, backbone, preprocessor=None, **kwargs):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor
        super().__init__(
            inputs=backbone.inputs,
            outputs=backbone.outputs,
            **kwargs,
        )

    def call_with_cache(
        self,
        token_ids,
        cache,
        cache_update_index,
    ):
        pass  # TODO

    def _build_cache(self, token_ids):
        pass  # TODO

    def generate_step(
        self,
        inputs,
        stop_token_ids=None,
    ):
        pass  # TODO

    def score(
        self,
        token_ids,
        padding_mask=None,
        scoring_mode="logits",
        layer_intercept_fn=None,
        target_ids=None,
    ):
        pass  # TODO
