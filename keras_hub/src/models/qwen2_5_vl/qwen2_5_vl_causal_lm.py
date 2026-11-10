from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_backbone import (
    Qwen2_5VLBackbone,
)
from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_causal_lm_preprocessor import (
    Qwen2_5VLCausalLMPreprocessor,
)
from keras_hub.src.models.qwen2_vl.qwen2_vl_causal_lm import Qwen2VLCausalLM


@keras_hub_export("keras_hub.models.Qwen2_5VLCausalLM")
class Qwen2_5VLCausalLM(Qwen2VLCausalLM):
    """An end-to-end Qwen2.5-VL model for causal language modeling.

    Functionally identical to ``Qwen2VLCausalLM`` — generation, KV-cache
    decoding and multimodal interleaving are unchanged. The Qwen2.5-VL
    delta lives in the vision encoder and preprocessor; this subclass
    exists so presets resolve to the correct backbone and preprocessor
    classes.

    This model supports multimodal (image/video + text) inputs when the
    backbone has a ``vision_encoder`` attached and the preprocessor has an
    ``image_converter``.

    Args:
        backbone: A ``keras_hub.models.Qwen2_5VLBackbone`` instance.
        preprocessor: A ``keras_hub.models.Qwen2_5VLCausalLMPreprocessor``
            instance or ``None``. If ``None``, this model will not apply
            preprocessing, and inputs should be preprocessed before
            calling the model.
    """

    backbone_cls = Qwen2_5VLBackbone
    preprocessor_cls = Qwen2_5VLCausalLMPreprocessor
