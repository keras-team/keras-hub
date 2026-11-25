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
    differs in terms of the vision encoder and preprocessor.

    This model supports multimodal (image/video + text) inputs when the
    backbone has a ``vision_encoder`` attached and the preprocessor has an
    ``image_converter``.

    Args:
        backbone: A ``keras_hub.models.Qwen2_5VLBackbone`` instance.
        preprocessor: A ``keras_hub.models.Qwen2_5VLCausalLMPreprocessor``
            instance or ``None``. If ``None``, this model will not apply
            preprocessing, and inputs should be preprocessed before
            calling the model. Defaults to ``None``.

    Examples:

    ```python
    # Load a preset with a greedy sampler and generate text.
    qwen2_5_vl_lm = keras_hub.models.Qwen2_5VLCausalLM.from_preset(
        "qwen2_5_vl_3b_instruct",
    )
    qwen2_5_vl_lm.generate("What is Keras?", max_length=32)

    # Multimodal prompt: describe an image.
    qwen2_5_vl_lm.generate(
        {
            "prompts": ["<|vision_start|><|image_pad|><|vision_end|>"
                        "Describe this image.<|im_end|>"],
            "images": [image],  # (H, W, 3) array
        },
        max_length=128,
    )
    ```
    """

    backbone_cls = Qwen2_5VLBackbone
    preprocessor_cls = Qwen2_5VLCausalLMPreprocessor
