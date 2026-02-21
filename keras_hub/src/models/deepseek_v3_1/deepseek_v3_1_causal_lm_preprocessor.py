"""DeepSeek V3.1 Causal LM Preprocessor."""

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.deepseek_v3_1.deepseek_v3_1_backbone import (
    DeepSeekV3_1Backbone,
)
from keras_hub.src.models.deepseek_v3_1.deepseek_v3_1_tokenizer import (
    DeepSeekV3_1Tokenizer,
)


@keras_hub_export("keras_hub.models.DeepSeekV3_1CausalLMPreprocessor")
class DeepSeekV3_1CausalLMPreprocessor(CausalLMPreprocessor):
    """DeepSeek V3.1 Causal LM preprocessor.

    This preprocessing layer is meant for use with
    `keras_hub.models.DeepSeekV3_1CausalLM`. By default, it will take in
    batches of strings, and return outputs in a `(x, y, sample_weight)`
    format, where the `y` label is the next token id in the `x` sequence.

    For use with generation, the layer also exposes two methods
    `generate_preprocess()` and `generate_postprocess()`. When this
    preprocessor is attached to a `keras_hub.models.DeepSeekV3_1CausalLM`
    instance, these methods will be called implicitly in `generate()`.

    Args:
        tokenizer: A `keras_hub.models.DeepSeekV3_1Tokenizer` instance.
        sequence_length: int. The length of the packed inputs.
        add_start_token: bool. Whether to prepend the start token.
        add_end_token: bool. Whether to append the end token.

    Example:
    ```python
    preprocessor = keras_hub.models.DeepSeekV3_1CausalLMPreprocessor.from_preset(
        "deepseek_v3_1_base"
    )

    # Preprocess a batch of strings
    sentences = ["Hello, world!", "How are you?"]
    x, y, sample_weight = preprocessor(sentences)
    ```
    """

    backbone_cls = DeepSeekV3_1Backbone
    tokenizer_cls = DeepSeekV3_1Tokenizer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
