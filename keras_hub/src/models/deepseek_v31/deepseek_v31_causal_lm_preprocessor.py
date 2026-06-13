"""DeepSeek V3.1 Causal LM Preprocessor."""

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.deepseek_v31.deepseek_v31_backbone import (
    DeepSeekV31Backbone,
)
from keras_hub.src.models.deepseek_v31.deepseek_v31_tokenizer import (
    DeepSeekV31Tokenizer,
)


@keras_hub_export("keras_hub.models.DeepSeekV31CausalLMPreprocessor")
class DeepSeekV31CausalLMPreprocessor(CausalLMPreprocessor):
    """DeepSeek V3.1 Causal LM preprocessor.

    This preprocessing layer is meant for use with
    `keras_hub.models.DeepSeekV31CausalLM`. By default, it will take in
    batches of strings, and return outputs in a `(x, y, sample_weight)`
    format, where the `y` label is the next token id in the `x` sequence.

    For use with generation, the layer also exposes two methods
    `generate_preprocess()` and `generate_postprocess()`. When this
    preprocessor is attached to a `keras_hub.models.DeepSeekV31CausalLM`
    instance, these methods will be called implicitly in `generate()`.

    Args:
        tokenizer: A `keras_hub.models.DeepSeekV31Tokenizer` instance.
        sequence_length: int. The length of the packed inputs.
        add_start_token: bool. Whether to prepend the start token.
        add_end_token: bool. Whether to append the end token.

    Example:
    ```python
    preprocessor = keras_hub.models.DeepSeekV31CausalLMPreprocessor.from_preset(
        "deepseek_v31_base"
    )

    # Preprocess a batch of strings
    sentences = ["Hello, world!", "How are you?"]
    x, y, sample_weight = preprocessor(sentences)
    ```
    """

    backbone_cls = DeepSeekV31Backbone
    tokenizer_cls = DeepSeekV31Tokenizer

    def __init__(
        self,
        tokenizer,
        sequence_length=1024,
        add_start_token=True,
        add_end_token=True,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            add_start_token=add_start_token,
            add_end_token=add_end_token,
            **kwargs,
        )
