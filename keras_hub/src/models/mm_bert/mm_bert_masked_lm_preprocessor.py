from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.mm_bert.mm_bert_backbone import MMBertBackbone
from keras_hub.src.models.mm_bert.mm_bert_tokenizer import MMBertTokenizer
from keras_hub.src.models.modern_bert.modern_bert_masked_lm_preprocessor import (  # noqa: E501
    ModernBertMaskedLMPreprocessor,
)


@keras_hub_export("keras_hub.models.MMBertMaskedLMPreprocessor")
class MMBertMaskedLMPreprocessor(ModernBertMaskedLMPreprocessor):
    """mmBERT Masked LM preprocessor.

    This preprocessor tokenizes and prepares inputs for masked language
    modeling with mmBERT. The masking, packing, and serialization logic is
    inherited unchanged from `ModernBertMaskedLMPreprocessor`, which drops the
    `segment_ids` that mmBERT does not use.

    Args:
        tokenizer: A `keras_hub.models.MMBertTokenizer` instance.
        sequence_length: int. The length of the packed sequence.
        mask_selection_rate: float. The probability of masking a token.
        mask_selection_length: int. The maximum number of tokens to mask
            per sequence.
        mask_token_rate: float. The fraction of selected tokens replaced
            with the mask token.
        random_token_rate: float. The fraction of selected tokens replaced
            with a randomly selected token.
        **kwargs: Additional keyword arguments passed to
            `MaskedLMPreprocessor`.

    Examples:
    ```python
    import keras_hub

    preprocessor = keras_hub.models.MMBertMaskedLMPreprocessor.from_preset(
        "./mm_bert_base_multi"
    )

    # Tokenize and mask a single sentence.
    preprocessor("The quick brown fox jumped.")

    # Tokenize and mask a batch of sentences.
    preprocessor(["The quick brown fox jumped.", "Call me Ishmael."])

    # Use in a `tf.data.Dataset`.
    import tensorflow as tf

    features = ["The quick brown fox jumped.", "Call me Ishmael."]
    ds = tf.data.Dataset.from_tensor_slices(features)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    ```
    """

    backbone_cls = MMBertBackbone
    tokenizer_cls = MMBertTokenizer
