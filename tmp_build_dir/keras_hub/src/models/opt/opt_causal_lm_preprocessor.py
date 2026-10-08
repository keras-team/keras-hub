from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.opt.opt_backbone import OPTBackbone
from keras_hub.src.models.opt.opt_tokenizer import OPTTokenizer


@keras_hub_export("keras_hub.models.OPTCausalLMPreprocessor")
class OPTCausalLMPreprocessor(CausalLMPreprocessor):
    """OPT Causal LM preprocessor.

    This preprocessing layer is primarily meant to be used with
    `keras_hub.models.OPTCausalLM`. By default, it will take in batches of
    strings, and return outputs in a `(x, y, sample_weight)` format, where the
    `y` label is the next token id in the `x` sequence. For use with generation,
    pass `return_labels=False` in which case the output will simply be the
    encoded string features.

    Args:
        tokenizer: A `keras_hub.models.OPTTokenizer` instance.
        sequence_length: The length of the packed inputs.
        add_start_token: If `True`, the preprocessor will prepend the tokenizer
            start token to each input sequence.
        add_end_token: If `True`, the preprocessor will append the tokenizer
            end token to each input sequence.

    Call arguments:
        x: A string, `tf.Tensor` or list of python strings.
        y: Label data. Should always be `None` as the layer generates labels.
        sample_weight: Label weights. Should always be `None` as the layer
            generates label weights.
        sequence_length: Pass to override the configured `sequence_length` of
            the layer.
        add_start_token: Pass to override the configured value of
            `add_start_token` on the layer.
        add_end_token: Pass to override the configured value of
            `add_end_token` on the layer.
        return_labels: If `True`, the output `"token_ids"` will be offset by one
            and returned as labels. If `False` only features will be returned.

    Examples:
    ```python
    # Load the preprocessor from a preset.
    preprocessor = keras_hub.models.OPTCausalLMPreprocessor.from_preset(
        "opt_125m_en"
    )

    # Tokenize and pack a single sentence.
    sentence = tf.constant("League of legends")
    preprocessor(sentence)
    # Same output.
    preprocessor("League of legends")

    # Tokenize a batch of sentences.
    sentences = tf.constant(["Taco tuesday", "Fish taco please!"])
    preprocessor(sentences)
    # Same output.
    preprocessor(["Taco tuesday", "Fish taco please!"])

    # Map a dataset to preprocess a single sentence.
    features = tf.constant(
        [
            "Avatar 2 is amazing!",
            "Well, I am not sure.",
        ]
    )
    labels = tf.constant([1, 0])
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Map a dataset to preprocess unlabled sentences.
    ds = tf.data.Dataset.from_tensor_slices(features)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    ```
    """

    backbone_cls = OPTBackbone
    tokenizer_cls = OPTTokenizer
