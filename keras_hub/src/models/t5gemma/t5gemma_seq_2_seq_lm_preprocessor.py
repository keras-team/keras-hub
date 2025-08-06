from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.seq_2_seq_lm_preprocessor import Seq2SeqLMPreprocessor
from keras_hub.src.models.t5gemma.t5gemma_backbone import T5GemmaBackbone
from keras_hub.src.models.t5gemma.t5gemma_tokenizer import T5GemmaTokenizer


@keras_hub_export("keras_hub.models.T5GemmaSeq2SeqLMPreprocessor")
class T5GemmaSeq2SeqLMPreprocessor(Seq2SeqLMPreprocessor):
    """T5Gemma Seq2Seq LM preprocessor.

    This preprocessing layer is meant for use with
    `keras_hub.models.T5GemmaSeq2SeqLM`. By default, it will take in batches of
    strings, and return outputs in a `(x, y, sample_weight)` format, where the
    `y` label is the next token id in the `x` sequence.

    For use with generation, the layer also exposes two methods
    `generate_preprocess()` and `generate_postprocess()`. When this preprocessor
    is attached to a `keras_hub.models.T5GemmaSeq2SeqLM` instance, these methods
    will be called implicitly in `generate()`. They can also be called
    standalone (e.g. to precompute preprocessing inputs for generation in a
    separate process).

    Args:
        tokenizer: A `keras_hub.models.T5GemmaTokenizer` instance.
        encoder_sequence_length: The length of the packed encoder inputs.
        decoder_sequence_length: The length of the packed decoder inputs.
        add_start_token: If `True`, the preprocessor will prepend the tokenizer
            start token to each input sequence. Defaults to `True`.
        add_end_token: If `True`, the preprocessor will append the tokenizer
            end token to each input sequence. Defaults to `False`.

    Call arguments:
        x: A string, `tf.Tensor` or list of python strings. Can also be a
            dictionary with `encoder_text` and `decoder_text` keys.
        y: Label data. Should always be `None` as the layer generates labels.
        sample_weight: Label weights. Should always be `None` as the layer
            generates label weights.
        encoder_sequence_length: Pass to override the configured
            `encoder_sequence_length` of the layer.
        decoder_sequence_length: Pass to override the configured
            `decoder_sequence_length` of the layer.

    Examples:
    ```python
    import tensorflow as tf
    import numpy as np

    # Load the preprocessor from a preset.
    preprocessor = keras_hub.models.T5GemmaSeq2SeqLMPreprocessor.from_preset(
        "t5gemma_b_b_prefixlm_it"
    )

    # Tokenize and pack a single sentence.
    sentence = tf.constant("The quick brown fox jumped.")
    preprocessor(sentence)

    # Tokenize a batch of sentences.
    preprocessor(["The quick brown fox jumped.", "Call me Ishmael."])
    # Tokenize a dictionary with separate encoder and decoder inputs.
    preprocessor({
        "encoder_text": "The quick brown fox jumped.",
        "decoder_text": "The fast fox."
    })

    # Apply tokenization to a `tf.data.Dataset`.
    encoder_features = tf.constant(["The quick brown fox.", "Call me Ishmael."])
    decoder_features = tf.constant(["The fast fox.", "I am Ishmael."])
    ds = tf.data.Dataset.from_tensor_slices(
        {"encoder_text": encoder_features, "decoder_text": decoder_features}
    )
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Prepare tokens for generation.
    preprocessor.generate_preprocess({
        "encoder_text": "The quick brown fox jumped.",
        "decoder_text": "The fast fox."
    })

    # Map generation outputs back to strings.
    preprocessor.generate_postprocess({
        'decoder_token_ids': np.array([[2, 714, 4320, 8426, 25341, 1, 0, 0]]),
        'decoder_padding_mask': np.array([[1, 1, 1, 1, 1, 1, 0, 0]]),
    })
    ```
    """

    backbone_cls = T5GemmaBackbone
    tokenizer_cls = T5GemmaTokenizer
