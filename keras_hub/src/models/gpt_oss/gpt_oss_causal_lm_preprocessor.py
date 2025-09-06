from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.gpt_oss.gpt_oss_backbone import GptOssBackbone
from keras_hub.src.models.gpt_oss.gpt_oss_tokenizer import GptOssTokenizer


@keras_hub_export("keras_hub.models.GptOssCausalLMPreprocessor")
class GptOssCausalLMPreprocessor(CausalLMPreprocessor):
    """GPT-OSS Causal LM preprocessor.

    This preprocessing layer is meant for use with
    `keras_hub.models.GptOssCausalLM`. By default, it will take in batches of
    strings, and return outputs in a `(x, y, sample_weight)` format, where the
    `y` label is the next token id in the `x` sequence.

    For use with generation, the layer also exposes two methods
    `generate_preprocess()` and `generate_postprocess()`. When this preprocessor
    is attached to a `keras_hub.models.GptOssCausalLM` instance, these methods
    will be called implicitly in `generate()`. They can also be called
    standalone (e.g. to precompute preprocessing inputs for generation in a
    separate process).

    Args:
        tokenizer: A `keras_hub.models.GptOssTokenizer` instance.
        sequence_length: The length of the packed inputs.
        add_start_token: If `True`, the preprocessor will prepend the tokenizer
            start token to each input sequence. Default is `True`.
        add_end_token: If `True`, the preprocessor will append the tokenizer
            end token to each input sequence. Default is `False`.

    Call arguments:
        x: A string, `tf.Tensor` or list of python strings.
        y: Label data. Should always be `None` as the layer generates labels.
        sample_weight: Label weights. Should always be `None` as the layer
            generates label weights.
        sequence_length: Pass to override the configured `sequence_length` of
            the layer.

    Examples:
    ```python
    import tensorflow as tf
    import keras_hub

    # Load the preprocessor from a preset.
    # Assuming a preset named "gpt_oss_base_en" exists for GPT-OSS.
    preprocessor = keras_hub.models.GptOssCausalLMPreprocessor.from_preset(
        "gpt_oss_base_en"
    )

    # Tokenize and pack a single sentence.
    sentence = tf.constant("The quick brown fox jumps over the lazy dog.")
    x, y, sample_weight = preprocessor(sentence)
    print("Single sentence output:")
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("sample_weight shape:", sample_weight.shape)

    # Same output with a Python string.
    x, y, sample_weight = preprocessor(
        "The quick brown fox jumps over the lazy dog.")
    print("\nSingle Python string output:")
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("sample_weight shape:", sample_weight.shape)

    # Tokenize a batch of sentences.
    sentences = tf.constant([
        "Hello, how are you doing today?",
        "Keras is an amazing deep learning framework!"
    ])
    x, y, sample_weight = preprocessor(sentences)
    print("\nBatch of sentences output:")
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("sample_weight shape:", sample_weight.shape)

    # Same output with a list of Python strings.
    x, y, sample_weight = preprocessor([
        "Hello, how are you doing today?",
        "Keras is an amazing deep learning framework!"
    ])
    print("\nBatch of Python strings output:")
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("sample_weight shape:", sample_weight.shape)

    # Map a dataset to preprocess a single sentence with labels.
    features = tf.constant(
        [
            "The weather is beautiful today.",
            "I love building models with Keras."
        ]
    )
    labels = tf.constant([1, 0])
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    print("\nDataset mapped with labels:")
    for x_ds, y_ds, sw_ds in ds.take(1):
        print("x_ds shape:", x_ds.shape)
        print("y_ds shape:", y_ds.shape)
        print("sw_ds shape:", sw_ds.shape)

    # Map a dataset to preprocess unlabeled sentences.
    ds_unlabeled = tf.data.Dataset.from_tensor_slices(features)
    ds_unlabeled = ds_unlabeled.map(
        preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    print("\nDataset mapped without labels:")
    for x_ds, y_ds, sw_ds in ds_unlabeled.take(1):
        print("x_ds shape:", x_ds.shape)
        print("y_ds shape:", y_ds.shape)
        print("sw_ds shape:", sw_ds.shape)
    ```
    """

    backbone_cls = GptOssBackbone
    tokenizer_cls = GptOssTokenizer

    def __init__(
        self,
        tokenizer: GptOssTokenizer,
        sequence_length: int,
        add_start_token: bool = True,
        add_end_token: bool = False,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            add_start_token=add_start_token,
            add_end_token=add_end_token,
            **kwargs,
        )
