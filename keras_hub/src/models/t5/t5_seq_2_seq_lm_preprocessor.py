from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_hub.src.models.seq_2_seq_lm_preprocessor import Seq2SeqLMPreprocessor
from keras_hub.src.models.t5.t5_backbone import T5Backbone
from keras_hub.src.models.t5.t5_tokenizer import T5Tokenizer


@keras_hub_export("keras_hub.models.T5Seq2SeqLMPreprocessor")
class T5Seq2SeqLMPreprocessor(Seq2SeqLMPreprocessor):
    """T5 Seq2Seq LM preprocessor.

    This layer is used as preprocessor for seq2seq tasks using the T5 model.
    This class subclasses `keras_hub.models.Seq2SeqLMPreprocessor` and keeps
    most of its functionality, changing only how the two sequences are packed:

     1. T5 has no beginning-of-sequence token, so the encoder input is packed
        as `[tokens..., "</s>", padding...]`.
     2. The decoder input is shifted one step to the right by T5's decoder
        start token, which is the padding token, giving
        `["<pad>", tokens..., "</s>", padding...]`.

    Like the superclass, this layer sets the `y` (label) and `sample_weight`
    fields by shifting the decoder input sequence one step to the left, and
    drops the last token of the decoder input as it has no successor.

    Args:
        tokenizer: A `keras_hub.models.T5Tokenizer` instance.
        encoder_sequence_length: The length of the packed encoder inputs.
        decoder_sequence_length: The length of the packed decoder inputs.

    Call arguments:
        x: A dictionary with `encoder_text` and `decoder_text` as its keys.
            Each value in the dictionary should be a tensor of single string
            sequences. Inputs may be batched or unbatched. Raw python inputs
            will be converted to tensors.
        y: Label data. Should always be `None` as the layer generates labels by
            shifting the decoder input sequence one step to the left.
        sample_weight: Label weights. Should always be `None` as the layer
            generates label weights by shifting the padding mask one step to
            the left.

    Examples:

    Directly calling the layer on data.
    ```python
    preprocessor = keras_hub.models.T5Seq2SeqLMPreprocessor.from_preset(
        "t5_small_multi"
    )

    # Preprocess unbatched inputs.
    inputs = {
        "encoder_text": "translate English to German: The fox was sleeping.",
        "decoder_text": "Der Fuchs schlief.",
    }
    preprocessor(inputs)

    # Preprocess batched inputs.
    inputs = {
        "encoder_text": ["summarize: The fox was sleeping."],
        "decoder_text": ["The fox slept."],
    }
    preprocessor(inputs)
    ```

    Mapping with `tf.data.Dataset`.
    ```python
    preprocessor = keras_hub.models.T5Seq2SeqLMPreprocessor.from_preset(
        "t5_small_multi"
    )

    features = {
        "encoder_text": tf.constant(["summarize: The fox was sleeping."]),
        "decoder_text": tf.constant(["The fox slept."]),
    }
    ds = tf.data.Dataset.from_tensor_slices(features)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    ```
    """

    backbone_cls = T5Backbone
    tokenizer_cls = T5Tokenizer

    def build(self, input_shape):
        # Both packers are built here rather than by the superclass, whose
        # `start_value` would be T5's `start_token` — an alias for `"</s>"`
        # that T5 packs into neither sequence.
        self.encoder_packer = StartEndPacker(
            start_value=None,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sequence_length=self.encoder_sequence_length,
            return_padding_mask=True,
        )
        self.decoder_packer = StartEndPacker(
            start_value=self.tokenizer.pad_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sequence_length=self.decoder_sequence_length,
            return_padding_mask=True,
        )
        self.built = True
