import keras

try:
    import tensorflow as tf
except ImportError:
    tf = None
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_hub.src.models.audio_to_text_preprocessor import (
    AudioToTextPreprocessor,
)
from keras_hub.src.models.moonshine.moonshine_backbone import MoonshineBackbone
from keras_hub.src.models.moonshine.moonshine_tokenizer import (
    MoonshineTokenizer,
)
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.models.MoonshineAudioToTextPreprocessor")
class MoonshineAudioToTextPreprocessor(AudioToTextPreprocessor):
    """Moonshine Seq2Seq LM preprocessor for audio-to-text tasks.

    This preprocessor converts raw audio and text inputs into a format suitable
    for the `MoonshineAudioToText` model. It processes audio waveforms using
    `MoonshineAudioConverter` for basic preprocessing (padding, normalization)
    and tokenizes text using `MoonshineTokenizer` for the decoder. It supports
    training and generation.

    Args:
        audio_converter: A `MoonshineAudioConverter` instance to process audio.
        tokenizer: A `MoonshineTokenizer` instance to tokenize text.
        decoder_sequence_length: int, optional. Maximum length for decoder token
            sequences. Defaults to 1024.
        **kwargs: Additional keyword arguments for the parent class.

    Examples:
    ```python
    import keras
    from keras_hub.layers import MoonshineAudioConverter
    from keras_hub.models import MoonshineTokenizer

    # Create audio converter and tokenizer instances.
    audio_converter = MoonshineAudioConverter()
    tokenizer = MoonshineTokenizer.from_preset("moonshine_base")

    # Initialize the preprocessor.
    preprocessor = keras_hub.models.MoonshineAudioToTextPreprocessor(
        audio_converter=audio_converter,
        tokenizer=tokenizer,
        decoder_sequence_length=8
    )

    # Prepare input data (audio tensor and text).
    inputs = {
        "audio": keras.random.normal((1, 16000)),
        "text": ["the quick brown fox"]
    }

    # Process the inputs for training.
    x, y, sample_weight = preprocessor(inputs)

    # Check output keys and shapes (shapes depend on padding/truncation).
    print(x.keys())
    # dict_keys(['encoder_input_values', 'encoder_padding_mask',
    # 'decoder_token_ids', 'decoder_padding_mask']).
    print(x["encoder_input_values"].shape) # e.g., (1, 16000, 1) / padded length
    print(x["encoder_padding_mask"].shape) # e.g., (1, 16000) or padded length
    print(x["decoder_token_ids"].shape) # (1, 8)
    print(x["decoder_padding_mask"].shape) # (1, 8)
    print(y.shape) # (1, 8) - Labels
    print(sample_weight.shape) # (1, 8) - Sample weights

    # Process inputs for generation.
    gen_inputs = preprocessor.generate_preprocess(inputs)
    print(gen_inputs.keys())
    # dict_keys(['encoder_input_values', 'encoder_padding_mask',
    # 'decoder_token_ids', 'decoder_padding_mask']).
    ```
    """

    backbone_cls = MoonshineBackbone
    tokenizer_cls = MoonshineTokenizer

    def __init__(
        self,
        audio_converter,
        tokenizer,
        decoder_sequence_length=1024,
        **kwargs,
    ):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.audio_converter = audio_converter
        self.decoder_sequence_length = decoder_sequence_length
        self.decoder_packer = None
        self._special_token_ids_set = None

    def build(self, input_shape):
        self.decoder_packer = StartEndPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sequence_length=self.decoder_sequence_length,
            return_padding_mask=True,
        )
        self._special_token_ids_set = set(self.tokenizer.special_token_ids)
        if self.tokenizer.pad_token_id is not None:
            self._special_token_ids_set.add(self.tokenizer.pad_token_id)
        self.built = True

    @preprocessing_function
    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        decoder_sequence_length=None,
        sequence_length=None,
    ):
        if not self.built:
            self.build(None)
        if isinstance(x, tuple) and len(x) == 1:
            x = x[0]
        decoder_sequence_length = (
            decoder_sequence_length
            or sequence_length
            or self.decoder_sequence_length
        )
        text = x["text"]
        encoder_inputs = self.audio_converter(
            x["audio"],
            padding="longest",
        )
        encoder_inputs_shape = keras.ops.shape(encoder_inputs)
        if len(encoder_inputs_shape) == 2:
            encoder_inputs = keras.ops.expand_dims(encoder_inputs, axis=-1)
        squeezed_inputs = encoder_inputs[:, :, 0]
        is_tf_symbolic = (
            tf is not None
            and hasattr(squeezed_inputs, "graph")
            and hasattr(squeezed_inputs.graph, "as_graph_def")
        )
        if is_tf_symbolic and keras.config.backend() != "tensorflow":
            encoder_padding_mask = tf.logical_not(
                tf.math.equal(
                    squeezed_inputs, float(self.audio_converter.padding_value)
                )
            )
        else:
            encoder_padding_mask = keras.ops.logical_not(
                keras.ops.equal(
                    squeezed_inputs, self.audio_converter.padding_value
                )
            )
        decoder_inputs = self.tokenizer(text)
        decoder_token_ids, decoder_padding_mask = self.decoder_packer(
            decoder_inputs,
            sequence_length=decoder_sequence_length + 1,
            add_end_value=True,
        )
        x_out = {
            "encoder_input_values": encoder_inputs,
            "encoder_padding_mask": encoder_padding_mask,
            "decoder_token_ids": decoder_token_ids[..., :-1],
            "decoder_padding_mask": decoder_padding_mask[..., :-1],
        }
        y_out = decoder_token_ids[..., 1:]
        sample_weight_out = decoder_padding_mask[..., 1:]

        return keras.utils.pack_x_y_sample_weight(
            x_out, y_out, sample_weight_out
        )

    @preprocessing_function
    def generate_preprocess(
        self,
        x,
        decoder_sequence_length=None,
        sequence_length=None,
    ):
        if not self.built:
            self.build(None)
        if isinstance(x, tuple) and len(x) == 1:
            x = x[0]
        decoder_sequence_length = (
            decoder_sequence_length
            or sequence_length
            or self.decoder_sequence_length
        )
        encoder_inputs = self.audio_converter(
            x["audio"],
            padding="longest",
        )
        encoder_inputs_shape = keras.ops.shape(encoder_inputs)
        if len(encoder_inputs_shape) == 2:
            encoder_inputs = keras.ops.expand_dims(encoder_inputs, axis=-1)
        squeezed_inputs = encoder_inputs[:, :, 0]
        is_tf_symbolic = (
            tf is not None
            and hasattr(squeezed_inputs, "graph")
            and hasattr(squeezed_inputs.graph, "as_graph_def")
        )
        if is_tf_symbolic and keras.config.backend() != "tensorflow":
            encoder_padding_mask = tf.logical_not(
                tf.math.equal(
                    squeezed_inputs, float(self.audio_converter.padding_value)
                )
            )
        else:
            encoder_padding_mask = keras.ops.logical_not(
                keras.ops.equal(
                    squeezed_inputs, self.audio_converter.padding_value
                )
            )
        audio_batch_size = keras.ops.shape(x["audio"])[0]
        decoder_text = x.get("text", None)
        if decoder_text is None:
            decoder_token_ids = [
                [self.tokenizer.start_token_id]
            ] * audio_batch_size
        else:
            if isinstance(decoder_text, str):
                decoder_text = [decoder_text] * audio_batch_size
            elif len(decoder_text) != audio_batch_size:
                if len(decoder_text) == 1:
                    decoder_text = decoder_text * audio_batch_size
                else:
                    raise ValueError(
                        f"Batch size mismatch between audio "
                        f"({audio_batch_size}) and text prompts "
                        f"({len(decoder_text)})"
                    )
            decoder_token_ids = self.tokenizer(decoder_text)
        decoder_token_ids, decoder_padding_mask = self.decoder_packer(
            decoder_token_ids,
            sequence_length=decoder_sequence_length,
            add_end_value=False,
        )

        return {
            "encoder_input_values": encoder_inputs,
            "encoder_padding_mask": encoder_padding_mask,
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_padding_mask,
        }

    @preprocessing_function
    def generate_postprocess(self, x):
        if not self.built:
            self.build(None)
        token_ids, padding_mask = (
            x["decoder_token_ids"],
            x["decoder_padding_mask"],
        )
        token_ids_np = keras.ops.convert_to_numpy(token_ids)
        padding_mask_np = keras.ops.convert_to_numpy(padding_mask)
        vocab_size = self.tokenizer.vocabulary_size()
        processed_sequences = []
        for i in range(token_ids_np.shape[0]):
            sequence = token_ids_np[i]
            mask = padding_mask_np[i].astype(bool)
            valid_tokens = sequence[mask]
            filtered_tokens = [
                int(token)
                for token in valid_tokens
                if token not in self._special_token_ids_set
                and 0 <= token < vocab_size
            ]
            processed_sequences.append(filtered_tokens)
        processed_sequences = tf.ragged.constant(
            processed_sequences, dtype=tf.int32
        )
        return self.tokenizer.detokenize(processed_sequences)
