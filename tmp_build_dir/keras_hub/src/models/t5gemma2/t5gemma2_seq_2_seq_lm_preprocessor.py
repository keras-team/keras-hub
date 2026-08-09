import keras
import numpy as np

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.seq_2_seq_lm_preprocessor import Seq2SeqLMPreprocessor
from keras_hub.src.models.t5gemma2.t5gemma2_backbone import T5Gemma2Backbone
from keras_hub.src.models.t5gemma2.t5gemma2_image_converter import (
    T5Gemma2ImageConverter,
)
from keras_hub.src.models.t5gemma2.t5gemma2_tokenizer import T5Gemma2Tokenizer
from keras_hub.src.utils.tensor_utils import preprocessing_function

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.models.T5Gemma2Seq2SeqLMPreprocessor")
class T5Gemma2Seq2SeqLMPreprocessor(Seq2SeqLMPreprocessor):
    """T5Gemma2 Seq2Seq LM preprocessor.

    This preprocessing layer is meant for use with
    `keras_hub.models.T5Gemma2Seq2SeqLM`. By default, it will take in
    batches of strings, and return outputs in a
    `(x, y, sample_weight)` format, where the `y` label is the next
    token id in the `x` sequence.

    For use with generation, the layer also exposes two methods
    `generate_preprocess()` and `generate_postprocess()`. When this
    preprocessor is attached to a `keras_hub.models.T5Gemma2Seq2SeqLM`
    instance, these methods will be called implicitly in `generate()`.

    When an `image_converter` is provided, the preprocessor also
    supports multimodal inputs with images. Images are inserted into
    the encoder sequence as placeholder tokens that the backbone's
    vision encoder will replace with image embeddings.

    Args:
        tokenizer: A `keras_hub.models.T5Gemma2Tokenizer` instance.
        encoder_sequence_length: The length of the packed encoder inputs.
        decoder_sequence_length: The length of the packed decoder inputs.
        image_converter: A `keras_hub.layers.ImageConverter` instance,
            or `None` for text-only. Defaults to `None`.
        add_start_token: If `True`, prepend the start token. Defaults
            to `False`.
        add_end_token: If `True`, append the end token. Defaults to
            `True`.
    """

    backbone_cls = T5Gemma2Backbone
    tokenizer_cls = T5Gemma2Tokenizer
    image_converter_cls = T5Gemma2ImageConverter

    def __init__(
        self,
        tokenizer,
        encoder_sequence_length=512,
        decoder_sequence_length=512,
        image_converter=None,
        image_size=None,
        num_vision_tokens_per_image=None,
        add_start_token=False,
        add_end_token=True,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            encoder_sequence_length=encoder_sequence_length,
            decoder_sequence_length=decoder_sequence_length,
            **kwargs,
        )
        self.add_start_token = add_start_token
        self.add_end_token = add_end_token
        self.image_converter = image_converter
        self._vision_image_size = image_size
        self.num_vision_tokens_per_image = num_vision_tokens_per_image

    def _add_vision_inputs(self, x, batch_size):
        """Add dummy image/vision_indices for multimodal text-only input.

        When a multimodal backbone (with vision encoder) is used for
        text-only inference, the functional model still requires
        `images` and `vision_indices` inputs. This method provides
        dummy values that act as a no-op: InterleaveEmbeddings
        restores position 0 after scattering zero-indexed updates.
        """
        if self._vision_image_size is not None and "images" not in x:
            x["images"] = np.zeros(
                (
                    batch_size,
                    1,
                    self._vision_image_size,
                    self._vision_image_size,
                    3,
                ),
                dtype="float32",
            )
            x["vision_indices"] = np.zeros(
                (batch_size, self.num_vision_tokens_per_image),
                dtype="int32",
            )
        return x

    def _preprocess_images(self, images):
        """Process images through the image converter.

        Accepts images in any of the following formats:

        - A single image `(H, W, C)`.
        - A batch of images `(B, H, W, C)`.
        - A batch with an explicit num-images dimension `(B, 1, H, W, C)`.

        Returns a `float32` tensor of shape
        `(batch_size, 1, image_size, image_size, 3)`.
        """
        image_size = self._vision_image_size

        if isinstance(images, (list, np.ndarray)):
            images = tf.cast(tf.constant(images), dtype="float32")

        # Handle unbatched single image: (H, W, C) → (1, H, W, C).
        if len(images.shape) == 3:
            images = tf.expand_dims(images, axis=0)

        if len(images.shape) == 5:
            # (B, 1, H, W, C): collapse the num-images dimension before
            # passing through the converter, then restore it.
            s = tf.shape(images)
            images = tf.reshape(images, [s[0] * s[1], s[2], s[3], s[4]])
            images = self.image_converter(images)
            images = tf.reshape(images, [s[0], s[1], image_size, image_size, 3])
        else:
            # (B, H, W, C): process directly, then add a num-images dim.
            images = self.image_converter(images)
            images = tf.expand_dims(images, axis=1)  # → (B, 1, H, W, C)

        return tf.cast(images, "float32")

    def _get_vision_indices(self, vision_mask):
        """Compute `vision_indices` from the encoder token vision mask.

        Finds the positions of all image placeholder tokens in the
        encoder sequence for each batch element, then pads (or
        truncates) the result to `num_vision_tokens_per_image` with
        `0` as the padding value (matching the dummy-input convention
        used by `_add_vision_inputs`).

        Args:
            vision_mask: bool tensor of shape `(batch_size, seq_len)`.
                `True` at every position occupied by an image
                placeholder token.

        Returns:
            `int32` tensor of shape
            `(batch_size, num_vision_tokens_per_image)`.
        """
        batch_size = tf.shape(vision_mask)[0]
        seq_len = tf.shape(vision_mask)[1]

        # Flatten and collect the indices of all True entries.
        flat_mask = tf.reshape(vision_mask, [-1])
        vision_indices = tf.cast(tf.where(flat_mask)[..., 0], dtype=tf.int32)

        # Per-sample image-token counts → ragged tensor of indices.
        row_lengths = tf.reduce_sum(tf.cast(vision_mask, tf.int32), axis=1)
        batched_indices = tf.RaggedTensor.from_row_lengths(
            values=vision_indices, row_lengths=row_lengths
        )

        # Subtract the per-sample flat offset so that each index is
        # relative to the start of its own sequence.
        offsets = tf.expand_dims(
            tf.range(batch_size, dtype=tf.int32) * seq_len, axis=-1
        )
        batched_indices = tf.math.subtract(batched_indices, offsets)

        # Convert to a dense tensor, padding short rows with 0.
        batched_indices = batched_indices.to_tensor(
            shape=[None, self.num_vision_tokens_per_image],
            default_value=0,
        )
        return tf.cast(batched_indices, tf.int32)

    @preprocessing_function
    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        *,
        encoder_sequence_length=None,
        decoder_sequence_length=None,
        sequence_length=None,
    ):
        if encoder_sequence_length is None:
            encoder_sequence_length = self.encoder_sequence_length
        decoder_sequence_length = decoder_sequence_length or sequence_length
        if decoder_sequence_length is None:
            decoder_sequence_length = self.decoder_sequence_length

        encoder_inputs = self.tokenizer(x["encoder_text"])
        encoder_token_ids, encoder_padding_mask = self.encoder_packer(
            encoder_inputs,
            sequence_length=encoder_sequence_length,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )
        decoder_inputs = self.tokenizer(x["decoder_text"])
        decoder_token_ids, decoder_padding_mask = self.decoder_packer(
            decoder_inputs,
            sequence_length=decoder_sequence_length + 1,
            add_start_value=True,
            add_end_value=self.add_end_token,
        )
        batch_size = tf.shape(encoder_token_ids)[0]
        x = {
            "encoder_token_ids": encoder_token_ids,
            "encoder_padding_mask": encoder_padding_mask,
            "decoder_token_ids": decoder_token_ids[..., :-1],
            "decoder_padding_mask": decoder_padding_mask[..., :-1],
        }
        x = self._add_vision_inputs(x, batch_size)
        y = decoder_token_ids[..., 1:]
        sample_weight = decoder_padding_mask[..., 1:]
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    @preprocessing_function
    def generate_preprocess(
        self,
        x,
        *,
        encoder_sequence_length=None,
        decoder_sequence_length=None,
        sequence_length=None,
    ):
        if not self.built:
            self.build(None)

        if isinstance(x, dict):
            encoder_text = x["encoder_text"]
            decoder_text = x["decoder_text"]
            images = x.get("images", None)
        else:
            encoder_text = x
            decoder_text = tf.fill((tf.shape(encoder_text)[0],), "")
            images = None

        if encoder_sequence_length is None:
            encoder_sequence_length = self.encoder_sequence_length
        decoder_sequence_length = decoder_sequence_length or sequence_length
        if decoder_sequence_length is None:
            decoder_sequence_length = self.decoder_sequence_length

        if self._vision_image_size is not None:
            encoder_text = tf.strings.regex_replace(
                encoder_text,
                "<start_of_image>",
                "\n\n<start_of_image>"
                + "<image_soft_token>" * self.num_vision_tokens_per_image
                + "<end_of_image>\n\n",
            )

        encoder_token_ids = self.tokenizer(encoder_text)
        encoder_token_ids, encoder_padding_mask = self.encoder_packer(
            encoder_token_ids,
            sequence_length=None,
            add_start_value=self.add_start_token,
            add_end_value=False,
        )

        decoder_token_ids = self.tokenizer(decoder_text)
        decoder_token_ids, decoder_padding_mask = self.decoder_packer(
            decoder_token_ids,
            sequence_length=decoder_sequence_length,
            add_start_value=True,
            add_end_value=False,
        )

        batch_size = tf.shape(encoder_token_ids)[0]

        out = {
            "encoder_token_ids": encoder_token_ids,
            "encoder_padding_mask": encoder_padding_mask,
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_padding_mask,
        }

        if self._vision_image_size is None:
            # Text-only model: no vision inputs needed.
            return out

        if images is not None:
            image_soft_token_id = self.tokenizer.token_to_id(
                "<image_soft_token>"
            )
            out["images"] = self._preprocess_images(images)
            vision_mask = tf.equal(
                encoder_token_ids,
                tf.cast(image_soft_token_id, encoder_token_ids.dtype),
            )
            if len(tf.shape(vision_mask)) == 1:
                vision_mask = tf.expand_dims(vision_mask, axis=0)
            out["vision_indices"] = self._get_vision_indices(vision_mask)
        else:
            # Vision model, text-only input: add dummy zero-filled tensors
            # so the functional model's input spec is satisfied.
            # `InterleaveEmbeddings` restores position 0 after scattering
            # zero-indexed updates, so these values are a no-op.
            out = self._add_vision_inputs(out, batch_size)

        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "add_start_token": self.add_start_token,
                "add_end_token": self.add_end_token,
                "image_size": self._vision_image_size,
                "num_vision_tokens_per_image": (
                    self.num_vision_tokens_per_image
                ),
            }
        )
        return config
