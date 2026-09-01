from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.blip2.blip2_backbone import BLIP2Backbone
from keras_hub.src.models.blip2.blip2_seq_2_seq_lm_preprocessor import (
    BLIP2Seq2SeqLMPreprocessor,
)
from keras_hub.src.models.seq_2_seq_lm import Seq2SeqLM
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.BLIP2Seq2SeqLM")
class BLIP2Seq2SeqLM(Seq2SeqLM):
    """An end-to-end multimodal BLIP-2 model for seq2seq language modeling.

    This is the encoder-decoder (Flan-T5) BLIP-2 task. A seq2seq language model
    is conditioned on an input "context" — here the encoder text prompt plus a
    Q-Former visual soft-prompt distilled from the image — and the decoder
    autoregressively predicts the output text (e.g. a caption or VQA answer).

    The forward pass runs the frozen vision encoder and the Q-Former once to
    obtain visual query features, projects and prepends them to the T5 encoder
    sequence, and decodes the answer. During generation the vision encoder,
    Q-Former and T5 encoder are run once per `generate()` call, and the decoder
    reuses cached self-attention and cross-attention key/value tensors so each
    step only attends over a single new token.

    This model has a `generate()` method, which generates text based on the
    image and an optional encoder/decoder prompt. The generation strategy used
    is controlled by an additional `sampler` argument on `compile()`. By
    default, `"greedy"` sampling will be used.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to raw inputs during
    `fit()`, `predict()`, `evaluate()` and `generate()`. This is done by default
    when creating the model with `from_preset()`.

    Args:
        backbone: A `keras_hub.models.BLIP2Backbone` instance whose
            `language_model` is a `keras_hub.models.BLIP2FlanT5`.
        preprocessor: A `keras_hub.models.BLIP2Seq2SeqLMPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model. Defaults
            to `None`.
    """

    backbone_cls = BLIP2Backbone
    preprocessor_cls = BLIP2Seq2SeqLMPreprocessor

    def __init__(self, backbone, preprocessor=None, **kwargs):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

        # === Functional Model ===
        inputs = backbone.input
        hidden_states = backbone(inputs)
        outputs = backbone.language_model.lm_head(hidden_states)
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

    def _qformer_features(self, images):
        """Run the vision encoder + Q-Former to obtain visual query features."""
        if ops.ndim(images) == 3:
            images = ops.expand_dims(images, axis=0)
        vision_features = self.backbone.vision_encoder(images)
        return self.backbone.qformer(vision_features)

    def call_encoder(
        self,
        encoder_token_ids,
        encoder_padding_mask,
        images=None,
    ):
        """Encode the prompt and (optionally) the image into encoder states.

        Returns the encoder hidden states and the encoder attention mask. When
        an image is provided, the projected Q-Former features are prepended to
        the encoder sequence as a visual soft-prompt and reflected in the mask.
        """
        qformer_features = None
        if images is not None:
            qformer_features = self._qformer_features(images)
        return self.backbone.language_model.call_encoder(
            encoder_token_ids, encoder_padding_mask, qformer_features
        )

    def call_decoder(
        self,
        decoder_token_ids,
        decoder_padding_mask,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        """Run the T5 decoder and return its hidden states."""
        return self.backbone.language_model.call_decoder(
            decoder_token_ids,
            decoder_padding_mask,
            encoder_hidden_states,
            encoder_attention_mask,
        )

    def call_decoder_with_cache(
        self,
        decoder_token_ids,
        encoder_hidden_states,
        encoder_attention_mask,
        self_attention_cache,
        self_attention_cache_update_index,
        cross_attention_cache,
        cross_attention_cache_update_index,
    ):
        """Run the T5 decoder with key/value caches and return its logits.

        Returns a `(logits, hidden_states, self_attention_cache,
        cross_attention_cache)` tuple.
        """
        language_model = self.backbone.language_model
        hidden_states, self_attention_cache, cross_attention_cache = (
            language_model.call_decoder_with_cache(
                decoder_token_ids,
                encoder_hidden_states,
                encoder_attention_mask,
                self_attention_cache,
                self_attention_cache_update_index,
                cross_attention_cache,
                cross_attention_cache_update_index,
            )
        )
        logits = language_model.lm_head(hidden_states)
        return (
            logits,
            hidden_states,
            self_attention_cache,
            cross_attention_cache,
        )

    def _initialize_cache(self, encoder_hidden_states, decoder_token_ids):
        """Initializes empty self-attention and cross-attention caches."""
        language_model = self.backbone.language_model
        batch_size = ops.shape(decoder_token_ids)[0]
        decoder_max_length = ops.shape(decoder_token_ids)[1]
        # The encoder sequence includes the prepended Q-Former query tokens,
        # so its length is read off the encoder output rather than the inputs.
        encoder_max_length = ops.shape(encoder_hidden_states)[1]

        num_heads = language_model.num_heads
        head_dim = language_model.key_value_dim or (
            language_model.hidden_dim // num_heads
        )
        shape = [
            batch_size,
            language_model.num_layers,
            2,
            decoder_max_length,
            num_heads,
            head_dim,
        ]
        self_attention_cache = ops.zeros(shape, dtype=self.compute_dtype)
        shape[3] = encoder_max_length
        cross_attention_cache = ops.zeros(shape, dtype=self.compute_dtype)
        return (self_attention_cache, cross_attention_cache)

    def _build_cache(
        self, encoder_hidden_states, encoder_attention_mask, decoder_token_ids
    ):
        """Seeds both caches with a single decoder forward pass."""
        self_attention_cache, cross_attention_cache = self._initialize_cache(
            encoder_hidden_states, decoder_token_ids
        )
        # Straight to the language model, skipping the `lm_head` projection —
        # the seeding pass only needs hidden states and the caches.
        return self.backbone.language_model.call_decoder_with_cache(
            decoder_token_ids,
            encoder_hidden_states,
            encoder_attention_mask,
            self_attention_cache,
            0,
            cross_attention_cache,
            0,
        )

    def generate_step(self, inputs, stop_token_ids=None):
        """A compilable generation function for a single batch of inputs.

        Args:
            inputs: A dictionary with keys `"encoder_token_ids"`,
                `"encoder_padding_mask"`, `"decoder_token_ids"` and
                `"decoder_padding_mask"`, plus optional `"images"`, with
                batched tensor values.
            stop_token_ids: Tuple of id's of end tokens to stop on. If all
                sequences have produced a new stop token, generation will stop.
        """
        encoder_token_ids = inputs["encoder_token_ids"]
        encoder_padding_mask = inputs["encoder_padding_mask"]
        decoder_token_ids = inputs["decoder_token_ids"]
        decoder_padding_mask = inputs["decoder_padding_mask"]
        images = inputs.get("images")

        # Encode the image + prompt once; reused for every decoding step.
        encoder_hidden_states, encoder_attention_mask = self.call_encoder(
            encoder_token_ids,
            encoder_padding_mask,
            images,
        )

        # Create and seed both decoder caches with a single forward pass.
        (
            hidden_states,
            self_attention_cache,
            cross_attention_cache,
        ) = self._build_cache(
            encoder_hidden_states, encoder_attention_mask, decoder_token_ids
        )

        batch_size = ops.shape(decoder_token_ids)[0]
        # Compute the lengths of all user inputted decoder token ids.
        row_lengths = ops.sum(ops.cast(decoder_padding_mask, "int32"), axis=-1)
        # Start at the first index that has no user inputted id.
        index = ops.min(row_lengths)

        def next(prompt, cache, index):
            # The cache index is the index of our previous token.
            cache_index = index - 1
            num_samples = ops.shape(prompt)[0]
            prompt = ops.slice(prompt, [0, cache_index], [num_samples, 1])

            def repeat_for_beams(x):
                """Repeats along the batch axis to match beam-search width."""
                if ops.shape(x)[0] == num_samples:
                    return x
                return ops.repeat(x, num_samples // batch_size, axis=0)

            logits, hidden, cache, _ = self.call_decoder_with_cache(
                decoder_token_ids=prompt,
                encoder_hidden_states=repeat_for_beams(encoder_hidden_states),
                encoder_attention_mask=repeat_for_beams(encoder_attention_mask),
                self_attention_cache=cache,
                self_attention_cache_update_index=cache_index,
                cross_attention_cache=repeat_for_beams(cross_attention_cache),
                cross_attention_cache_update_index=None,
            )
            return (
                ops.squeeze(logits, axis=1),
                ops.squeeze(hidden, axis=1),
                cache,
            )

        decoder_token_ids = self.sampler(
            next=next,
            prompt=decoder_token_ids,
            cache=self_attention_cache,
            index=index,
            mask=decoder_padding_mask,
            stop_token_ids=stop_token_ids,
            hidden_states=hidden_states,
            model=self,
        )

        # Compute an output padding mask with the token ids we updated.
        if stop_token_ids is not None:
            end_locations = any_equal(
                decoder_token_ids,
                stop_token_ids,
                ops.logical_not(decoder_padding_mask),
            )
            end_locations = ops.cast(end_locations, "int32")
            cumsum = ops.cast(ops.cumsum(end_locations, axis=-1), "int32")
            overflow = cumsum - end_locations
            decoder_padding_mask = ops.logical_not(ops.cast(overflow, "bool"))
        else:
            decoder_padding_mask = ops.ones_like(
                decoder_token_ids, dtype="bool"
            )

        return {
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_padding_mask,
        }
