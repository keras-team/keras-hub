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
    sequence, and decodes the answer. Because the underlying T5 stack does not
    support a key/value cache, generation recomputes the decoder at each step;
    the vision encoder and Q-Former are only run once per `generate()` call.

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

    def _qformer_features(
        self, images, qformer_token_ids=None, qformer_padding_mask=None
    ):
        """Run the vision encoder + Q-Former to obtain visual query features."""
        if ops.ndim(images) == 3:
            images = ops.expand_dims(images, axis=0)
        vision_features = self.backbone.vision_encoder(images)
        qformer = self.backbone.qformer
        if getattr(qformer, "instruction_aware", False):
            return qformer(
                {
                    "vision_features": vision_features,
                    "qformer_token_ids": qformer_token_ids,
                    "qformer_padding_mask": qformer_padding_mask,
                }
            )
        return qformer(vision_features)

    def call_encoder(
        self,
        encoder_token_ids,
        encoder_padding_mask,
        images=None,
        qformer_token_ids=None,
        qformer_padding_mask=None,
    ):
        """Encode the prompt and (optionally) the image into encoder states.

        Returns the encoder hidden states and the encoder attention mask. When
        an image is provided, the projected Q-Former features are prepended to
        the encoder sequence as a visual soft-prompt and reflected in the mask.
        """
        qformer_features = None
        if images is not None:
            qformer_features = self._qformer_features(
                images, qformer_token_ids, qformer_padding_mask
            )
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

    def generate_step(self, inputs, stop_token_ids=None):
        """A compilable generation function for a single batch of inputs.

        Args:
            inputs: A dictionary with keys `"encoder_token_ids"`,
                `"encoder_padding_mask"`, `"decoder_token_ids"` and
                `"decoder_padding_mask"`, plus optional `"images"` (and
                `"qformer_token_ids"` / `"qformer_padding_mask"` for
                instruction-aware variants), with batched tensor values.
            stop_token_ids: Tuple of id's of end tokens to stop on. If all
                sequences have produced a new stop token, generation will stop.
        """
        encoder_token_ids = inputs["encoder_token_ids"]
        encoder_padding_mask = inputs["encoder_padding_mask"]
        decoder_token_ids = inputs["decoder_token_ids"]
        decoder_padding_mask = inputs["decoder_padding_mask"]
        images = inputs.get("images")
        qformer_token_ids = inputs.get("qformer_token_ids")
        qformer_padding_mask = inputs.get("qformer_padding_mask")
        lm_head = self.backbone.language_model.lm_head

        # Encode the image + prompt once; reused for every decoding step.
        encoder_hidden_states, encoder_attention_mask = self.call_encoder(
            encoder_token_ids,
            encoder_padding_mask,
            images,
            qformer_token_ids,
            qformer_padding_mask,
        )

        # Seed the decoder hidden states for the sampler.
        hidden_states = self.call_decoder(
            decoder_token_ids,
            decoder_padding_mask,
            encoder_hidden_states,
            encoder_attention_mask,
        )

        batch_size = ops.shape(decoder_token_ids)[0]
        # Compute the lengths of all user inputted decoder token ids.
        row_lengths = ops.sum(ops.cast(decoder_padding_mask, "int32"), axis=-1)
        # Start at the first index that has no user inputted id.
        index = ops.min(row_lengths)

        def next(prompt, cache, index):
            num_samples = ops.shape(prompt)[0]

            def repeat_for_beams(x):
                """Repeats along the batch axis to match beam-search width."""
                if ops.shape(x)[0] == num_samples:
                    return x
                return ops.repeat(x, num_samples // batch_size, axis=0)

            # T5 has no KV cache, so recompute the decoder over the full prompt.
            # The causal mask ensures position `index - 1` only attends to real
            # (already generated) tokens, so a constant decoder mask is safe.
            decoder_mask = ops.ones_like(prompt)
            hidden = self.call_decoder(
                prompt,
                decoder_mask,
                repeat_for_beams(encoder_hidden_states),
                repeat_for_beams(encoder_attention_mask),
            )
            logits = lm_head(hidden)
            # Select the next-token logits/hidden states at position index - 1.
            logits = ops.take(logits, index - 1, axis=1)
            hidden = ops.take(hidden, index - 1, axis=1)
            return logits, hidden, cache

        decoder_token_ids = self.sampler(
            next=next,
            prompt=decoder_token_ids,
            cache=None,
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
