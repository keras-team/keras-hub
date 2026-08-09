import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.edrec.edrec_backbone import EdRecBackbone
from keras_hub.src.models.seq_2_seq_lm import Seq2SeqLM
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.EdRecSeq2SeqLM")
class EdRecSeq2SeqLM(Seq2SeqLM):
    """EdRec Seq2SeqLM.

    Args:
        backbone: A `keras_hub.models.EdRecBackbone` instance.
        preprocessor: Optional preprocessor.
    """

    backbone_cls = EdRecBackbone
    preprocessor_cls = None

    def __init__(
        self,
        backbone,
        preprocessor=None,
        **kwargs,
    ):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

        # LM Head
        self.lm_head = keras.layers.Dense(
            backbone.vocab_size, use_bias=False, name="lm_head"
        )

        # === Functional Model ===
        encoder_token_ids = keras.Input(
            shape=(None,), dtype="int32", name="encoder_token_ids"
        )
        decoder_token_ids = keras.Input(
            shape=(None,), dtype="int32", name="decoder_token_ids"
        )
        encoder_padding_mask = keras.Input(
            shape=(None,), dtype="bool", name="encoder_padding_mask"
        )
        decoder_padding_mask = keras.Input(
            shape=(None,), dtype="bool", name="decoder_padding_mask"
        )

        inputs = {
            "encoder_token_ids": encoder_token_ids,
            "decoder_token_ids": decoder_token_ids,
            "encoder_padding_mask": encoder_padding_mask,
            "decoder_padding_mask": decoder_padding_mask,
        }

        backbone_outputs = backbone(inputs)
        # The backbone returns a dict; we likely want the decoder output for the
        # LM head if both are present, or just use what makes sense.
        # For a Seq2Seq model training, we usually consume the decoder output.
        outputs = self.lm_head(backbone_outputs["decoder_sequence_output"])

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

    def call_decoder_with_cache(
        self,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_token_ids,
        decoder_padding_mask=None,
        self_attention_cache=None,
        self_attention_cache_update_index=None,
        cross_attention_cache=None,
        cross_attention_cache_update_index=None,
    ):
        x = self.backbone.embedding(decoder_token_ids)
        if decoder_padding_mask is None:
            decoder_padding_mask = ops.not_equal(decoder_token_ids, 0)

        self_attention_caches = []
        cross_attention_caches = []

        for i, layer in enumerate(self.backbone.decoder_layers):
            current_self_cache = (
                self_attention_cache[:, i, ...]
                if self_attention_cache is not None
                else None
            )
            current_cross_cache = (
                cross_attention_cache[:, i, ...]
                if cross_attention_cache is not None
                else None
            )

            x, next_self, next_cross = layer(
                x,
                encoder_outputs=encoder_hidden_states,
                decoder_padding_mask=decoder_padding_mask,
                encoder_padding_mask=encoder_padding_mask,
                self_attention_cache=current_self_cache,
                self_attention_cache_update_index=self_attention_cache_update_index,
                cross_attention_cache=current_cross_cache,
                cross_attention_cache_update_index=cross_attention_cache_update_index,
            )

            if next_self is not None:
                self_attention_caches.append(next_self)
            if next_cross is not None:
                cross_attention_caches.append(next_cross)

        if self_attention_cache_update_index is not None:
            self_attention_cache = ops.stack(self_attention_caches, axis=1)
        if cross_attention_cache_update_index is not None:
            cross_attention_cache = ops.stack(cross_attention_caches, axis=1)

        hidden_states = x
        logits = self.lm_head(x)
        return (
            logits,
            hidden_states,
            self_attention_cache,
            cross_attention_cache,
        )

    def call_encoder(self, token_ids, padding_mask):
        x = self.backbone.embedding(token_ids)
        for layer in self.backbone.encoder_layers:
            x = layer(x, padding_mask=padding_mask)
        return x

    def _initialize_cache(self, encoder_token_ids, decoder_token_ids):
        batch_size = ops.shape(encoder_token_ids)[0]
        encoder_max_length = ops.shape(encoder_token_ids)[1]
        decoder_max_length = ops.shape(decoder_token_ids)[1]

        num_layers = self.backbone.num_layers_dec
        num_heads = self.backbone.num_heads
        head_dim = self.backbone.hidden_dim // num_heads

        shape = [
            batch_size,
            num_layers,
            2,
            decoder_max_length,
            num_heads,
            head_dim,
        ]
        self_attention_cache = ops.zeros(shape, dtype=self.compute_dtype)

        shape[3] = encoder_max_length
        cross_attention_cache = ops.zeros(shape, dtype=self.compute_dtype)

        return self_attention_cache, cross_attention_cache

    def generate_step(self, inputs, stop_token_ids=None):
        encoder_token_ids = inputs["encoder_token_ids"]
        encoder_padding_mask = inputs["encoder_padding_mask"]
        decoder_token_ids = inputs.get("decoder_token_ids")
        if decoder_token_ids is None:
            batch_size = ops.shape(encoder_token_ids)[0]
            decoder_token_ids = ops.zeros((batch_size, 1), dtype="int32")

        decoder_padding_mask = inputs.get("decoder_padding_mask")
        if decoder_padding_mask is None:
            decoder_padding_mask = ops.ones_like(
                decoder_token_ids, dtype="bool"
            )

        batch_size = ops.shape(encoder_token_ids)[0]

        encoder_hidden_states = self.call_encoder(
            encoder_token_ids, encoder_padding_mask
        )
        self_attention_cache, cross_attention_cache = self._initialize_cache(
            encoder_token_ids, decoder_token_ids
        )

        row_lengths = ops.sum(ops.cast(decoder_padding_mask, "int32"), axis=-1)
        start_index = ops.min(row_lengths)

        # Init cache logic for step 0
        token_0 = ops.slice(decoder_token_ids, [0, 0], [batch_size, 1])
        mask_0 = ops.slice(decoder_padding_mask, [0, 0], [batch_size, 1])
        _, _, s_cache, c_cache = self.call_decoder_with_cache(
            encoder_hidden_states,
            encoder_padding_mask,
            token_0,
            mask_0,
            self_attention_cache,
            0,
            cross_attention_cache,
            0,
        )

        # We define cache as tuple
        cache = (s_cache, c_cache)
        hidden_states = ops.zeros_like(token_0, dtype="float32")

        def next(prompt, cache, index):
            s_c, c_c = cache

            # Handle beam search replication if needed
            curr_batch = ops.shape(prompt)[0]
            enc_batch = ops.shape(encoder_hidden_states)[0]

            enc_states = encoder_hidden_states
            enc_mask = encoder_padding_mask

            if curr_batch != enc_batch:
                repeats = curr_batch // enc_batch
                enc_states = ops.repeat(enc_states, repeats, axis=0)
                enc_mask = ops.repeat(enc_mask, repeats, axis=0)

            cache_index = index - 1
            num_samples = ops.shape(prompt)[0]
            prompt_slice = ops.slice(prompt, [0, cache_index], [num_samples, 1])

            logits, h_states, next_s, next_c = self.call_decoder_with_cache(
                enc_states,
                enc_mask,
                prompt_slice,
                None,
                s_c,
                index - 1,
                c_c,
                None,  # Cross cache re-use
            )

            # If the backbone returns the full sequence, we only need the last
            # token.
            if ops.shape(logits)[1] != 1:
                logits = ops.take(logits, [cache_index], axis=1)
                h_states = ops.take(h_states, [cache_index], axis=1)

            return (
                ops.squeeze(logits, axis=1),
                ops.squeeze(h_states, axis=1),
                (next_s, next_c),
            )

        new_tokens = self.sampler(
            next=next,
            prompt=decoder_token_ids,
            cache=cache,
            index=start_index,
            mask=decoder_padding_mask,
            stop_token_ids=stop_token_ids,
            hidden_states=hidden_states,
            model=self,
        )

        if stop_token_ids is not None:
            end_locations = any_equal(
                new_tokens,
                stop_token_ids,
                ops.logical_not(decoder_padding_mask),
            )
            end_locations = ops.cast(end_locations, "int32")
            cumsum = ops.cast(ops.cumsum(end_locations, axis=-1), "int32")
            overflow = cumsum - end_locations
            decoder_padding_mask = ops.logical_not(ops.cast(overflow, "bool"))
        else:
            decoder_padding_mask = ops.ones_like(new_tokens, dtype="bool")

        return {
            "decoder_token_ids": new_tokens,
            "decoder_padding_mask": decoder_padding_mask,
        }
