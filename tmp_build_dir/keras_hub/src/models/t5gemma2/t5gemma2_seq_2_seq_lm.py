from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.seq_2_seq_lm import Seq2SeqLM
from keras_hub.src.models.t5gemma2.t5gemma2_backbone import T5Gemma2Backbone
from keras_hub.src.models.t5gemma2.t5gemma2_seq_2_seq_lm_preprocessor import (
    T5Gemma2Seq2SeqLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.T5Gemma2Seq2SeqLM")
class T5Gemma2Seq2SeqLM(Seq2SeqLM):
    """An end-to-end T5Gemma2 model for seq2seq language modeling.

    A seq2seq language model (LM) is an encoder-decoder model which is
    used for conditional text generation. The encoder is given a
    "context" text (fed to the encoder), and the decoder predicts the
    next token based on both the encoder inputs and the previous tokens.

    T5Gemma2 extends T5Gemma by using Gemma3-based components, with
    merged self+cross attention in the decoder, Gemma3-style Q/K
    normalization, and per-layer-type RoPE.

    This model has a `generate()` method, which generates text based on
    a prompt. The generation strategy used is controlled by an additional
    `sampler` argument on `compile()`.

    Args:
        backbone: A `keras_hub.models.T5Gemma2Backbone` instance.
        preprocessor: A `keras_hub.models.T5Gemma2Seq2SeqLMPreprocessor`
            or `None`. Defaults to `None`.

    Examples:

    Use `generate()` to do text generation.
    ```python
    t5gemma2_lm = keras_hub.models.T5Gemma2Seq2SeqLM.from_preset(
        "t5gemma2_270m_270m"
    )
    t5gemma2_lm.generate(
        "The quick brown fox jumped.", max_length=30
    )
    ```

    Custom backbone and vocabulary.
    ```python
    tokenizer = keras_hub.models.T5Gemma2Tokenizer(
        proto="proto.spm",
    )
    preprocessor = keras_hub.models.T5Gemma2Seq2SeqLMPreprocessor(
        tokenizer=tokenizer,
        encoder_sequence_length=128,
        decoder_sequence_length=128,
    )
    backbone = keras_hub.models.T5Gemma2Backbone(
        vocabulary_size=32000,
        encoder_hidden_dim=256,
        encoder_intermediate_dim=512,
        encoder_num_layers=4,
        encoder_num_attention_heads=4,
        encoder_num_key_value_heads=2,
        encoder_head_dim=64,
        encoder_layer_types=["full_attention"] * 4,
        decoder_hidden_dim=256,
        decoder_intermediate_dim=512,
        decoder_num_layers=4,
        decoder_num_attention_heads=4,
        decoder_num_key_value_heads=2,
        decoder_head_dim=64,
        decoder_layer_types=["full_attention"] * 4,
        dropout_rate=0.1,
        rms_norm_eps=1e-6,
        query_pre_attn_scalar=1.0,
        attention_bias=False,
        hidden_activation="gelu_approximate",
    )
    t5gemma2_lm = keras_hub.models.T5Gemma2Seq2SeqLM(
        backbone=backbone,
        preprocessor=preprocessor,
    )
    ```
    """

    backbone_cls = T5Gemma2Backbone
    preprocessor_cls = T5Gemma2Seq2SeqLMPreprocessor

    def __init__(self, backbone, preprocessor=None, **kwargs):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

        # === Functional Model ===
        inputs = backbone.input
        sequence_output = backbone(inputs)["decoder_sequence_output"]
        logits = backbone.decoder_token_embedding(sequence_output, reverse=True)
        if self.backbone.final_logit_softcapping is not None:
            logits = logits / self.backbone.final_logit_softcapping
            logits = ops.tanh(logits)
            logits = logits * self.backbone.final_logit_softcapping
        super().__init__(
            inputs=inputs,
            outputs=logits,
            **kwargs,
        )

    def call_encoder(
        self,
        token_ids,
        padding_mask,
        images=None,
        vision_indices=None,
    ):
        """Process inputs through the encoder stack."""
        encoder_embeddings = self.backbone.token_embedding(token_ids)
        encoder_embeddings *= ops.cast(
            ops.sqrt(self.backbone.encoder_hidden_dim),
            encoder_embeddings.dtype,
        )

        if not self.backbone.text_only_model:
            eoi_mask = ops.cast(
                ops.expand_dims(
                    ops.equal(token_ids, self.backbone.eoi_token_index),
                    axis=-1,
                ),
                encoder_embeddings.dtype,
            )
            encoder_embeddings = (
                eoi_mask * self.backbone.encoder_eoi_embedding
                + (1 - eoi_mask) * encoder_embeddings
            )

            img_embeddings = self.backbone.vision_encoder(images)
            encoder_embeddings = self.backbone.interleave_embeddings(
                image_embeddings=img_embeddings,
                text_embeddings=encoder_embeddings,
                vision_indices=vision_indices,
            )

        encoder_hidden_states = self.backbone.encoder_dropout(
            encoder_embeddings, training=False
        )
        for layer in self.backbone.encoder_layers:
            encoder_hidden_states = layer(
                encoder_hidden_states,
                padding_mask=padding_mask,
                training=False,
            )
        encoder_output = self.backbone.encoder_norm(encoder_hidden_states)
        encoder_output = self.backbone.encoder_dropout(
            encoder_output, training=False
        )
        return encoder_output, padding_mask

    def call_decoder_with_cache(
        self,
        decoder_token_ids,
        decoder_padding_mask,
        cache,
        cache_update_index,
        encoder_output,
        encoder_padding_mask,
    ):
        """Forward pass of the decoder with cache.

        `call_decoder_with_cache` adds an additional forward pass for
        autoregressive inference. The cache stores previous key/value
        tensors in the attention layers.

        Args:
            decoder_token_ids: Dense int Tensor of shape
                `(batch_size, max_length)`.
            decoder_padding_mask: Dense int Tensor of shape
                `(batch_size, max_length)`.
            cache: Dense float Tensor, the cache of key/value states.
            cache_update_index: int or int Tensor.
            encoder_output: Dense float Tensor, encoder output.
            encoder_padding_mask: Dense int Tensor.

        Returns:
            Tuple of (logits, hidden_states, updated_cache).
        """
        self_attention_cache, cross_attention_cache = cache
        hidden_states = self.backbone.decoder_token_embedding(decoder_token_ids)
        hidden_states *= ops.cast(
            ops.sqrt(self.backbone.decoder_hidden_dim),
            hidden_states.dtype,
        )
        hidden_states = self.backbone.decoder_dropout(
            hidden_states, training=False
        )
        updated_self_attention_caches = []
        updated_cross_attention_caches = []
        for i, layer in enumerate(self.backbone.decoder_layers):
            layer_self_cache = (
                self_attention_cache[:, i, ...]
                if self_attention_cache is not None
                else None
            )
            layer_cross_cache = (
                cross_attention_cache[:, i, ...]
                if cross_attention_cache is not None
                else None
            )
            layer_cache = (layer_self_cache, layer_cross_cache)
            hidden_states, updated_layer_cache = layer(
                (hidden_states, encoder_output),
                self_attention_padding_mask=decoder_padding_mask,
                cross_attention_padding_mask=encoder_padding_mask,
                cache=layer_cache,
                cache_update_index=cache_update_index,
                training=False,
            )
            new_self_cache, new_cross_cache = updated_layer_cache
            updated_self_attention_caches.append(new_self_cache)
            updated_cross_attention_caches.append(new_cross_cache)
        self_attention_cache = ops.stack(updated_self_attention_caches, axis=1)
        cross_attention_cache = ops.stack(
            updated_cross_attention_caches, axis=1
        )
        hidden_states = self.backbone.decoder_norm(hidden_states)
        logits = self.backbone.decoder_token_embedding(
            hidden_states, reverse=True
        )
        if self.backbone.final_logit_softcapping is not None:
            logits = logits / self.backbone.final_logit_softcapping
            logits = ops.tanh(logits)
            logits = logits * self.backbone.final_logit_softcapping
        return (
            logits,
            hidden_states,
            (self_attention_cache, cross_attention_cache),
        )

    def _build_cache(
        self,
        encoder_token_ids,
        encoder_padding_mask,
        decoder_token_ids,
        decoder_padding_mask,
        images=None,
        vision_indices=None,
    ):
        """Build an empty cache for use with `call_with_cache()`."""
        encoder_output, encoder_padding_mask = self.call_encoder(
            encoder_token_ids,
            encoder_padding_mask,
            images=images,
            vision_indices=vision_indices,
        )
        batch_size = ops.shape(decoder_token_ids)[0]
        num_layers = self.backbone.decoder_num_layers
        num_kv_heads = self.backbone.decoder_num_key_value_heads
        head_dim = self.backbone.decoder_head_dim
        self_cache_shape = (
            batch_size,
            num_layers,
            2,
            ops.shape(decoder_token_ids)[1],
            num_kv_heads,
            head_dim,
        )
        self_attention_cache = ops.zeros(
            self_cache_shape, dtype=self.compute_dtype
        )
        cross_attention_cache = None
        _, hidden_states, cache = self.call_decoder_with_cache(
            decoder_token_ids=decoder_token_ids,
            decoder_padding_mask=decoder_padding_mask,
            cache=(self_attention_cache, cross_attention_cache),
            cache_update_index=0,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding_mask,
        )
        extra_cache_info = (encoder_output, encoder_padding_mask)
        return hidden_states, cache, extra_cache_info

    def generate_step(self, inputs, stop_token_ids=None):
        """A compilable generation function for a single batch.

        Args:
            inputs: A dictionary with keys `"encoder_token_ids"`,
                `"encoder_padding_mask"`, `"decoder_token_ids"`, and
                `"decoder_padding_mask"`.
            stop_token_ids: Tuple of end token ids to stop on.
        """
        encoder_token_ids = inputs["encoder_token_ids"]
        encoder_padding_mask = inputs["encoder_padding_mask"]
        decoder_token_ids = inputs["decoder_token_ids"]
        decoder_padding_mask = inputs["decoder_padding_mask"]
        images = inputs.get("images", None)
        vision_indices = inputs.get("vision_indices", None)
        hidden_states, cache, extra_cache_info = self._build_cache(
            encoder_token_ids=encoder_token_ids,
            encoder_padding_mask=encoder_padding_mask,
            decoder_token_ids=decoder_token_ids,
            decoder_padding_mask=decoder_padding_mask,
            images=images,
            vision_indices=vision_indices,
        )
        encoder_output, encoder_padding_mask = extra_cache_info
        row_lengths = ops.sum(ops.cast(decoder_padding_mask, "int32"), axis=-1)
        index = ops.min(row_lengths)

        def next(prompt, cache, index):
            cache_update_index = index - 1
            batch_size = ops.shape(prompt)[0]
            prompt = ops.slice(prompt, [0, cache_update_index], [batch_size, 1])
            (
                logits,
                _,
                updated_cache,
            ) = self.call_decoder_with_cache(
                decoder_token_ids=prompt,
                decoder_padding_mask=None,
                cache_update_index=cache_update_index,
                cache=cache,
                encoder_output=encoder_output,
                encoder_padding_mask=encoder_padding_mask,
            )
            return (
                ops.squeeze(logits, axis=1),
                None,
                updated_cache,
            )

        decoder_token_ids = self.sampler(
            next=next,
            prompt=decoder_token_ids,
            cache=cache,
            index=index,
            mask=decoder_padding_mask,
            stop_token_ids=stop_token_ids,
            hidden_states=hidden_states,
            model=self,
        )

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
