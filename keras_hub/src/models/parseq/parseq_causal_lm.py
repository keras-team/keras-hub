from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.parseq.parseq_backbone import PARSeqBackbone
from keras_hub.src.models.parseq.parseq_preprocessor import PARSeqPreprocessor
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.ParSeqCausalLM")
class ParSeqCausalLM(CausalLM):
    backbone_cls = PARSeqBackbone
    preprocessor_cls = PARSeqPreprocessor

    def __init__(
        self,
        preprocessor,
        backbone,
        **kwargs,
    ):
        # === Layers ===
        self.preprocessor = preprocessor
        self.backbone = backbone

        # === Functional Model ===
        # This must be "backbone.input" i.e. the full input structure,
        # rather than "backbone.inputs" which is the flattened list of inputs.
        inputs = backbone.input
        outputs = backbone(inputs=inputs)
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

    def compile(
        self,
        optimizer="auto",
        loss="auto",
        *,
        weighted_metrics="auto",
        sampler="greedy",
        **kwargs,
    ):
        super().compile(
            optimizer=optimizer,
            loss=loss,
            weighted_metrics=weighted_metrics,
            sampler=sampler,
            **kwargs,
        )

    def call_with_cache(
        self,
        token_ids,
        cache,
        cache_update_index,
        img_embeddings,
        padding_mask=None,
    ):
        bs = ops.shape(token_ids)[0]
        # <bos> stands for the null context. We only supply position information
        # for characters after <bos>.
        content = ops.where(
            cache_update_index == 0,
            self.backbone.decoder_hidden_dim**0.5
            * self.backbone.decoder.token_embedding(token_ids),
            ops.expand_dims(
                self.backbone.decoder.pos_query_embeddings[
                    :, cache_update_index - 1, :
                ],
                axis=0,
            )
            + self.backbone.decoder_hidden_dim**0.5
            * self.backbone.decoder.token_embedding(token_ids),
        )
        content = self.backbone.decoder.dropout(content)

        query = ops.ones((bs, 1, 1)) * ops.expand_dims(
            self.backbone.decoder.pos_query_embeddings[
                :, cache_update_index, :
            ],
            axis=0,
        )
        query = self.backbone.decoder.dropout(query)

        query_cache = []
        content_cache = []
        for i, decoder_layer in enumerate(self.backbone.decoder.decoder_layers):
            last = i == self.backbone.num_decoder_layers - 1
            current_query_cache = cache[:, i, 0, ...]
            current_content_cache = cache[:, i, 1, ...]
            (
                query,
                content,
                query_self_attention_new_cache,
                content_self_attention_cache,
            ) = decoder_layer(
                query=query,
                content=content,
                memory=img_embeddings,
                padding_mask=padding_mask,
                update_content=not last,
                query_self_attention_cache=current_query_cache,
                query_self_attention_cache_update_index=cache_update_index,
                content_self_attention_cache=current_content_cache,
                content_self_attention_cache_update_index=cache_update_index,
            )
            query_cache.append(query_self_attention_new_cache)
            content_cache.append(content_self_attention_cache)

        query_cache = ops.stack(query_cache, axis=1)
        content_cache = ops.stack(content_cache, axis=1)
        cache = ops.stack([query_cache, content_cache], axis=2)
        hidden_states = self.backbone.decoder.layer_norm(query)
        logits = self.backbone.head(hidden_states)
        return logits, hidden_states, cache

    def _build_cache(self, token_ids, img_embeddings, padding_mask):
        batch_size = ops.shape(token_ids)[0]
        max_length = ops.shape(token_ids)[1]
        num_layers = self.backbone.num_decoder_layers
        head_dim = self.backbone.deocder_head_dim
        num_heads = self.backbone.num_decoder_heads
        shape = [batch_size, num_layers, 2, 2, max_length, num_heads, head_dim]
        cache = ops.zeros(shape)

        # Seed the cache.
        logits, hidden_states, cache = self.call_with_cache(
            token_ids=token_ids,
            img_embeddings=img_embeddings,
            cache=cache,
            cache_update_index=0,
            padding_mask=padding_mask,
        )
        return hidden_states, cache

    def generate_step(self, inputs, stop_token_ids=None):
        token_ids, padding_mask, images = (
            inputs["token_ids"],
            inputs["padding_mask"],
            inputs["images"],
        )
        images_shape = ops.shape(images)
        if len(images_shape) == 3:
            # Handle an unbatched image. Unlike `token_ids` and `padding_mask`
            # this will not automatically be upranked.
            images = ops.expand_dims(images, axis=0)

        img_embeddings = self.backbone.image_encoder(images)
        # Create and seed cache with a single forward pass.
        hidden_states, cache = self._build_cache(
            token_ids=token_ids,
            img_embeddings=img_embeddings,
            padding_mask=padding_mask,
        )

        # Create and seed cache with a single forward pass.
        hidden_states, cache = self._build_cache(
            token_ids, img_embeddings, padding_mask
        )
        # Compute the lengths of all user inputted tokens ids.
        row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
        # Start at the first index that has no user inputted id.
        index = ops.min(row_lengths)

        def next(prompt, cache, index):
            # The cache index is the index of our previous token.
            cache_update_index = index - 1
            batch_size = ops.shape(prompt)[0]
            prompt = ops.slice(prompt, [0, index - 1], [batch_size, 1])
            logits, hidden_states, cache = self.call_with_cache(
                token_ids=prompt,
                cache=cache,
                cache_update_index=cache_update_index,
                img_embeddings=img_embeddings,
            )
            return (
                ops.squeeze(logits, axis=1),
                ops.squeeze(hidden_states, axis=1),
                cache,
            )

        token_ids = self.sampler(
            next=next,
            prompt=token_ids,
            cache=cache,
            index=index,
            mask=padding_mask,
            stop_token_ids=stop_token_ids,
            hidden_states=hidden_states,
            model=self,
        )

        # Compute an output padding mask with the token ids we updated.
        if stop_token_ids is not None:
            # Build a mask of `stop_token_ids` locations not in the original
            # prompt (not in locations where `padding_mask` is True).
            end_locations = any_equal(
                token_ids, stop_token_ids, ops.logical_not(padding_mask)
            )

            end_locations = ops.cast(end_locations, "int32")
            # Use cumsum to get ones in all locations after end_locations.
            cumsum = ops.cast(ops.cumsum(end_locations, axis=-1), "int32")
            overflow = cumsum - end_locations
            # Our padding mask is the inverse of these overflow locations.
            padding_mask = ops.logical_not(ops.cast(overflow, "bool"))
        else:
            # Without early stopping, all locations will have been updated.
            padding_mask = ops.ones_like(token_ids, dtype="bool")
        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "images": images,
        }
