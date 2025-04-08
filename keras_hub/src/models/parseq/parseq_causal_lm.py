from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.parseq.parseq_backbone import PARSeqBackbone
from keras_hub.src.models.parseq.parseq_preprocessor import PARSeqPreprocessor


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
        bs, tokens_length = ops.shape(token_ids)
        # <bos> stands for the null context. We only supply position information
        # for characters after <bos>.
        null_context = (
            self.backbone.decoder.hidden_dim** 0.5
            * self.backbone.decoder.token_embedding(
                ops.slice(token_ids, [0, 0], [bs, 1])
            )
        )
        content = self.backbone.decoder.pos_query_embeddings[
            :, : tokens_length - 1, :
        ]
        content = (
            content
            + self.backbone.decoder.hidden_dim** 0.5
            * self.backbone.decoder.token_embedding(
                ops.slice(token_ids, [0, 1], [bs, tokens_length - 1])
            )
        )
        content = ops.concatenate([null_context, content], axis=1)
        content = self.dropout(content)
        query = (
            ops.ones((bs, 1, 1))
            * self.backbone.decoder.pos_query_embeddings[:, :tokens_length, :]
        )
        query = self.dropout(query)
        for i, decoder_layer in enumerate(self.backbone.decoder.decoder_layers):
            last = i == self.backbone.decoder.num_layers - 1
            query, content = decoder_layer(
                query=query,
                content=content,
                memory=img_embeddings,
                padding_mask=padding_mask,
                update_content=not last,
            )

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
