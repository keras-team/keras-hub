import math

import keras
from keras import ops
from keras import random

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
        num_perms=6,
        add_forward_perms=True,
        add_mirrored_perms=True,
        seed=None,
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

        # === Config ===
        self.num_perms = num_perms
        self.add_forward_perms = add_forward_perms
        self.add_mirrored_perms = add_mirrored_perms
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_perms": self.num_perms,
                "add_forward_perms": self.add_forward_perms,
                "add_mirrored_perms": self.add_mirrored_perms,
                "seed": self.seed,
            }
        )

        return config

    def compile(
        self,
        optimizer="auto",
        loss="auto",
        *,
        weighted_metrics="auto",
        sampler="greedy",
        **kwargs,
    ):
        if loss == "auto":
            loss = keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction="mean_with_sample_weight"
            )
        super().compile(
            optimizer=optimizer,
            loss=loss,
            weighted_metrics=weighted_metrics,
            sampler=sampler,
            **kwargs,
        )

    def compute_loss(
        self, x, y, y_pred, sample_weight, training=True, *args, **kwargs
    ):
        # For keras we have fixed input for all batches, so in this case
        # we permute 23 tokens excluding BOS and EOS tokens instead of max
        # characters for current batch used in torch implementation
        # -1 because we will be generating permutation mask for considering
        # tokens before creating target label.
        max_num_chars = self.backbone.max_label_length - 1
        perms = self.generate_training_permutations(max_num_chars)
        memory = self.backbone.image_encoder(x["images"])
        losses = []
        for i in range(ops.shape(perms)[0]):
            query_mask, content_mask = self.generate_attention_masks(perms[i])
            out = self.backbone.decoder(
                x["token_ids"],
                memory,
                padding_mask=x["padding_mask"],
                query_mask=query_mask,
                content_mask=content_mask,
            )
            y_pred = self.backbone.head(out)
            loss = super().compute_loss(
                x=x, y=y, y_pred=y_pred, sample_weight=sample_weight, **kwargs
            )
            losses.append(loss)

        return ops.sum(losses) / ops.shape(perms)[0]

    def generate_training_permutations(self, max_num_chars):
        max_gen_perms = (
            self.num_perms // 2 if self.add_mirrored_perms else self.num_perms
        )

        if max_num_chars == 1:
            return ops.expand_dims(ops.arange(3), axis=0)

        perms = [ops.arange(max_num_chars)] if self.add_forward_perms else []
        max_num_perms = math.factorial(max_num_chars)
        max_gen_perms = min(max_gen_perms, max_num_perms)

        for _ in range(max_gen_perms - len(perms)):
            perm = random.shuffle(
                ops.arange(max_num_chars), seed=self.seed_generator
            )
            perms.append(perm)

        perms = ops.stack(perms)
        comp = ops.flip(perms, axis=-1)
        perms = ops.stack([perms, comp])
        perms = ops.reshape(
            ops.transpose(perms, (1, 0, 2)), (-1, max_num_chars)
        )

        bos_idx = ops.zeros((ops.shape(perms)[0], 1), dtype="int32")
        eos_idx = ops.full(
            (ops.shape(perms)[0], 1), max_num_chars + 1, dtype="int32"
        )
        perms = ops.concatenate([bos_idx, perms + 1, eos_idx], axis=1)

        if perms.shape[0] > 1:
            perms = ops.scatter_update(
                perms,
                ops.concatenate(
                    [
                        ops.ones((max_num_chars + 1, 1), dtype="int32"),
                        ops.expand_dims(
                            ops.arange(1, max_num_chars + 2, dtype="int32"),
                            axis=1,
                        ),
                    ],
                    axis=1,
                ),
                max_num_chars + 1 - ops.arange(max_num_chars + 1),
            )

        return perms

    def generate_attention_masks(self, perm):
        n = ops.shape(perm)[0]

        # i represents the row index (0 to n-1), needs shape (n, 1)
        i_coords = ops.expand_dims(ops.arange(n), axis=1)
        # j represents the column index (0 to n-1), needs shape (1, n)
        j_coords = ops.expand_dims(ops.arange(n), axis=0)

        lower_triangle_mask = j_coords <= i_coords

        # Find the (row, col) indices where the mask is True
        # ops.where returns a tuple of arrays: (i_indices, j_indices)
        i_idx, j_idx = ops.where(lower_triangle_mask)

        # Map these i, j indices through the permutation
        target_rows = perm[i_idx]
        target_cols = perm[j_idx]

        # Combine target_rows and target_cols into a single indices array
        content_indices = ops.stack([target_rows, target_cols], axis=1)
        mask = ops.zeros((n, n), dtype=bool)
        mask = ops.scatter_update(
            mask,
            content_indices,
            ops.ones(content_indices.shape[0], dtype=bool),
        )

        # mask "self"
        query_indices = ops.stack(
            [ops.squeeze(i_coords), ops.squeeze(j_coords)], axis=1
        )
        query_mask = ops.scatter_update(
            mask, query_indices, ops.zeros(query_indices.shape[0], dtype=bool)
        )[1:, :-1]

        return mask[:-1, :-1], query_mask

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
