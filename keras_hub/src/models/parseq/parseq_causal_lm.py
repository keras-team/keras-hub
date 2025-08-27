import math

import keras
from keras import ops
from keras import random

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.parseq.parseq_backbone import PARSeqBackbone
from keras_hub.src.models.parseq.parseq_causal_lm_preprocessor import (
    PARSeqCausalLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.PARSeqCausalLM")
class PARSeqCausalLM(CausalLM):
    """Scene Text Recognition with PARSeq.
    Performs OCR in natural scenes using the PARSeq model described in
    [Scene Text Recognition with Permuted Autoregressive Sequence Models](
    https://arxiv.org/abs/2207.06966). PARSeq is a ViT-based model that allows
    iterative decoding by performing an autoregressive decoding phase, followed
    by a refinement phase.
    Args:
        preprocessor: A `keras_hub.models.Preprocessor` instance or a
            `keras.Layer` instance. The preprocessor to use for the model.
        backbone: A `keras_hub.models.PARSeqBackbone` instance or a
            `keras.Model`. The backbone model to use for the model.
        num_perms: int. The number of permutations to generate for training.
            Defaults to 6.
        add_forward_perms: bool. Whether to add forward permutations to the
            generated permutations. Defaults to `True`.
        add_mirrored_perms: bool. Whether to add mirrored permutations to the
            generated permutations. Defaults to `True`.
        seed: int. The random seed to use for generating permutations.
            Defaults to `None`, which means no seed is set.
        **kwargs: Additional keyword arguments passed to the base
            `keras_hub.models.CausalLM` constructor.

    Examples:

    Call `predict()` to run inference.
    ```python
    # Load preset and run inference
    images = np.random.randint(0, 256, size=(2, 32, 128, 3))
    parseq = keras_hub.models.PARSeqCausalLM.from_preset(
        "parseq_vit"
    )
    parseq.generate(images)

    # Call `fit()` on a single batch.
    images = np.random.randint(0, 256, size=(2, 32, 128, 3))
    token_ids = np.array([[1, 2, 3, 4], [1, 2, 3, 0]])
    padding_mask = np.array([[1, 1, 1, 1], [1, 1, 1, 0]])
    parseq = keras_hub.models.PARSeqCausalLM.from_preset(
        "parseq_vit"
    )
    parseq.fit(
        x={
            "images": images,
            "token_ids": token_ids,
            "padding_mask": padding_mask
        },
        batch_size=2,
    )
    ```
    # Call `fit()` with custom loss, optimizer and image encoder.
    ```python
    # Initialize the image encoder, preprocessor and tokenizer
    mean, std = 0.5, 0.5
    image_converter = PARSeqImageConverter(
        image_size=(32, 128),
        offset=-mean / std,
        scale=1.0 / 255.0 / std,
        interpolation="bicubic",
    )
    tokenizer = PARSeqTokenizer(max_label_length=25)
    preprocessor = keras_hub.models.PARSeqCausalLMPreprocessor(
        image_converter=image_converter,
        tokenizer=tokenizer,
    )

    # Create the backbone
    image_encoder = ViTBackbone(
        image_shape=(32, 128, 3),
        patch_size=(4, 8),
        num_layers=12,
        num_heads=6,
        hidden_dim=384,
        mlp_dim=384 * 4,
        use_class_token=False,
        name="encoder",
    )
    backbone = PARSeqBackbone(
        vocabulary_size=97,
        max_label_length=25,
        image_encoder=image_encoder,
        num_decoder_heads=12,
        num_decoder_layers=1,
        decoder_hidden_dim=384,
        decoder_mlp_dim=4 * 384,
    )
    # Create the PARSeq model
    parseq = keras_hub.models.PARSeqCausalLM(
        backbone=backbone,
        preprocessor=preprocessor,
    )
    parseq.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(5e-5),
    )
    parseq.fit(
        x={
            "images": images,
            "token_ids": token_ids,
            "padding_mask": padding_mask
        },
        batch_size=2,
    )
    ```
    """

    backbone_cls = PARSeqBackbone
    preprocessor_cls = PARSeqCausalLMPreprocessor

    def __init__(
        self,
        preprocessor,
        backbone,
        num_perms=6,
        add_forward_perms=True,
        add_mirrored_perms=True,
        seed=None,
        end_token_id=0,  # default tokenizer.end_token_id
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
        self.end_token_id = end_token_id
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
                "end_token_id": self.end_token_id,
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
                from_logits=True,
                ignore_class=self.preprocessor.tokenizer.pad_token_id,
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
        max_label_length = self.backbone.max_label_length
        memory = self.backbone.image_encoder(x["images"])
        batch_size = ops.shape(x["images"])[0]
        losses = []
        for i in range(ops.shape(perms)[0]):
            query_mask, content_mask = self.generate_attention_masks(perms[i])
            query_mask = ops.broadcast_to(
                query_mask, (batch_size, max_label_length, max_label_length)
            )
            content_mask = ops.broadcast_to(
                content_mask, (batch_size, max_label_length, max_label_length)
            )
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
            if i == 1:
                # Sample weights are set to zero for end-of-sequence (EOS)
                # tokens to prevent them from affecting loss calculations.
                # reference: https://github.com/baudm/parseq/blob/1902db043c029a7e03a3818c616c06600af574be/strhub/models/parseq/system.py#L194 # noqa: E501
                sample_weight = ops.logical_and(
                    y != self.end_token_id, sample_weight
                )

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
        """Generate attention masks given a sequence permutation
        (includes pos. for BOS and EOS tokens)"""
        input_length = ops.shape(perm)[0]
        mask = ops.ones((input_length, input_length))
        for i in range(input_length - 1):
            masked_keys = perm[i + 1 : input_length]
            query_idx = ops.broadcast_to(perm[i], ops.shape(masked_keys))
            indices = ops.stack((query_idx, masked_keys), axis=1)
            mask = keras.ops.scatter_update(
                mask, indices, keras.ops.zeros(ops.shape(masked_keys)[0])
            )
        content_mask = mask[:-1, :-1]
        mask = mask * (1 - ops.eye(input_length))
        query_mask = mask[1:, :-1]
        return query_mask, content_mask

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
        head_dim = (
            self.backbone.decoder_hidden_dim // self.backbone.num_decoder_heads
        )
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
