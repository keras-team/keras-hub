from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.muse_glimmer.muse_glimmer_backbone import (
    MuseGlimmerBackbone,
)


@keras_hub_export("keras_hub.models.MuseGlimmerAssistantCausalLM")
class MuseGlimmerAssistantCausalLM(CausalLM):
    """A DFlash block-diffusion drafter for speculative decoding.

    Drafts an entire `block_size`-token block of denoised hidden states in
    one forward pass, conditioned on hidden states pulled from several
    layers of the separate, larger MuseGlimmer target model. The target
    model then verifies the block in parallel. Origin: DFlash
    (arXiv:2602.06036).

    Unlike a standard causal-LM drafter, this model has no vocabulary and
    no LM head. It predicts continuous denoised block hidden states, not
    discrete tokens. It must NOT be used standalone — `call()` and
    `generate_step()` both raise `NotImplementedError`. Pass this model
    as `assistant_model` to `keras_hub.models.MuseGlimmerCausalLM.generate()`
    for end-to-end speculative decoding — that path calls
    `call_with_cache()`, which persists the context stream across
    drafting cycles (matches `modeling_muse_glimmer_assistant.py`'s real
    `forward()`, which is likewise always called with a real
    `DFlashCache` from the actual speculative-decoding driver,
    `DFlashTokenCandidateGenerator.get_candidates()` — its cache-free
    mode exists on the base class but is never exercised in practice).

    Args:
        backbone: A `keras_hub.models.MuseGlimmerBackbone` configured with
            the drafter's own dimensions and
            `use_bidirectional_attention=True`,
            `context_projection_layer_ids` set to the target model's
            `target_layer_ids`, `use_external_embeddings=True`,
            `enable_qk_scale_and_gate=False`, and
            `use_sandwich_norm=False`.
        block_size: int. Number of denoised-block positions drafted per
            forward pass. Defaults to `16`.
        mask_token_id: int. Token id used to fill the noise/mask positions
            of the block, one anchor token followed by `block_size - 1`
            copies of this id, before embedding via the target model's own
            token embedding table. Required for speculative decoding via
            `MuseGlimmerCausalLM.generate(assistant_model=...)`.
    """

    backbone_cls = MuseGlimmerBackbone

    def __init__(
        self,
        backbone,
        block_size=16,
        mask_token_id=None,
        **kwargs,
    ):
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.backbone = backbone

        inputs = backbone.input
        outputs = backbone(inputs=inputs)

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            backbone=backbone,
            **kwargs,
        )

    def call(self, *args, **kwargs):
        raise NotImplementedError(
            "MuseGlimmerAssistantCausalLM is a DFlash block-diffusion drafter "
            "that requires the target model's context and cannot be "
            "called standalone. Pass it as `assistant_model` to "
            "MuseGlimmerCausalLM.generate() instead."
        )

    def generate_step(self, *args, **kwargs):
        raise NotImplementedError(
            "MuseGlimmerAssistantCausalLM is a DFlash block-diffusion drafter "
            "that requires the target model's context and cannot be "
            "called standalone. Pass it as `assistant_model` to "
            "MuseGlimmerCausalLM.generate() instead."
        )

    def call_with_cache(
        self,
        noise_embeds,
        context_hidden_states,
        cache,
        cache_update_index,
        padding_mask=None,
    ):
        """Cached forward pass for one drafting cycle, for speculative
        decoding.

        Persists the context stream in `cache` across drafting cycles —
        see `MuseGlimmerTextAttention`'s docstring — so later cycles
        reuse previously-computed context key/value instead of
        recomputing them every time. This is DFlash's real caching
        benefit: only the context stream (one new real target position
        per cycle) is cached; the noise block is always freshly
        computed, since it's discarded/replaced every cycle regardless
        of accept/reject. Used by `MuseGlimmerCausalLM.generate_step()`'s
        speculative-decoding path.

        Args:
            noise_embeds: float tensor `(batch, block_size, hidden_dim)`.
                The noise-embedding window for the block being drafted.
            context_hidden_states: float tensor
                `(batch, 1, len(target_layer_ids) * target_hidden_dim)`.
                The single new target position to write into the cache
                this cycle.
            cache: float tensor
                `(batch, num_layers, 2, max_length, num_key_value_heads,
                head_dim)`. Persistent per-layer context key/value cache.
            cache_update_index: int or int tensor. Absolute
                target-sequence position of `context_hidden_states`.
            padding_mask: optional int tensor `(batch, block_size)`.
                Defaults to all ones (no padding).

        Returns:
            `(hidden_states, next_cache)`: the drafted block's denoised
            hidden states, `(batch, block_size, hidden_dim)`, and the
            updated cache.
        """
        backbone = self.backbone
        noise_embeds = ops.convert_to_tensor(noise_embeds)
        context_hidden_states = ops.convert_to_tensor(context_hidden_states)
        if padding_mask is None:
            padding_mask = ops.ones(ops.shape(noise_embeds)[:2], dtype="int32")
        else:
            padding_mask = ops.convert_to_tensor(padding_mask)

        projected_context = backbone.context_projection(context_hidden_states)

        x = noise_embeds
        next_cache = []
        for i in range(backbone.num_layers):
            layer = backbone.transformer_layers[i]
            x, layer_cache = layer(
                x,
                context_hidden_states=projected_context,
                decoder_padding_mask=padding_mask,
                self_attention_cache=cache[:, i, ...],
                self_attention_cache_update_index=cache_update_index,
            )
            next_cache.append(layer_cache)
        next_cache = ops.stack(next_cache, axis=1)
        hidden_states = backbone.layer_norm(x)
        return hidden_states, next_cache

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "block_size": self.block_size,
                "mask_token_id": self.mask_token_id,
            }
        )
        return config
