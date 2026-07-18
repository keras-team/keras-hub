"""HRM-Text backbone."""

import keras
from keras import ops
from keras.layers import ReversibleEmbedding

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.hrm_text.hrm_text_layers import HrmTextAttentionMask
from keras_hub.src.models.hrm_text.hrm_text_layers import HrmTextInitialState
from keras_hub.src.models.hrm_text.hrm_text_layers import HrmTextStack


def make_hrm_text_attention_mask(token_type_ids, padding_mask):
    """Builds HRM-Text's causal/PrefixLM boolean attention mask."""
    sequence_length = ops.shape(token_type_ids)[1]
    positions = ops.arange(sequence_length)
    causal = positions[None, :] <= positions[:, None]
    prefix = ops.cast(token_type_ids == 1, "bool")
    prefix_mask = ops.logical_and(prefix[:, :, None], prefix[:, None, :])
    allowed = ops.logical_or(causal[None, :, :], prefix_mask)
    valid = ops.cast(padding_mask, "bool")
    valid_mask = ops.logical_and(valid[:, :, None], valid[:, None, :])
    return ops.logical_and(allowed, valid_mask)


def make_hrm_text_cache_mask(
    token_ids, cache_update_index, cache_length, token_type_ids=None
):
    """Builds a prefill PrefixLM mask or a causal decode mask."""
    batch_size = ops.shape(token_ids)[0]
    sequence_length = ops.shape(token_ids)[1]
    # Keep `arange()` bounds static for JAX's compiled sampler loop. The cache
    # offset is dynamic during decoding, so add it after creating the range.
    query_positions = (
        ops.arange(sequence_length, dtype="int32") + cache_update_index
    )[:, None]
    key_positions = ops.arange(cache_length, dtype="int32")[None, :]
    causal = key_positions <= query_positions
    causal = ops.broadcast_to(
        causal[None, :, :],
        (batch_size, sequence_length, cache_length),
    )
    if token_type_ids is None:
        return causal
    prefix = ops.cast(token_type_ids == 1, "bool")
    prefix_padding = ops.zeros(
        (batch_size, cache_length - sequence_length), dtype="bool"
    )
    prefix_keys = ops.concatenate((prefix, prefix_padding), axis=1)
    return ops.logical_or(
        causal,
        ops.logical_and(prefix[:, :, None], prefix_keys[:, None, :]),
    )


@keras_hub_export("keras_hub.models.HrmTextBackbone")
class HrmTextBackbone(Backbone):
    """HRM-Text hierarchical recurrent decoder backbone.

    HRM-Text uses two shared Transformer stacks rather than a single sequence
    of independent decoder layers. In every forward pass, a fast low-level
    (L) stack runs ``l_cycles`` times for each high-level (H) stack update.
    Each stack uses parameterless pre-RMS normalization, RoPE, gated
    multi-head attention, and SwiGLU. The L and H states are reinitialized for
    each model call; autoregressive generation instead caches the key/value
    tensors for every logical stack invocation.

    Inputs are a dictionary with ``token_ids``, ``padding_mask``, and
    ``token_type_ids``. Set ``token_type_ids`` to zero for ordinary causal
    attention. Positions marked one form a bidirectional PrefixLM prefix;
    later positions remain causal.

    Args:
        vocabulary_size: Number of tokens in the vocabulary.
        hidden_dim: Width of the token embeddings and Transformer states.
        intermediate_dim: Width of the SwiGLU intermediate projection.
        num_layers_per_stack: Number of shared decoder blocks in each H and L
            stack.
        num_attention_heads: Number of attention heads.
        head_dim: Dimension of each attention head.
        h_cycles: Number of high-level recurrent cycles. Defaults to ``2``.
        l_cycles: Number of low-level updates within each high-level cycle.
            Defaults to ``3``.
        max_sequence_length: Maximum sequence length used by the preset.
            Defaults to ``4096``.
        rope_theta: Rotary-position-embedding base. Defaults to ``10000.0``.
        rms_norm_epsilon: Epsilon used by RMS normalization. Defaults to
            ``1e-6``.
        embedding_scale: Scale applied to token embeddings before recurrence.
            Defaults to ``1.0``.
        tie_word_embeddings: Whether input and output embeddings are tied.
            Defaults to ``False``.
        dtype: Dtype policy for the backbone.

    Examples:

    ```python
    backbone = keras_hub.models.HrmTextBackbone(
        vocabulary_size=128,
        hidden_dim=64,
        intermediate_dim=128,
        num_layers_per_stack=2,
        num_attention_heads=4,
        head_dim=16,
        h_cycles=2,
        l_cycles=2,
        max_sequence_length=32,
    )
    inputs = {
        "token_ids": np.array([[5, 9, 17, 0]], dtype="int32"),
        "padding_mask": np.array([[1, 1, 1, 0]], dtype="int32"),
        "token_type_ids": np.zeros((1, 4), dtype="int32"),
    }
    hidden_states = backbone(inputs)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        hidden_dim,
        intermediate_dim,
        num_layers_per_stack,
        num_attention_heads,
        head_dim,
        h_cycles=2,
        l_cycles=3,
        max_sequence_length=4096,
        rope_theta=10000.0,
        rms_norm_epsilon=1e-6,
        embedding_scale=1.0,
        tie_word_embeddings=False,
        dtype=None,
        **kwargs,
    ):
        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers_per_stack = num_layers_per_stack
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.h_cycles = h_cycles
        self.l_cycles = l_cycles
        self.max_sequence_length = max_sequence_length
        self.rope_theta = rope_theta
        self.rms_norm_epsilon = rms_norm_epsilon
        self.embedding_scale = embedding_scale
        self.tie_word_embeddings = tie_word_embeddings
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=tie_word_embeddings,
            name="token_embedding",
            dtype=dtype,
        )
        block_kwargs = {
            "num_heads": num_attention_heads,
            "head_dim": head_dim,
            "intermediate_dim": intermediate_dim,
            "rope_theta": rope_theta,
            "rms_norm_epsilon": rms_norm_epsilon,
        }
        self.L_module = HrmTextStack(
            num_layers_per_stack, name="L_module", dtype=dtype, **block_kwargs
        )
        self.H_module = HrmTextStack(
            num_layers_per_stack, name="H_module", dtype=dtype, **block_kwargs
        )
        self.initial_state = HrmTextInitialState(
            hidden_dim, name="initial_state", dtype=dtype
        )
        self.attention_mask = HrmTextAttentionMask(
            name="attention_mask", dtype=dtype
        )

        token_ids = keras.Input(shape=(None,), dtype="int32", name="token_ids")
        padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        token_type_ids = keras.Input(
            shape=(None,), dtype="int32", name="token_type_ids"
        )
        hidden_states = self._forward(token_ids, padding_mask, token_type_ids)
        super().__init__(
            inputs={
                "token_ids": token_ids,
                "padding_mask": padding_mask,
                "token_type_ids": token_type_ids,
            },
            outputs=hidden_states,
            dtype=dtype,
            **kwargs,
        )

    @property
    def cache_slots(self):
        return self.num_layers_per_stack * self.h_cycles * (self.l_cycles + 1)

    def _forward(self, token_ids, padding_mask, token_type_ids):
        attention_mask = self.attention_mask(token_type_ids, padding_mask)
        high = self.token_embedding(token_ids) * self.embedding_scale
        low = self.initial_state(high)
        for high_cycle in range(self.h_cycles):
            for low_cycle in range(self.l_cycles):
                offset = (
                    high_cycle * (self.l_cycles + 1) + low_cycle
                ) * self.num_layers_per_stack
                low = self.L_module(
                    low + high, attention_mask, cycle_offset=offset
                )
            offset = (
                high_cycle * (self.l_cycles + 1) + self.l_cycles
            ) * self.num_layers_per_stack
            high = self.H_module(
                high + low, attention_mask, cycle_offset=offset
            )
        return high

    def call_with_cache(
        self, token_ids, cache, cache_update_index, token_type_ids=None
    ):
        """Runs HRM-Text with one KV cache per recurrent invocation.

        The official 1B configuration has 16 layers in each stack, two H
        cycles, and three L cycles, so it needs ``16 * 2 * (3 + 1) == 128``
        logical cache slots. These slots correspond to recurrent invocations,
        not to the 32 parameterized Transformer blocks.
        """
        attention_mask = make_hrm_text_cache_mask(
            token_ids,
            cache_update_index,
            ops.shape(cache)[3],
            token_type_ids,
        )
        high = self.token_embedding(token_ids) * self.embedding_scale
        low = self.initial_state(high)
        for high_cycle in range(self.h_cycles):
            for low_cycle in range(self.l_cycles):
                offset = (
                    high_cycle * (self.l_cycles + 1) + low_cycle
                ) * self.num_layers_per_stack
                low, updates = self.L_module(
                    low + high,
                    attention_mask,
                    cache=cache,
                    cache_update_index=cache_update_index,
                    cycle_offset=offset,
                )
                cache = ops.slice_update(
                    cache,
                    [0, offset, 0, 0, 0, 0],
                    ops.stack(updates, axis=1),
                )
            offset = (
                high_cycle * (self.l_cycles + 1) + self.l_cycles
            ) * self.num_layers_per_stack
            high, updates = self.H_module(
                high + low,
                attention_mask,
                cache=cache,
                cache_update_index=cache_update_index,
                cycle_offset=offset,
            )
            cache = ops.slice_update(
                cache,
                [0, offset, 0, 0, 0, 0],
                ops.stack(updates, axis=1),
            )
        return high, cache

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_layers_per_stack": self.num_layers_per_stack,
                "num_attention_heads": self.num_attention_heads,
                "head_dim": self.head_dim,
                "h_cycles": self.h_cycles,
                "l_cycles": self.l_cycles,
                "max_sequence_length": self.max_sequence_length,
                "rope_theta": self.rope_theta,
                "rms_norm_epsilon": self.rms_norm_epsilon,
                "embedding_scale": self.embedding_scale,
                "tie_word_embeddings": self.tie_word_embeddings,
            }
        )
        return config
