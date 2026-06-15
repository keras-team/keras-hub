import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.llama.llama_backbone import LlamaBackbone


@keras_hub_export("keras_hub.models.BLIP2Vicuna")
class BLIP2Vicuna(keras.Model):
    """Vicuna (LLaMA) language-model adapter for InstructBLIP.

    Wraps ``LlamaBackbone`` so it accepts BLIP-2's input format as a functional
    ``keras.Model``, mirroring `BLIP2CustomOPT`. The projected Q-Former visual
    query embeddings are prepended to the text token embeddings as a soft
    prompt; LLaMA's rotary position embeddings then index the concatenated
    ``[visual; text]`` sequence, so — unlike the OPT adapter — there are **no**
    learned position embeddings to add (`uses_learned_positions=False`).

    The model returns the final decoder hidden states. Logits are produced by
    `BLIP2CausalLM` via the tied token embedding (`reverse=True`), matching
    InstructBLIP-Vicuna (`tie_word_embeddings=False` keeps a separate output
    embedding inside the LLaMA backbone).

    Args:
        vocabulary_size: int. Token vocabulary size of the LLaMA tokenizer.
        num_layers: int. Number of transformer decoder layers.
        num_query_heads: int. Number of query attention heads.
        num_key_value_heads: int. Number of key/value heads (equal to the query
            heads for Vicuna-7B, i.e. no grouped-query attention).
        hidden_dim: int. Transformer hidden size.
        intermediate_dim: int. FFN intermediate (gate/up) dimension.
        num_query_tokens: int. Number of Q-Former visual query tokens prepended.
            Pass ``0`` for text-only mode.
        qformer_hidden_dim: int. Q-Former output dimension.
        rope_max_wavelength: int. RoPE maximum wavelength. Defaults to
            ``10000``.
        rope_position_scaling_factor: float. RoPE position scaling. Defaults to
            ``1.0``.
        layer_norm_epsilon: float. Epsilon for the RMSNorm layers.
        language_projection: `keras.layers.Layer` or None. Projects Q-Former
            features ``qformer_hidden_dim`` -> ``hidden_dim``. A ``Dense`` layer
            is created when ``None``.
        dtype: dtype for model weights and compute.
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_query_heads,
        num_key_value_heads,
        hidden_dim,
        intermediate_dim,
        num_query_tokens,
        qformer_hidden_dim,
        rope_max_wavelength=10000,
        rope_position_scaling_factor=1.0,
        layer_norm_epsilon=1e-6,
        language_projection=None,
        dtype=None,
        name=None,
        **kwargs,
    ):
        llama = LlamaBackbone(
            vocabulary_size=vocabulary_size,
            num_layers=num_layers,
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            rope_max_wavelength=rope_max_wavelength,
            rope_position_scaling_factor=rope_position_scaling_factor,
            layer_norm_epsilon=layer_norm_epsilon,
            tie_word_embeddings=False,
            dtype=dtype,
            name="llama",
        )

        if language_projection is None:
            language_projection = keras.layers.Dense(
                hidden_dim,
                use_bias=True,
                dtype=dtype,
                name="language_projection",
            )

        token_ids_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        inputs = {
            "token_ids": token_ids_input,
            "padding_mask": padding_mask_input,
        }

        if num_query_tokens > 0:
            qformer_features_input = keras.Input(
                shape=(num_query_tokens, qformer_hidden_dim),
                name="qformer_features",
            )
            inputs["qformer_features"] = qformer_features_input

            projected_qf = language_projection(qformer_features_input)
            token_embeds = llama.token_embedding(token_ids_input)
            x = ops.concatenate([projected_qf, token_embeds], axis=1)

            vis_mask = ops.cast(
                ops.ones_like(qformer_features_input[..., 0]),
                padding_mask_input.dtype,
            )
            full_padding_mask = ops.concatenate(
                [vis_mask, padding_mask_input], axis=1
            )
        else:
            x = llama.token_embedding(token_ids_input)
            full_padding_mask = padding_mask_input

        for layer in llama.transformer_layers:
            x = layer(x, decoder_padding_mask=full_padding_mask)
        outputs = llama.layer_norm(x)

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            dtype=dtype,
            name=name,
            **kwargs,
        )

        self.llama = llama
        self.language_projection = language_projection

        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_query_tokens = num_query_tokens
        self.qformer_hidden_dim = qformer_hidden_dim
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_position_scaling_factor = rope_position_scaling_factor
        self.layer_norm_epsilon = layer_norm_epsilon

        # LLaMA uses rotary positions, not learned ones; the causal-LM
        # generation glue branches on this flag.
        self.uses_learned_positions = False
        # Cache geometry consumed by `BLIP2CausalLM._build_cache`. Vicuna-7B has
        # no grouped-query attention, so kv-heads == query-heads.
        self.num_heads = num_key_value_heads
        self.head_dim = hidden_dim // num_query_heads

    def call_with_cache(self, x, padding_mask, cache, cache_update_index):
        """LLaMA forward pass over precomputed embeddings with a KV cache."""
        updated_caches = []
        for i, layer in enumerate(self.transformer_layers):
            x, new_layer_cache = layer(
                x,
                decoder_padding_mask=padding_mask,
                self_attention_cache=cache[:, i],
                self_attention_cache_update_index=cache_update_index,
            )
            updated_caches.append(new_layer_cache)
        new_cache = ops.stack(updated_caches, axis=1)
        x = self.layer_norm(x)
        return x, new_cache

    @property
    def token_embedding(self):
        return self.llama.token_embedding

    @property
    def transformer_layers(self):
        return self.llama.transformer_layers

    @property
    def layer_norm(self):
        return self.llama.layer_norm

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_query_tokens": self.num_query_tokens,
                "qformer_hidden_dim": self.qformer_hidden_dim,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_position_scaling_factor": (
                    self.rope_position_scaling_factor
                ),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "language_projection": keras.layers.serialize(
                    self.language_projection
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        if config.get("language_projection") is not None:
            config["language_projection"] = keras.layers.deserialize(
                config["language_projection"]
            )
        return cls(**config)
