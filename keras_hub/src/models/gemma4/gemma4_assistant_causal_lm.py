from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone


@keras_hub_export("keras_hub.models.Gemma4AssistantCausalLM")
class Gemma4AssistantCausalLM(CausalLM):
    """An MTP assistant model for speculative decoding with Gemma4.

    This model drafts candidate tokens to be verified by a larger target model.
    It conditions on the target model's last hidden state and last token
    embedding, and borrows the target model's KV cache via the KV-sharing
    mechanism in `Gemma4Backbone`.

    This model must NOT be used standalone. Each `call_with_cache()` step
    requires:
      - `last_hidden_state`: target model last-position hidden state,
        shape `(batch, 1, backbone_hidden_size)`.
      - `last_token_id`: last accepted token id, shape `(batch, 1)`.
      - `target_cache`: target model full KV cache tensor, shape
        `(batch, num_target_layers, 2, max_len, num_heads, head_dim)`.

    Args:
        backbone: A `keras_hub.models.Gemma4Backbone` configured with the
            assistant's own dimensions (e.g. `hidden_dim=256`, `num_layers=4`)
            and `num_kv_shared_layers=4`.
        backbone_hidden_size: int. Hidden dimension of the **target** model
            (e.g. `1536` for Gemma4 2B). Used to size the projection layers.
            Defaults to `1536`.
        num_centroids: int. Number of vocabulary centroids for the ordered
            embedding head. Defaults to `2048`.
        centroid_intermediate_top_k: int. Number of active centroids per
            forward pass. Defaults to `32`.
        use_ordered_embeddings: bool. Whether to use sparse centroid-based
            embedding routing for logits. Larger variants (e.g. 26B) disable
            this for a standard output head. Defaults to `True`.
        sampler: A `keras_hub.samplers.Sampler` instance or string. The
            sampling strategy. Defaults to `"greedy"`.
        num_speculative_tokens: int. Number of draft tokens proposed per
            speculative decoding step. Passed to `SpeculativeSampler` by
            the target model's `generate()`. Defaults to `5`.

    Examples:
    ```python
    import keras_hub

    target_lm = keras_hub.models.Gemma4CausalLM.from_preset(
        "gemma4_instruct_2b"
    )
    assistant_lm = keras_hub.models.Gemma4AssistantCausalLM.from_preset(
        "gemma4_instruct_2b_assistant"
    )
    response = target_lm.generate(
        "What is the capital of France?",
        assistant_model=assistant_lm,
        max_length=128,
    )
    ```
    """

    backbone_cls = Gemma4Backbone

    def __init__(
        self,
        backbone,
        backbone_hidden_size,
        num_centroids,
        centroid_intermediate_top_k,
        use_ordered_embeddings,
        num_speculative_tokens=5,
        sampler="greedy",
        **kwargs,
    ):
        self.backbone_hidden_size = backbone_hidden_size
        self.num_centroids = num_centroids
        self.centroid_intermediate_top_k = centroid_intermediate_top_k
        self.use_ordered_embeddings = use_ordered_embeddings
        self.num_speculative_tokens = num_speculative_tokens
        self.backbone = backbone

        hidden_size = backbone.hidden_dim
        vocabulary_size = backbone.token_embedding.input_dim
        if use_ordered_embeddings and vocabulary_size % num_centroids != 0:
            raise ValueError(
                f"`vocabulary_size` ({vocabulary_size}) must be divisible by "
                f"`num_centroids` ({num_centroids})."
            )
        self._vocab_size_per_centroid = vocabulary_size // num_centroids

        for layer in self.backbone.transformer_layers:
            layer.attention.is_kv_shared_layer = True

        # Build the functional model (used for weight loading / summary).
        inputs = backbone.input
        hidden_state = backbone(inputs=inputs)
        outputs = backbone.token_embedding(hidden_state, reverse=True)

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            sampler=sampler,
            backbone=backbone,
            **kwargs,
        )

        # Non-graph projection layers created AFTER super().__init__() so
        # Keras registers them as tracked sub-layers of this model.  They are
        # used only in call_with_cache (not in the functional inputs→outputs
        # graph), but must be saved/loaded as part of the model weights.
        #
        # pre_projection: concat(embed_tokens(last_id), last_hidden) → hidden.
        # Input dim = 2 * backbone_hidden_size.
        self.pre_projection = layers.Dense(
            hidden_size,
            use_bias=False,
            dtype=backbone.dtype_policy,
            name="pre_projection",
        )
        self.pre_projection.build((None, 1, 2 * backbone_hidden_size))

        # post_projection: hidden → backbone_hidden_size.
        self.post_projection = layers.Dense(
            backbone_hidden_size,
            use_bias=False,
            dtype=backbone.dtype_policy,
            name="post_projection",
        )
        self.post_projection.build((None, 1, hidden_size))
        self.centroids = None
        self.token_ordering = None
        if use_ordered_embeddings:
            self.centroids = layers.Dense(
                num_centroids,
                use_bias=False,
                dtype=backbone.dtype_policy,
                name="centroids",
            )
            self.centroids.build((None, 1, hidden_size))

            # Store as float32 so TensorFlow places this variable on GPU
            # alongside other model weights. Cast to int32 at use time in
            # _apply_centroid_logits. Using int32 here causes TF to place
            # the variable on CPU, which breaks XLA execution on GPU.
            self.token_ordering = self.add_weight(
                name="token_ordering",
                shape=(vocabulary_size,),
                dtype="float32",
                trainable=False,
                initializer="zeros",
            )

    def _apply_centroid_logits(self, hidden_states):
        """Compute sparse logits via centroid-based vocabulary masking.

          1. Route hidden_states to `num_centroids` cluster scores.
          2. Select the top-k active clusters per token position.
          3. For each active cluster, retrieve the canonical token positions
             stored in `token_ordering`.
          4. Dot-product with the lm_head (tied embedding) weights.
          5. Scatter into a full-vocab output; non-active positions are masked
             with a value below the minimum active logit.

        Args:
            hidden_states: float tensor `(batch, seq, hidden_size)`.

        Returns:
            logits: float tensor `(batch, seq, vocabulary_size)`.
        """
        vocab_size = self.backbone.vocabulary_size
        top_k = self.centroid_intermediate_top_k  # active centroids per step
        k_per_centroid = (
            self._vocab_size_per_centroid
        )  # vocab_size // num_centroids

        # (batch, seq, num_centroids)
        centroid_logits = self.centroids(hidden_states)
        # top-k active centroid indices: (batch, seq, top_k)
        top_k_indices = ops.top_k(centroid_logits, k=top_k)[1]

        # token_ordering: (vocab_size,) → (num_centroids, k_per_centroid)
        token_ord = ops.reshape(
            ops.cast(self.token_ordering, "int32"),
            (self.num_centroids, k_per_centroid),
        )

        # For each position, gather the canonical token ids of active clusters.
        # Shape: (batch, seq, top_k, k_per_centroid)
        selected_canonical = ops.take(token_ord, top_k_indices, axis=0)

        # Gather lm_head embedding weights at selected canonical positions.
        embedding_weights = self.backbone.token_embedding.embeddings
        batch = ops.shape(hidden_states)[0]
        seq = ops.shape(hidden_states)[1]
        n_tokens = top_k * k_per_centroid

        selected_flat = ops.reshape(selected_canonical, (-1,))
        gathered = ops.take(embedding_weights, selected_flat, axis=0)
        # (batch, seq, n_tokens, hidden_size)
        gathered = ops.reshape(gathered, (batch, seq, n_tokens, -1))

        # Dot-product: (batch, seq, n_tokens)
        hs_exp = ops.expand_dims(hidden_states, axis=-2)
        selected_logits = ops.squeeze(
            ops.matmul(hs_exp, ops.transpose(gathered, (0, 1, 3, 2))),
            axis=-2,
        )

        # Scatter active logits into a full-vocab output tensor.
        # Inactive positions are set far below the active range so their
        # softmax probability is negligible.
        flat_hs = batch * seq
        scatter_idx = ops.reshape(selected_canonical, (flat_hs, n_tokens))
        flat_logits = ops.reshape(selected_logits, (flat_hs, n_tokens))
        min_active = ops.min(flat_logits)
        inactive_fill = min_active - ops.cast(100.0, flat_logits.dtype)
        output = (
            ops.ones((flat_hs, vocab_size), dtype=hidden_states.dtype)
            * inactive_fill
        )

        # Memory-efficient sparse scatter to avoid full OOM one-hot allocation.
        row_idx = ops.expand_dims(ops.arange(flat_hs, dtype="int32"), axis=-1)
        row_idx = ops.broadcast_to(row_idx, (flat_hs, n_tokens))

        # Form sparse coordinates: (flat_hs * n_tokens, 2)
        coords = ops.stack(
            [ops.reshape(row_idx, (-1,)), ops.reshape(scatter_idx, (-1,))],
            axis=1,
        )
        updates = ops.reshape(flat_logits, (-1,))

        output = ops.scatter_update(output, coords, updates)
        return ops.reshape(output, (batch, seq, vocab_size))

    def call_with_cache(
        self,
        last_token_embedding,
        last_hidden_state,
        target_cache,
        cache_update_index,
        padding_mask=None,
        target_kv_source_full_idx=None,
        target_kv_source_local_idx=None,
    ):
        """Single-step forward pass for speculative decoding.

        Drafts one candidate token given the target model's state.

        Args:
            last_token_embedding: float tensor
                `(batch, 1, backbone_hidden_size)`.
                The target model's embedding of the last accepted token.
            last_hidden_state: float tensor
                `(batch, 1, backbone_hidden_size)`. Target model's last
                position hidden state.
            target_cache: float tensor
                `(batch, num_target_layers, 2, max_len, num_heads, head_dim)`.
                The target model's full KV cache. The last non-KV-shared
                full/sliding attention layers' K/V slices are injected into
                the assistant's KV-shared layers.
            cache_update_index: int or int tensor. The current decode position.
            padding_mask: optional int tensor `(batch, seq_len)`.

        Returns:
            `(logits, next_hidden_state)` where `logits` has shape
            `(batch, 1, vocabulary_size)` and `next_hidden_state` has shape
            `(batch, 1, backbone_hidden_size)`.
        """
        # Concatenate target token embedding and target hidden state:
        # (batch, 1, 2 * backbone_hidden_size).
        inputs_embeds = ops.concatenate(
            [last_token_embedding, last_hidden_state], axis=-1
        )

        x = self.pre_projection(inputs_embeds)

        # Build a shared KV map from the target model's KV cache.
        # `target_kv_source_full_idx` / `target_kv_source_local_idx` point to
        # the last non-KV-shared full/sliding attention layers in the target,
        # which are the correct source for shared K/V.
        num_target = ops.shape(target_cache)[1]
        if target_kv_source_full_idx is not None:
            shared_kv_global = target_cache[:, target_kv_source_full_idx, ...]
        else:
            shared_kv_global = target_cache[:, num_target - 1, ...]
        if target_kv_source_local_idx is not None:
            shared_kv_local = target_cache[:, target_kv_source_local_idx, ...]
        else:
            shared_kv_local = target_cache[:, num_target - 2, ...]

        layer_types = self.backbone.layer_types or []
        shared_kv_map = {
            i: (shared_kv_global if lt == "full_attention" else shared_kv_local)
            for i, lt in enumerate(layer_types)
        }

        # Run through all 4 KV-shared transformer layers (no cache write).
        for i, transformer_layer in enumerate(self.backbone.transformer_layers):
            layer_shared_kv = shared_kv_map.get(i)
            x, _ = transformer_layer(
                x,
                padding_mask=padding_mask,
                # Pass shared_kv also as `cache` so the decoder block can
                # compute the correct attention mask using the full target
                # key sequence length (cache_seq).  The attention module
                # will use `shared_kv` for actual K/V (is_kv_shared_layer
                # path) and will return `cache` unchanged.
                cache=layer_shared_kv,
                cache_update_index=cache_update_index,
                shared_kv=layer_shared_kv,
            )

        hidden_states = self.backbone.layer_norm(x)
        # Conditionally utilize sparse centroid projection
        if self.use_ordered_embeddings:
            logits = self._apply_centroid_logits(hidden_states)
        else:
            logits = self.backbone.token_embedding(hidden_states, reverse=True)
        next_hidden_state = self.post_projection(hidden_states)

        return logits, next_hidden_state

    @property
    def _layers(self):
        base = self.__dict__.get("_layers_storage", [])
        extra = []
        for attr in ("pre_projection", "post_projection", "centroids"):
            layer = getattr(self, attr, None)
            if layer is not None and layer not in base:
                extra.append(layer)
        return base + extra

    @_layers.setter
    def _layers(self, value):
        # Use object.__setattr__ to bypass __setattr__ → tracker recursion.
        object.__setattr__(self, "_layers_storage", value)

    def _flatten_layers(self, include_self=True, recursive=True):
        """Delegate to super(); the ``_layers`` property already adds extras."""
        return super()._flatten_layers(
            include_self=include_self, recursive=recursive
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone_hidden_size": self.backbone_hidden_size,
                "num_centroids": self.num_centroids,
                "centroid_intermediate_top_k": (
                    self.centroid_intermediate_top_k
                ),
                "use_ordered_embeddings": self.use_ordered_embeddings,
                "num_speculative_tokens": self.num_speculative_tokens,
                "sampler": self._serialize_sampler(),
            }
        )
        return config

    def _serialize_sampler(self):
        from keras_hub.src.samplers.serialization import serialize

        return serialize(self.sampler)

    @classmethod
    def from_config(cls, config):
        if "sampler" in config and isinstance(config["sampler"], dict):
            from keras_hub.src.samplers.serialization import deserialize

            config["sampler"] = deserialize(config["sampler"])
        return super().from_config(config)
