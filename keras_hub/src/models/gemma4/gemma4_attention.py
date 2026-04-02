import inspect

import keras
from keras import ops

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.models.gemma4.gemma4_layers import Gemma4ClippableEinsumDense
from keras_hub.src.models.gemma4.gemma4_layers import Gemma4VNorm
from keras_hub.src.models.gemma4.gemma4_layers import RMSNormalization
from keras_hub.src.utils.keras_utils import clone_initializer
from keras_hub.src.utils.keras_utils import fused_attention_op_available
from keras_hub.src.utils.keras_utils import gpu_supports_fused_attention_op
from keras_hub.src.utils.keras_utils import running_on_gpu
from keras_hub.src.utils.keras_utils import running_on_tpu


class Gemma4TextAttention(keras.layers.Layer):
    """A cached grouped query attention layer for Gemma4.

    This is similar to Gemma3 attention, but with the following differences:

    1. `scaling = 1.0` instead of `1/sqrt(head_dim)`. Gemma4 uses Q/K
       normalization to stabilize attention logits instead of explicit scaling.
    2. `v_norm` is applied to value vectors (pure L2 normalization, no scale).
    3. Q/K normalization is always enabled (not configurable).
    4. `logit_soft_cap` is optional (None by default for Gemma4 text).

    These changes are based on the Gemma4 architecture from the Transformers
    implementation.
    """

    def __init__(
        self,
        head_dim,
        num_query_heads,
        num_key_value_heads,
        kernel_initializer="glorot_uniform",
        logit_soft_cap=None,
        use_sliding_window_attention=False,
        sliding_window_size=4096,
        layer_norm_epsilon=1e-6,
        rope_wavelength=10_000.0,
        rope_scaling_factor=1.0,
        rope_partial_rotary_factor=1.0,
        use_bidirectional_attention=False,
        is_global_attention=False,
        is_kv_shared_layer=False,
        attention_k_eq_v=False,
        num_global_key_value_heads=None,
        dropout=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.logit_soft_cap = logit_soft_cap
        self.use_sliding_window_attention = use_sliding_window_attention
        self.sliding_window_size = sliding_window_size
        self.layer_norm_epsilon = layer_norm_epsilon
        self.rope_wavelength = rope_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_partial_rotary_factor = rope_partial_rotary_factor
        self.use_bidirectional_attention = use_bidirectional_attention
        self.is_global_attention = is_global_attention
        self.is_kv_shared_layer = is_kv_shared_layer
        # attention_k_eq_v: K and V share the same projection. Only active
        # on non-sliding (global) attention layers.
        self.attention_k_eq_v = (
            attention_k_eq_v and not use_sliding_window_attention
        )
        self.num_global_key_value_heads = num_global_key_value_heads
        self.dropout = dropout
        # Partial rotary: number of head-dim positions that actually get RoPE.
        # Remaining positions pass through unchanged (NoPE).
        # `rotary_dim` must be even; round down and clamp to [2, head_dim].
        raw_rotary_dim = int(rope_partial_rotary_factor * head_dim)
        self.rotary_dim = max(2, raw_rotary_dim - raw_rotary_dim % 2)

        # Effective KV head count: may be reduced for global layers.
        if self.attention_k_eq_v and num_global_key_value_heads is not None:
            self.effective_num_kv_heads = num_global_key_value_heads
        else:
            self.effective_num_kv_heads = num_key_value_heads
        self.num_key_value_groups = (
            num_query_heads // self.effective_num_kv_heads
        )

        self._kernel_initializer = keras.initializers.get(
            clone_initializer(kernel_initializer)
        )

    def build(self, inputs_shape):
        self.hidden_dim = inputs_shape[-1]

        self.query_dense = keras.layers.EinsumDense(
            "btd,ndh->btnh",
            output_shape=(None, self.num_query_heads, self.head_dim),
            kernel_initializer=self._kernel_initializer,
            dtype=self.dtype_policy,
            name="query",
        )
        self.query_dense.build(inputs_shape)

        self.key_dense = keras.layers.EinsumDense(
            "bsd,kdh->bskh",
            output_shape=(None, self.effective_num_kv_heads, self.head_dim),
            kernel_initializer=self._kernel_initializer,
            dtype=self.dtype_policy,
            name="key",
        )
        self.key_dense.build(inputs_shape)

        # When attention_k_eq_v, V reuses K's projection — no separate weight.
        if not self.attention_k_eq_v:
            self.value_dense = keras.layers.EinsumDense(
                "bsd,kdh->bskh",
                output_shape=(None, self.effective_num_kv_heads, self.head_dim),
                kernel_initializer=self._kernel_initializer,
                dtype=self.dtype_policy,
                name="value",
            )
            self.value_dense.build(inputs_shape)
        else:
            self.value_dense = None

        # Always apply Q/K norms in Gemma4.
        query_shape = self.query_dense.compute_output_shape(inputs_shape)
        self.query_norm = RMSNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="query_norm",
        )
        self.query_norm.build(query_shape)

        key_shape = self.key_dense.compute_output_shape(inputs_shape)
        self.key_norm = RMSNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="key_norm",
        )
        self.key_norm.build(key_shape)

        # V norm: pure L2 normalization (no learnable scale).
        # When attention_k_eq_v, value comes from key_dense so same shape.
        kv_norm_shape = (None, None, self.effective_num_kv_heads, self.head_dim)
        self.value_norm = Gemma4VNorm(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="value_norm",
        )
        self.value_norm.build(kv_norm_shape)

        self.dropout_layer = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
        )

        self.output_dense = keras.layers.EinsumDense(
            equation="btnh,nhd->btd",
            output_shape=(None, self.hidden_dim),
            kernel_initializer=self._kernel_initializer,
            dtype=self.dtype_policy,
            name="attention_output",
        )
        self.output_dense.build(
            (None, None, self.num_query_heads, self.head_dim)
        )
        self.softmax = keras.layers.Softmax(dtype="float32")

        self.rope_layer = RotaryEmbedding(
            max_wavelength=self.rope_wavelength,
            scaling_factor=self.rope_scaling_factor,
            denominator_dim=self.head_dim,
            dtype=self.dtype_policy,
        )

        self.built = True

    def _apply_rope(self, x, start_index):
        """Apply RoPE, with optional partial (proportional) rotation.

        In Gemma 4, global attention uses partial proportionate rotation where
        only `rotary_dim` // 2 elements of each `head_dim // 2` half
        are rotated.
        """
        if self.rotary_dim < self.head_dim:
            half_rotary = self.rotary_dim // 2
            half_head = self.head_dim // 2

            x1 = x[..., :half_head]
            x2 = x[..., half_head:]

            x1_rot = x1[..., :half_rotary]
            x1_nope = x1[..., half_rotary:]

            x2_rot = x2[..., :half_rotary]
            x2_nope = x2[..., half_rotary:]

            x_rot = ops.concatenate([x1_rot, x2_rot], axis=-1)
            x_rot = self.rope_layer(x_rot, start_index=start_index)

            x1_rot, x2_rot = ops.split(x_rot, 2, axis=-1)

            y1 = ops.concatenate([x1_rot, x1_nope], axis=-1)
            y2 = ops.concatenate([x2_rot, x2_nope], axis=-1)

            return ops.concatenate([y1, y2], axis=-1)
        return self.rope_layer(x, start_index=start_index)

    def _use_fused_attention_op(self):
        if not fused_attention_op_available():
            return False
        if self.dropout > 0.0:
            return False
        if running_on_gpu():
            # GPU never supports softcap in the fused op.
            if self.logit_soft_cap is not None:
                return False
            return gpu_supports_fused_attention_op()
        elif running_on_tpu():
            # TPU supports softcap with on keras >= 3.10.
            sig = inspect.signature(ops.dot_product_attention)
            return "attn_logits_soft_cap" in sig.parameters
        else:
            return False

    def _compute_attention(
        self,
        q,
        k,
        v,
        attention_mask,
        training=False,
        cache_update_index=0,
    ):
        # Gemma4 uses scaling = 1.0 (Q/K norm replaces 1/sqrt(head_dim)).
        query_normalization = 1.0

        # Note: the sliding window mask is now applied in the decoder block's
        # _compute_attention_mask(), before any vision bidirec OR, to match HF:
        #   (causal AND sliding) OR vision_bidirec
        # So we do NOT apply it again here.

        if self._use_fused_attention_op():
            if attention_mask is not None:
                attention_mask = ops.expand_dims(attention_mask, axis=1)
                attention_mask = ops.cast(attention_mask, dtype="bool")
            if self.logit_soft_cap:
                kwargs = {"attn_logits_soft_cap": self.logit_soft_cap}
            else:
                kwargs = {}

            if self.num_key_value_groups > 1:
                k = ops.repeat(k, repeats=self.num_key_value_groups, axis=2)
                v = ops.repeat(v, repeats=self.num_key_value_groups, axis=2)

            return ops.dot_product_attention(
                query=q,
                key=k,
                value=v,
                mask=attention_mask,
                scale=query_normalization,
                **kwargs,
            )

        q_shape = ops.shape(q)
        q = ops.reshape(
            q,
            (
                *q_shape[:-2],
                self.effective_num_kv_heads,
                self.num_query_heads // self.effective_num_kv_heads,
                q_shape[-1],
            ),
        )
        b, q_len, _, _, h = ops.shape(q)

        # Compute attention logits.
        attention_logits = ops.einsum("btkgh,bskh->bkgts", q, k)
        if self.logit_soft_cap is not None:
            attention_logits = ops.divide(attention_logits, self.logit_soft_cap)
            attention_logits = ops.multiply(
                ops.tanh(attention_logits), self.logit_soft_cap
            )

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :, :]
        orig_dtype = attention_logits.dtype
        attention_softmax = self.softmax(attention_logits, mask=attention_mask)
        attention_softmax = ops.cast(attention_softmax, orig_dtype)

        if self.dropout:
            attention_softmax = self.dropout_layer(
                attention_softmax, training=training
            )

        results = ops.einsum("bkgts,bskh->btkgh", attention_softmax, v)
        return ops.reshape(results, (b, q_len, self.num_query_heads, h))

    def _compute_bidirectional_sliding_mask(self, batch_size, sequence_length):
        """Computes a bidirectional sliding window attention mask."""
        i = keras.ops.expand_dims(
            keras.ops.arange(sequence_length, dtype="int32"), axis=1
        )
        j = keras.ops.arange(sequence_length, dtype="int32")

        w_right = self.sliding_window_size // 2
        w_left = self.sliding_window_size - w_right - 1

        distance = i - j
        mask = keras.ops.logical_and(distance <= w_left, distance >= -w_right)
        mask = keras.ops.expand_dims(mask, axis=0)
        return keras.ops.broadcast_to(
            mask, (batch_size, sequence_length, sequence_length)
        )

    def _mask_sliding_window(
        self,
        attention_mask,
        cache_update_index=0,
    ):
        batch_size, query_len, key_len = ops.shape(attention_mask)

        if self.use_bidirectional_attention:
            bidirectional_sliding_mask = (
                self._compute_bidirectional_sliding_mask(
                    batch_size=batch_size,
                    sequence_length=query_len,
                )
            )
            return ops.logical_and(attention_mask, bidirectional_sliding_mask)

        all_ones = ops.ones((key_len, key_len), "bool")
        if keras.config.backend() == "tensorflow":
            import tensorflow as tf

            band_size = ops.minimum(key_len, self.sliding_window_size - 1)
            band_size = ops.cast(band_size, "int32")
            sliding_mask = tf.linalg.band_part(all_ones, band_size, band_size)
        else:
            sliding_mask = ops.triu(
                all_ones, -1 * self.sliding_window_size + 1
            ) * ops.tril(all_ones, self.sliding_window_size - 1)
        start = (cache_update_index, 0)
        sliding_mask = ops.slice(sliding_mask, start, (query_len, key_len))
        sliding_mask = ops.expand_dims(sliding_mask, 0)
        return ops.logical_and(attention_mask, ops.cast(sliding_mask, "bool"))

    def call(
        self,
        x,
        attention_mask=None,
        cache=None,
        cache_update_index=0,
        cache_update_mask=None,
        shared_kv=None,
        training=False,
    ):
        query = self.query_dense(x)
        query = self.query_norm(query)
        query = self._apply_rope(query, cache_update_index)

        if cache is not None:
            key_cache = cache[:, 0, ...]
            value_cache = cache[:, 1, ...]

            if self.is_kv_shared_layer and shared_kv is not None:
                # KV-shared layer: fetch K/V from the reference layer's cache
                # instead of computing them. The cache slot for this layer is
                # returned unchanged.
                key = shared_kv[:, 0, ...]
                value = shared_kv[:, 1, ...]
                new_cache = cache
            else:
                key_update = self.key_dense(x)
                key_update = self.key_norm(key_update)
                key_update = self._apply_rope(key_update, cache_update_index)
                if self.attention_k_eq_v:
                    # K=V: value projection reuses the key_dense computation.
                    value_update = self.value_norm(self.key_dense(x))
                else:
                    value_update = self.value_dense(x)
                    value_update = self.value_norm(value_update)

                start = [0, cache_update_index, 0, 0]
                if cache_update_mask is not None:
                    cache_update_mask_exp = ops.expand_dims(
                        ops.expand_dims(cache_update_mask, axis=-1),
                        axis=-1,
                    )
                    key_original = ops.slice(
                        key_cache, start, ops.shape(key_update)
                    )
                    value_original = ops.slice(
                        value_cache, start, ops.shape(value_update)
                    )
                    key_update = ops.where(
                        cache_update_mask_exp,
                        key_update,
                        key_original,
                    )
                    value_update = ops.where(
                        cache_update_mask_exp,
                        value_update,
                        value_original,
                    )

                key = ops.slice_update(key_cache, start, key_update)
                value = ops.slice_update(value_cache, start, value_update)
                new_cache = ops.stack((key, value), axis=1)
        else:
            if self.is_kv_shared_layer and shared_kv is not None:
                key = shared_kv[:, 0, ...]
                value = shared_kv[:, 1, ...]
            else:
                if self.attention_k_eq_v:
                    # K=V: value projection reuses key_dense weights.
                    raw_kv = self.key_dense(x)
                    key = self.key_norm(raw_kv)
                    key = self._apply_rope(key, cache_update_index)
                    value = self.value_norm(raw_kv)
                else:
                    key = self.key_dense(x)
                    key = self.key_norm(key)
                    key = self._apply_rope(key, cache_update_index)
                    value = self.value_dense(x)
                    value = self.value_norm(value)
            new_cache = ops.stack((key, value), axis=1)

        # When global attention layers use global_head_dim > head_dim, the
        # cache is allocated with max_head_dim = max(head_dim, global_head_dim)
        # to keep all per-layer cache tensors the same shape for ops.stack().
        # Slice key/value back to this layer's actual head_dim before the
        # attention dot-product.  new_cache already holds the full-width tensor.
        if cache is not None:
            key = key[..., : self.effective_num_kv_heads, : self.head_dim]
            value = value[..., : self.effective_num_kv_heads, : self.head_dim]

        attention_vec = self._compute_attention(
            query,
            key,
            value,
            attention_mask,
            training=training,
            cache_update_index=cache_update_index,
        )

        # Wipe attn vec if there are no attended tokens.
        if attention_mask is not None:
            no_attended_tokens = ops.all(
                ops.equal(attention_mask, 0), axis=-1, keepdims=True
            )[..., None]
            attention_vec = ops.where(
                no_attended_tokens,
                ops.zeros_like(attention_vec),
                attention_vec,
            )

        attention_output = self.output_dense(attention_vec)

        return attention_output, new_cache

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        cache_shape = (
            batch_size,
            2,
            seq_length,
            self.num_key_value_heads,
            self.head_dim,
        )
        return input_shape, cache_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "head_dim": self.head_dim,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "logit_soft_cap": self.logit_soft_cap,
                "use_sliding_window_attention": (
                    self.use_sliding_window_attention
                ),
                "sliding_window_size": self.sliding_window_size,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "rope_wavelength": self.rope_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "use_bidirectional_attention": self.use_bidirectional_attention,
                "is_global_attention": self.is_global_attention,
                "is_kv_shared_layer": self.is_kv_shared_layer,
                "attention_k_eq_v": self.attention_k_eq_v,
                "num_global_key_value_heads": self.num_global_key_value_heads,
                "dropout": self.dropout,
                "rope_partial_rotary_factor": self.rope_partial_rotary_factor,
            }
        )
        return config


class Gemma4VisionAttention(keras.layers.Layer):
    """Vision attention block for Gemma4.

    This is structurally aligned with the model attention block but enforces
    bidirectional attention natively without caching or causal masking.
    """

    def __init__(
        self,
        head_dim,
        num_query_heads,
        num_key_value_heads,
        logit_soft_cap=None,
        layer_norm_epsilon=1e-6,
        rope_wavelength=10_000.0,
        rope_scaling_factor=1.0,
        rope_partial_rotary_factor=1.0,
        dropout=0,
        use_clipped_linears=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.head_dim = head_dim
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.logit_soft_cap = logit_soft_cap
        self.layer_norm_epsilon = layer_norm_epsilon
        self.rope_wavelength = rope_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_partial_rotary_factor = rope_partial_rotary_factor
        self.dropout = dropout
        self.use_clipped_linears = use_clipped_linears

        self.hidden_dim = self.num_query_heads * self.head_dim
        self.rotary_dim = int(self.head_dim * self.rope_partial_rotary_factor)

        # Instantiate dense layers and norms
        self.query_dense = Gemma4ClippableEinsumDense(
            "bsd,qdh->bsqh",
            output_shape=(None, self.num_query_heads, self.head_dim),
            kernel_initializer="glorot_uniform",
            use_clipped_linears=use_clipped_linears,
            dtype=self.dtype_policy,
            name="query",
        )
        self.key_dense = Gemma4ClippableEinsumDense(
            "bsd,kdh->bskh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            kernel_initializer="glorot_uniform",
            use_clipped_linears=use_clipped_linears,
            dtype=self.dtype_policy,
            name="key",
        )
        self.value_dense = Gemma4ClippableEinsumDense(
            "bsd,kdh->bskh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            kernel_initializer="glorot_uniform",
            use_clipped_linears=use_clipped_linears,
            dtype=self.dtype_policy,
            name="value",
        )
        self.query_norm = RMSNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="query_norm",
        )
        self.key_norm = RMSNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="key_norm",
        )
        self.value_norm = Gemma4VNorm(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="value_norm",
        )
        self.dropout_layer = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
        )
        self.output_dense = Gemma4ClippableEinsumDense(
            equation="btnh,nhd->btd",
            output_shape=(None, self.hidden_dim),
            kernel_initializer="glorot_uniform",
            use_clipped_linears=use_clipped_linears,
            dtype=self.dtype_policy,
            name="attention_output",
        )
        self.softmax = keras.layers.Softmax(dtype="float32")
        self.x_rope = RotaryEmbedding(
            max_wavelength=self.rope_wavelength,
            scaling_factor=self.rope_scaling_factor,
            denominator_dim=self.head_dim // 2,
            dtype=self.dtype_policy,
        )
        self.y_rope = RotaryEmbedding(
            max_wavelength=self.rope_wavelength,
            scaling_factor=self.rope_scaling_factor,
            denominator_dim=self.head_dim // 2,
            dtype=self.dtype_policy,
        )

    def build(self, input_shape):
        self.query_dense.build(input_shape)
        self.key_dense.build(input_shape)
        self.value_dense.build(input_shape)

        query_shape = self.query_dense.compute_output_shape(input_shape)
        self.query_norm.build(query_shape)
        key_shape = self.key_dense.compute_output_shape(input_shape)
        self.key_norm.build(key_shape)

        kv_norm_shape = (None, None, self.num_key_value_heads, self.head_dim)
        self.value_norm.build(kv_norm_shape)

        self.output_dense.build(
            (None, None, self.num_query_heads, self.head_dim)
        )
        self.built = True

    def _apply_rope(self, x, position_ids):
        half_head = self.head_dim // 2
        x_part = x[..., :half_head]
        y_part = x[..., half_head:]

        x_ids = position_ids[..., 0]
        y_ids = position_ids[..., 1]

        # Custom symmetric RoPE to match HF
        def get_rope(part, ids):
            dim = half_head
            idx = ops.arange(0, dim, 2, dtype="float32")
            inv_freq = ops.power(
                ops.cast(self.rope_wavelength, "float32"), -idx / dim
            )
            # ids shape is (B, Tokens)
            # inv_freq shape is (dim // 2,)
            freqs = ops.einsum("bi,j->bij", ops.cast(ids, "float32"), inv_freq)
            # freqs shape is (B, Tokens, dim // 2)
            # Concatenate freqs and freqs to get size dim
            emb = ops.concatenate([freqs, freqs], axis=-1)
            cos = ops.expand_dims(ops.cos(emb), axis=2)
            sin = ops.expand_dims(ops.sin(emb), axis=2)

            # rotate_half (split into halves)
            x1, x2 = ops.split(part, 2, axis=-1)
            half_rot = ops.concatenate([-x2, x1], axis=-1)
            return (part * cos) + (half_rot * sin)

        x_rot = get_rope(x_part, x_ids)
        y_rot = get_rope(y_part, y_ids)

        out = ops.concatenate([x_rot, y_rot], axis=-1)
        return out

    def _compute_attention(self, q, k, v, attention_mask, training=False):
        q_shape = ops.shape(q)
        q = ops.reshape(
            q,
            (
                *q_shape[:-2],
                self.num_key_value_heads,
                self.num_query_heads // self.num_key_value_heads,
                q_shape[-1],
            ),
        )
        b, q_len, _, _, h = ops.shape(q)

        attention_logits = ops.einsum("btkgh,bskh->bkgts", q, k)
        if self.logit_soft_cap is not None:
            attention_logits = ops.divide(attention_logits, self.logit_soft_cap)
            attention_logits = ops.multiply(
                ops.tanh(attention_logits), self.logit_soft_cap
            )

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, None, :]
        orig_dtype = attention_logits.dtype
        attention_softmax = self.softmax(attention_logits, mask=attention_mask)
        attention_softmax = ops.cast(attention_softmax, orig_dtype)

        if self.dropout:
            attention_softmax = self.dropout_layer(
                attention_softmax, training=training
            )

        results = ops.einsum("bkgts,bskh->btkgh", attention_softmax, v)
        return ops.reshape(results, (b, q_len, self.num_query_heads, h))

    def call(self, x, attention_mask=None, position_ids=None, training=False):
        query = self.query_dense(x)
        query = self.query_norm(query)
        if position_ids is not None:
            query = self._apply_rope(query, position_ids)

        key = self.key_dense(x)
        key = self.key_norm(key)
        if position_ids is not None:
            key = self._apply_rope(key, position_ids)

        value = self.value_dense(x)
        value = self.value_norm(value)

        # Vision doesn't utilize padding cache maps over generations
        attention_vec = self._compute_attention(
            query,
            key,
            value,
            attention_mask,
            training=training,
        )

        # NOTE: We intentionally do NOT zero padding query outputs here.
        # HF masks only keys (padding keys get -inf), so padding queries attend
        # to valid keys and produce non-zero attention outputs. This matches
        # HF's behavior where padding patches accumulate meaningful hidden
        # states that contribute to pool bin 0 in the pooler.

        attention_output = self.output_dense(attention_vec)
        return attention_output, None

    def compute_output_shape(self, input_shape):
        return input_shape, None

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "head_dim": self.head_dim,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "logit_soft_cap": self.logit_soft_cap,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "rope_wavelength": self.rope_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "dropout": self.dropout,
                "rope_partial_rotary_factor": self.rope_partial_rotary_factor,
            }
        )
        return config
