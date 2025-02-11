import keras
from keras import layers
from keras import ops
from moonshine_utils import apply_rotary_pos_emb


class MHAWithRope(layers.MultiHeadAttention):
    def call(self, query, value, key, rot_pos_emb):
        query = self._query_dense(query)
        key = self._key_dense(key)
        value = self._value_dense(value)
        query = apply_rotary_pos_emb(query, rot_pos_emb)
        key = apply_rotary_pos_emb(key, rot_pos_emb)
        attention_output, _ = self._compute_attention(
            query=query,
            key=key,
            value=value,
            training=None,
        )
        output = self._output_dense(attention_output)
        return output

    def compute_output_spec(self, **kwargs):
        kwargs.pop("rot_pos_emb", None)
        return super(MHAWithRope, self).compute_output_spec(**kwargs)


class MHACausalWithRope(layers.MultiHeadAttention):
    def call(
        self, query, value, key, rot_pos_emb, value_cache=None, key_cache=None
    ):
        query = self._query_dense(query)
        key = self._key_dense(key)
        value = self._value_dense(value)
        query = apply_rotary_pos_emb(query, rot_pos_emb)
        key = apply_rotary_pos_emb(key, rot_pos_emb)

        if value_cache is not None:
            assert key_cache is not None, (
                "key_cache should not be None when value_cache is not"
            )
            key = ops.concatenate((key_cache, key), axis=-3)
            value = ops.concatenate((value_cache, value), axis=-3)

        causal_mask = self._compute_causal_mask(
            query, value, for_cache=value_cache is not None
        )

        attention_output, _ = self._compute_attention(
            query=query,
            key=key,
            value=value,
            attention_mask=causal_mask,
            training=None,
        )
        output = self._output_dense(attention_output)
        return output, key, value

    def _compute_causal_mask(self, query, value=None, for_cache=False):
        # When for_cache is True, ensure that value is not None.
        if for_cache:
            assert value is not None, (
                "value cannot be none if for_cache is True"
            )
        q_seq_length = ops.shape(query)[1]
        v_seq_length = q_seq_length if value is None else ops.shape(value)[1]
        n_rows = v_seq_length if for_cache else q_seq_length
        ones_mask = ops.ones((1, n_rows, v_seq_length), dtype="int32")
        row_index = ops.cumsum(ones_mask, axis=-2)
        col_index = ops.cumsum(ones_mask, axis=-1)
        mask = ops.greater_equal(row_index, col_index)
        if for_cache:
            mask = mask[:, -q_seq_length:, :]
        return mask

    def compute_output_spec(self, **kwargs):
        kwargs.pop("rot_pos_emb", None)
        kwargs.pop("key_cache", None)
        kwargs.pop("value_cache", None)
        attention_spec = super(MHACausalWithRope, self).compute_output_spec(
            **kwargs
        )
        key_spec = keras.KerasTensor(
            (None, None, self.num_heads, self.key_dim), dtype=self.compute_dtype
        )
        value_spec = keras.KerasTensor(
            (None, None, self.num_heads, self.value_dim),
            dtype=self.compute_dtype,
        )
        return attention_spec, key_spec, value_spec


class MHAPrecomputedKV(layers.MultiHeadAttention):
    def call(self, query, value, key, key_cache=None, value_cache=None):
        query = self._query_dense(query)
        if key_cache is None:
            # No cache provided: compute key and value normally.
            assert value_cache is None, (
                "Both key and value cache have to be None"
            )
            key = self.key_dense(key)
            value = self.value_dense(value)
        else:
            key = key_cache
            value = value_cache

        attention_output, _ = self._compute_attention(
            query=query,
            key=key,
            value=value,
            training=None,
        )
        output = self._output_dense(attention_output)
        if key_cache is None:
            return output, key, value
        return output

    def compute_output_spec(self, **kwargs):
        key_cache = kwargs.pop("key_cache", None)
        attention_spec = super(MHAPrecomputedKV, self).compute_output_spec(
            **kwargs
        )
        if key_cache is None:
            key_spec = keras.KerasTensor(
                (None, None, self.num_heads, self.key_dim),
                dtype=self.compute_dtype,
            )
            value_spec = keras.KerasTensor(
                (None, None, self.num_heads, self.value_dim),
                dtype=self.compute_dtype,
            )
            return attention_spec, key_spec, value_spec
        return attention_spec
