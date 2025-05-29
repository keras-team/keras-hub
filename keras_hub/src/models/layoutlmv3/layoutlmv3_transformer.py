from keras import layers
from keras.saving import register_keras_serializable

@register_keras_serializable()
class LayoutLMv3TransformerLayer(layers.Layer):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        qkv_bias=True,
        use_rel_pos=True,
        rel_pos_bins=32,
        max_rel_pos=128,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.use_rel_pos = use_rel_pos
        self.rel_pos_bins = rel_pos_bins
        self.max_rel_pos = max_rel_pos

    def call(self, hidden_states, attention_mask=None, **kwargs):
        # Minimal stub: just return hidden_states unchanged
        return hidden_states 