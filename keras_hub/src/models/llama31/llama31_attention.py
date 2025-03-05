import keras_hub
from keras_hub.src.models.llama31.llama3_rotary_embedding import Llama31RotaryEmbedding
from keras_hub.src.utils.keras_utils import clone_initializer

class Llama31Attention(keras_hub.models.llama.llama_attention.LlamaAttention):
    """A cached grounded query attention layer with sliding window."""

    def __init__(
        self,
        num_query_heads,
        num_key_value_heads,
        rope_max_wavelength=10000,
        rope_scaling_factor=1.0,
        rope_factor=8,
        rope_low_freq_factor=1,
        rope_high_freq_factor=4,
        rope_old_context_len=8192,
        kernel_initializer="glorot_uniform",
        dropout=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.dropout = dropout

        self.num_key_value_groups = num_query_heads // num_key_value_heads
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_factor=rope_factor,
        self.rope_low_freq_factor=rope_low_freq_factor,
        self.rope_high_freq_factor=rope_high_freq_factor,
        self.rope_old_context_len=rope_old_context_len,

        self.kernel_initializer = keras.initializers.get(
            clone_initializer(kernel_initializer)
        )

        self.rope_scaling_factor = rope_scaling_factor

    def build(self, inputs_shape):
        # Einsum variables:
        # b = batch size
        # q = query length
        # k = key/value length
        # m = model dim
        # u = num query heads
        # v = num key/value heads
        # h = head dim
        hidden_dim = inputs_shape[-1]
        head_dim = hidden_dim // self.num_query_heads
        self._inv_norm_factor = 1.0 / math.sqrt(head_dim)

        self._query_dense = keras.layers.EinsumDense(
            equation="bqm,muh->bquh",
            output_shape=(None, self.num_query_heads, head_dim),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="query",
        )
        self._query_dense.build(inputs_shape)

        self._key_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(
                None,
                self.num_key_value_heads,
                head_dim,
            ),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="key",
        )
        self._key_dense.build(inputs_shape)

        self._value_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(
                None,
                self.num_key_value_heads,
                head_dim,
            ),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="value",
        )
        self._value_dense.build(inputs_shape)

        self._softmax = keras.layers.Softmax(
            axis=-1,
            dtype="float32",
            name="attention_softmax",
        )

        self._dropout_layer = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
        )

        self._output_dense = keras.layers.EinsumDense(
            equation="bquh,uhm->bqm",
            output_shape=(None, hidden_dim),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="attention_output",
        )
        self._output_dense.build((None, None, self.num_query_heads, head_dim))

        self.rotary_embedding_layer = Llama31RotaryEmbedding(
            max_wavelength=self.rope_max_wavelength,
            scaling_factor=self.rope_scaling_factor,
            factor=self.rope_factor,
            low_freq_factor=self.rope_low_freq_factor,
            high_freq_factor=self.rope_high_freq_factor,
            old_context_len=self.rope_old_context_len,
        )

        self._dot_product_equation = "bquh,bkuh->buqk"
        self._combine_equation = "buqk,bkuh->bquh"

        self.built = True

        