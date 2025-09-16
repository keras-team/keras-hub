import keras
import numpy as np

from keras_hub.src.models.gemma3n.rms_normalization import Gemma3nRMSNorm


class Gemma3nTextScaledWordEmbedding(keras.layers.Layer):
    """A layer that computes scaled word embeddings for Gemma3n models.

    This layer performs a standard embedding lookup and then scales the
    resulting vectors by a specified factor.

    Args:
        num_embeddings: int. The size of the vocabulary.
        embedding_dim: int. The dimension of the embedding vectors.
        embed_scale: float. The scaling factor applied to the embeddings.
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        embed_scale=1.0,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embed_scale = embed_scale
        self.embedding = keras.layers.Embedding(
            self.num_embeddings,
            self.embedding_dim,
            name="embedding",
            dtype=self.dtype_policy,
        )

    def build(self, input_shape):
        self.embedding.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        return self.embedding(inputs) * self.embed_scale

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_embeddings": self.num_embeddings,
                "embedding_dim": self.embedding_dim,
                "embed_scale": self.embed_scale,
            }
        )
        return config


class Gemma3nTextRotaryEmbedding(keras.layers.Layer):
    """A layer that computes rotary positional embeddings for Gemma3n models.

    This layer calculates the cosine and sine matrices for Rotary Positional
    Embedding (RoPE), which are then applied to query and key tensors in the
    attention mechanism to inject positional information.

    Args:
        head_dim: int. The dimension of each attention head.
        rope_theta: float. The base for the rotary frequency.
        max_position_embeddings: int. The maximum sequence length that this
            model might be used with.
        rope_scaling: dict or `None`. Specifies the scaling strategy for RoPE.
        base: float. The base value for the inverse frequency calculation.
    """

    def __init__(
        self,
        head_dim,
        rope_theta,
        max_position_embeddings,
        rope_scaling,
        base=10000,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.rope_scaling = rope_scaling
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (np.arange(0, self.head_dim, 2, dtype="float32") / self.head_dim)
        )
        self.inv_freq = keras.ops.convert_to_tensor(inv_freq)
        self.attention_scaling = 1.0

    def call(self, x, position_ids):
        inv_freq_expanded = keras.ops.expand_dims(
            keras.ops.expand_dims(self.inv_freq, 0), -1
        )
        inv_freq_expanded = keras.ops.repeat(
            inv_freq_expanded, repeats=keras.ops.shape(position_ids)[0], axis=0
        )
        position_ids_expanded = keras.ops.expand_dims(
            keras.ops.cast(position_ids, "float32"), 1
        )

        freqs = keras.ops.transpose(
            keras.ops.matmul(inv_freq_expanded, position_ids_expanded),
            (0, 2, 1),
        )
        emb = keras.ops.concatenate([freqs, freqs], axis=-1)
        cos = keras.ops.cos(emb) * self.attention_scaling
        sin = keras.ops.sin(emb) * self.attention_scaling
        return keras.ops.cast(cos, x.dtype), keras.ops.cast(sin, x.dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "head_dim": self.head_dim,
                "rope_theta": self.rope_theta,
                "max_position_embeddings": self.max_position_embeddings,
                "rope_scaling": self.rope_scaling,
                "base": self.base,
            }
        )
        return config


class Gemma3nTextMLP(keras.layers.Layer):
    """A Gemma3n-specific feed-forward network (MLP) layer.

    This layer implements the MLP block used in Gemma3n transformer layers,
    featuring a gated linear unit (GLU) structure. It can also apply activation
    sparsity using a Gaussian top-k mechanism.

    Args:
        hidden_size: int. The dimension of the hidden state.
        intermediate_size: int. The dimension of the intermediate layer in the
            MLP.
        hidden_activation: str or callable. The activation function to use.
        activation_sparsity: float. The target sparsity for activations,
            enabling the Gaussian top-k mechanism if greater than 0.
    """

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        hidden_activation,
        activation_sparsity,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_activation = hidden_activation
        self.activation_sparsity = activation_sparsity
        self.gate_proj = keras.layers.Dense(
            intermediate_size,
            use_bias=False,
            name="gate_proj",
            dtype=self.dtype_policy,
        )
        self.up_proj = keras.layers.Dense(
            intermediate_size,
            use_bias=False,
            name="up_proj",
            dtype=self.dtype_policy,
        )
        self.down_proj = keras.layers.Dense(
            hidden_size,
            use_bias=False,
            name="down_proj",
            dtype=self.dtype_policy,
        )
        if hidden_activation == "gelu_approximate":
            # NOTE: `gelu_pytorch_tanh` is the same as `gelu(approximate=True)`.
            self.act_fn = lambda x: keras.activations.gelu(x, approximate=True)
        else:
            self.act_fn = keras.activations.get(hidden_activation)

    def build(self, input_shape):
        self.gate_proj.build(input_shape)
        self.up_proj.build(input_shape)
        self.down_proj.build((None, self.intermediate_size))
        super().build(input_shape)

    def _gaussian_topk(self, inputs):
        target_sparsity_tensor = keras.ops.convert_to_tensor(
            self.activation_sparsity, dtype="float32"
        )
        std_multiplier = keras.ops.erfinv(
            2 * target_sparsity_tensor - 1
        ) * keras.ops.sqrt(keras.ops.convert_to_tensor(2.0, dtype="float32"))
        std_multiplier = keras.ops.cast(std_multiplier, dtype=inputs.dtype)
        inputs_mean = keras.ops.mean(inputs, axis=-1, keepdims=True)
        inputs_std = keras.ops.std(inputs, axis=-1, keepdims=True)
        cutoff_x = inputs_mean + inputs_std * std_multiplier
        return keras.ops.relu(inputs - cutoff_x)

    def call(self, hidden_states):
        gate_proj = self.gate_proj(hidden_states)
        if self.activation_sparsity > 0.0:
            gate_proj = self._gaussian_topk(gate_proj)
        activations = self.act_fn(gate_proj)
        up_proj = self.up_proj(hidden_states)
        down_proj = self.down_proj(activations * up_proj)
        return down_proj

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "hidden_activation": self.hidden_activation,
                "activation_sparsity": self.activation_sparsity,
            }
        )
        return config


class Gemma3nTextLaurelBlock(keras.layers.Layer):
    """A Laurel block layer for the Gemma3n model.

    This layer implements a low-rank residual block which applies a
    down-projection to a specified rank, followed by an up-projection. The
    result is normalized and added back to the original input, forming a
    residual connection.

    Args:
        hidden_size: int. The dimension of the hidden state.
        laurel_rank: int. The rank of the low-rank adaptation.
        rms_norm_eps: float. The epsilon value for the RMS normalization layer.
    """

    def __init__(
        self, hidden_size, laurel_rank, rms_norm_eps, dtype=None, **kwargs
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_size = hidden_size
        self.laurel_rank = laurel_rank
        self.rms_norm_eps = rms_norm_eps
        self.linear_left = keras.layers.Dense(
            laurel_rank,
            use_bias=False,
            name="linear_left",
            dtype=self.dtype_policy,
        )
        self.linear_right = keras.layers.Dense(
            hidden_size,
            use_bias=False,
            name="linear_right",
            dtype=self.dtype_policy,
        )
        self.post_laurel_norm = Gemma3nRMSNorm(
            hidden_size,
            eps=rms_norm_eps,
            name="post_laurel_norm",
            dtype=self.dtype_policy,
        )

    def build(self, input_shape):
        self.linear_left.build(input_shape)
        self.linear_right.build((None, self.laurel_rank))
        self.post_laurel_norm.build(input_shape)
        super().build(input_shape)

    def call(self, hidden_states):
        laurel_hidden_states = self.linear_left(hidden_states)
        laurel_hidden_states = self.linear_right(laurel_hidden_states)
        normed_laurel_hidden_states = self.post_laurel_norm(
            laurel_hidden_states
        )
        return hidden_states + normed_laurel_hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "laurel_rank": self.laurel_rank,
                "rms_norm_eps": self.rms_norm_eps,
            }
        )
        return config


class Gemma3nTextAltUp(keras.layers.Layer):
    """An Alternating Update (AltUp) layer for the Gemma3n model.

    This layer implements the AltUp mechanism, which combines multiple input
    modalities through a predict-and-correct cycle. It uses a router to compute
    modality-specific coefficients for predicting and correcting hidden states.

    Args:
        hidden_size: int. The dimension of the hidden state.
        altup_num_inputs: int. The number of input modalities to the AltUp
            block.
        altup_coef_clip: float. The clipping value for coefficients.
        altup_active_idx: int. The index of the currently active input.
        rms_norm_eps: float. The epsilon value for the Gemma 3n RMS
            normalization layers.
        altup_correct_scale: bool. If `True`, enables a learnable scaling
            factor on the corrected output.
    """

    def __init__(
        self,
        hidden_size,
        altup_num_inputs,
        altup_coef_clip,
        altup_active_idx,
        rms_norm_eps,
        altup_correct_scale,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_size = hidden_size
        self.altup_num_inputs = altup_num_inputs
        self.altup_coef_clip = altup_coef_clip
        self.altup_active_idx = altup_active_idx
        self.rms_norm_eps = rms_norm_eps

        self.altup_correct_scale = altup_correct_scale
        self.correct_output_scale = None
        self.correction_coefs = keras.layers.Dense(
            self.altup_num_inputs,
            use_bias=False,
            name="correction_coefs",
            dtype=self.dtype_policy,
        )
        self.prediction_coefs = keras.layers.Dense(
            self.altup_num_inputs**2,
            use_bias=False,
            name="prediction_coefs",
            dtype=self.dtype_policy,
        )
        self.modality_router = keras.layers.Dense(
            self.altup_num_inputs,
            use_bias=False,
            name="modality_router",
            dtype=self.dtype_policy,
        )
        self.router_norm = Gemma3nRMSNorm(
            self.hidden_size,
            eps=self.rms_norm_eps,
            name="router_norm",
            dtype=self.dtype_policy,
        )
        self.router_input_scale = self.hidden_size**-1.0

    def build(self, input_shape):
        if self.altup_correct_scale:
            self.correct_output_scale = self.add_weight(
                shape=(self.hidden_size,),
                initializer="zeros",
                trainable=True,
                name="correct_output_scale",
                dtype=self.dtype_policy.variable_dtype,
            )
        router_input_shape = input_shape[1:]
        self.router_norm.build(router_input_shape)
        self.modality_router.build(router_input_shape)
        coefs_input_shape = router_input_shape[:-1] + (self.altup_num_inputs,)
        self.correction_coefs.build(coefs_input_shape)
        self.prediction_coefs.build(coefs_input_shape)
        super().build(input_shape)

    def compute_router_modalities(self, x):
        router_inputs = self.router_norm(x) * self.router_input_scale
        routed = self.modality_router(router_inputs)
        return keras.ops.cast(
            keras.ops.tanh(keras.ops.cast(routed, "float32")), x.dtype
        )

    def predict(self, hidden_states):
        modalities = self.compute_router_modalities(
            hidden_states[self.altup_active_idx]
        )
        all_coefs = keras.ops.reshape(
            self.prediction_coefs(modalities),
            modalities.shape[:-1]
            + (self.altup_num_inputs, self.altup_num_inputs),
        )
        all_coefs = keras.ops.transpose(all_coefs, (0, 1, 3, 2))
        predictions = keras.ops.matmul(
            keras.ops.transpose(hidden_states, (1, 2, 3, 0)), all_coefs
        )
        predictions = keras.ops.transpose(predictions, (3, 0, 1, 2))
        predictions += hidden_states
        return predictions

    def correct(self, predictions, activated):
        modalities = self.compute_router_modalities(activated)
        innovation = activated - predictions[self.altup_active_idx]
        innovation = keras.ops.repeat(
            keras.ops.expand_dims(innovation, 0), self.altup_num_inputs, axis=0
        )
        all_coefs = self.correction_coefs(modalities) + 1.0
        all_coefs = keras.ops.expand_dims(
            keras.ops.transpose(all_coefs, (2, 0, 1)), -1
        )
        corrected = innovation * all_coefs
        corrected += predictions
        return corrected

    def scale_corrected_output(self, corrected):
        return corrected * self.correct_output_scale

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "altup_num_inputs": self.altup_num_inputs,
                "altup_coef_clip": self.altup_coef_clip,
                "altup_active_idx": self.altup_active_idx,
                "rms_norm_eps": self.rms_norm_eps,
                "altup_correct_scale": self.altup_correct_scale,
            }
        )
        return config
