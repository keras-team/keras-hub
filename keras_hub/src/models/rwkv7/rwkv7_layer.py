import keras
from keras import initializers
from keras import ops
from keras.layers import Layer
import warnings


def transpose_head(x, head_first):
    x = ops.cast(x, dtype="float32")
    if head_first:
        return ops.transpose(x, (0, 2, 1, 3))
    else:
        return x


def rnn_generalized_delta_rule(
    r,
    w,
    k,
    v,
    a,
    b,
    initial_state=None,
    output_final_state: bool = True,
    head_first: bool = False,
):
    """
    Implements the generalized delta rule.

    Parameters:
    r: Input tensor.
    w: Weight tensor.
    k, v, a, b: Other input tensors.
    initial_state: Initial state tensor.
    output_final_state: Whether to return the final state.
    head_first: Whether to place the head dimension first during computation.

    Returns:
    Final state if output_final_state is True, otherwise only the output.
    """
    DTYPE = r.dtype
    B, T, H, N = ops.shape(r)
    r = transpose_head(r, head_first)

    k = transpose_head(k, head_first)

    v = transpose_head(v, head_first)
    a = transpose_head(a, head_first)
    b = transpose_head(b, head_first)
    w = transpose_head(w, head_first)
    w = ops.exp(-ops.exp(w))

    if initial_state is not None:
        state = initial_state
        if ops.shape(state)[0] == 1:
            state = ops.broadcast_to(state, (B, H, N, N))
    else:
        state = ops.zeros((B, H, N, N))
    state = ops.cast(state, "float32")
    out = ops.zeros((B, T, H, N), DTYPE)

    def step(t, inputs):
        state, out = inputs
        kk = ops.reshape(k[:, t, :], (B, H, 1, N))
        rr = ops.reshape(r[:, t, :], (B, H, N, 1))
        vv = ops.reshape(v[:, t, :], (B, H, N, 1))
        aa = ops.reshape(a[:, t, :], (B, H, N, 1))
        bb = ops.reshape(b[:, t, :], (B, H, 1, N))
        state = state * w[:, t, :, None, :] + state @ aa @ bb + vv @ kk
        o = ops.cast((state @ rr), out.dtype)
        out = ops.slice_update(out, [0, t, 0, 0], ops.reshape(o, (B, 1, H, N)))
        return [state, out]

    state, out = ops.fori_loop(0, T, step, [state, out])

    if output_final_state:
        return ops.cast(out, DTYPE), state
    return ops.cast(out, DTYPE)


class TimeShift(Layer):
    def __init__(self, name="time_shift"):
        super(TimeShift, self).__init__(name=name)

    def call(self, inputs, cache_x=None):
        if cache_x is not None:
            x = ops.concatenate([cache_x, inputs], axis=1)
        else:
            x = ops.pad(inputs, [[0, 0], [1, 0], [0, 0]], constant_values=0.0)
        return x[:, :-1, :]

    def compute_output_shape(self, input_shape):
        return input_shape


class RWKV7_ChannelMix(Layer):
    def __init__(self, dim_ffn, kernel_initializer="glorot_uniform", **kwargs):
        super().__init__(**kwargs)
        self.dim_ffn = dim_ffn
        self.kernel_initializer = initializers.get(kernel_initializer)

    def call(self, x, last_cache_x=None, train_mode=True):
        xx = self.time_shift(x, last_cache_x) - x
        if last_cache_x is not None or not train_mode:
            last_cache_x = x[:, -1:]
        k = x + xx * self.x_k
        k = ops.relu(self.key(k)) ** 2
        output = self.value(k)
        if train_mode:
            return output
        return output, last_cache_x

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return input_shape[0]
        return input_shape

    def build(self, input_shape):
        super().build(input_shape)
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        self.x_k = self.add_weight(
            shape=(1, 1, input_shape[-1]),
            name="time_mix_k",
            initializer=self.kernel_initializer,
        )
        self.time_shift = TimeShift()
        self.key = keras.layers.Dense(
            self.dim_ffn,
            use_bias=False,
            name="dense_k",
            kernel_initializer=self.kernel_initializer,
        )
        self.value = keras.layers.Dense(
            input_shape[-1],
            use_bias=False,
            name="dense_v",
            kernel_initializer=self.kernel_initializer,
        )
        self.key.build(input_shape)
        self.value.build([None, None, self.dim_ffn])

    def get_config(self):
        config = {
            "dim_ffn": self.dim_ffn,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GroupNorm(keras.layers.GroupNormalization):
    def call(self, inputs):
        if keras.config.backend() == "torch":
            import torch.nn.functional as F

            return F.group_norm(
                inputs, self.groups, self.gamma, self.beta, self.epsilon
            )
        return super().call(inputs)


class RWKV7_TimeMix(Layer):
    def __init__(
        self,
        hidden_size,
        head_size,
        gate_lora=128,
        mv_lora=32,
        aaa_lora=64,
        decay_lora=64,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.head_size = head_size
        self.hidden_size = hidden_size
        self.n_head = hidden_size // self.head_size
        self.gate_lora = gate_lora
        self.mv_lora = mv_lora
        self.aaa_lora = aaa_lora
        self.decay_lora = decay_lora
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.initial_state = None
        try:
            from rwkv_ops import generalized_delta_rule

            self.RWKV7_OP = generalized_delta_rule
        except ImportError:
            warnings.warn(
                "The 'rwkv_ops' package is not installed. "
                "Falling back to the default (pure-Python) operators, which will be very slow. "
                "Please install 'rwkv_ops' to enable the optimized kernels.",
                UserWarning,
                stacklevel=2,
            )
            self.RWKV7_OP = rnn_generalized_delta_rule

        assert self.hidden_size % self.n_head == 0

    def build(self, input_shape):
        super().build(input_shape)
        if isinstance(input_shape[0], list):
            input_shape = input_shape[0]
        H = self.n_head
        N = self.head_size
        B, T, C = input_shape

        self.x_r = self.add_weight(
            shape=(1, 1, C), name="x_r", initializer=self.kernel_initializer
        )
        self.x_w = self.add_weight(
            shape=(1, 1, C), name="x_w", initializer=self.kernel_initializer
        )
        self.x_k = self.add_weight(
            shape=(1, 1, C), name="x_k", initializer=self.kernel_initializer
        )
        self.x_v = self.add_weight(
            shape=(1, 1, C), name="x_v", initializer=self.kernel_initializer
        )
        self.x_a = self.add_weight(
            shape=(1, 1, C), name="x_a", initializer=self.kernel_initializer
        )
        self.x_g = self.add_weight(
            shape=(1, 1, C), name="x_g", initializer=self.kernel_initializer
        )

        self.w0 = self.add_weight(
            shape=(1, 1, C), name="w0", initializer=self.kernel_initializer
        )
        self.w1 = self.add_weight(
            shape=(C, self.decay_lora),
            name="w1",
            initializer=self.kernel_initializer,
        )
        self.w2 = self.add_weight(
            shape=(self.decay_lora, C),
            name="w2",
            initializer=self.kernel_initializer,
        )

        self.a0 = self.add_weight(
            shape=(1, 1, C), name="a0", initializer=self.kernel_initializer
        )
        self.a1 = self.add_weight(
            shape=(C, self.aaa_lora),
            name="a1",
            initializer=self.kernel_initializer,
        )
        self.a2 = self.add_weight(
            shape=(self.aaa_lora, C),
            name="a2",
            initializer=self.kernel_initializer,
        )

        self.v0 = self.add_weight(
            shape=(1, 1, C), name="v0", initializer=self.kernel_initializer
        )
        self.v1 = self.add_weight(
            shape=(C, self.mv_lora),
            name="v1",
            initializer=self.kernel_initializer,
        )
        self.v2 = self.add_weight(
            shape=(self.mv_lora, C),
            name="v2",
            initializer=self.kernel_initializer,
        )

        self.g1 = self.add_weight(
            shape=(C, self.gate_lora),
            name="g1",
            initializer=self.kernel_initializer,
        )
        self.g2 = self.add_weight(
            shape=(self.gate_lora, C),
            name="g2",
            initializer=self.kernel_initializer,
        )

        self.k_k = self.add_weight(
            shape=(1, 1, C), name="k_k", initializer=self.kernel_initializer
        )
        self.k_a = self.add_weight(
            shape=(1, 1, C), name="k_a", initializer=self.kernel_initializer
        )
        self.r_k = self.add_weight(
            shape=(H, N), name="r_k", initializer=self.kernel_initializer
        )

        self.time_shift = TimeShift()
        self.receptance = keras.layers.Dense(
            C,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name="receptance",
        )
        self.key = keras.layers.Dense(
            C,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name="key",
        )
        self.value = keras.layers.Dense(
            C,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name="value",
        )
        self.output_layer = keras.layers.Dense(
            C,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name="output_layer",
        )
        self.ln_x = GroupNorm(groups=H, epsilon=64e-5)

        self.receptance.build(input_shape)
        self.value.build(input_shape)
        self.key.build(input_shape)
        self.output_layer.build(input_shape)
        self.ln_x.build((None, C))

    def call(
        self,
        x,
        v_first=None,
        padding_mask=None,
        last_cache_x=None,
        cache_state=None,
        rnn_mode=False,
        train_mode=True,
    ):
        if cache_state == None:
            initial_state = self.initial_state
        else:
            initial_state = cache_state
        if padding_mask is not None:
            if ops.ndim(padding_mask) == 2:
                padding_mask = padding_mask[..., None]
            padding_mask = ops.cast(padding_mask, x.dtype)
            x *= padding_mask
        B, T, C = ops.shape(x)
        H = self.n_head
        xx = self.time_shift(x, last_cache_x) - x
        if last_cache_x is not None or not train_mode:
            last_cache_x = x[:, -1:]
        if padding_mask is not None:
            xx *= padding_mask

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = (
            -ops.softplus(
                -(self.w0 + ops.matmul(ops.tanh(ops.matmul(xw, self.w1)), self.w2))
            )
            - 0.5
        )  # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if v_first is None:
            v_first = v
        else:
            v = v + (v_first - v) * ops.sigmoid(
                self.v0 + ops.matmul(ops.matmul(xv, self.v1), self.v2)
            )

        a = ops.sigmoid(
            self.a0 + ops.matmul(ops.matmul(xa, self.a1), self.a2)
        )  # a is "in-context learning rate"
        g = ops.matmul(ops.sigmoid(ops.matmul(xg, self.g1)), self.g2)

        kk = k * self.k_k

        kk = self.normalize(ops.reshape(kk, (B, T, H, -1)))
        kk = ops.reshape(kk, (B, T, C))

        k = k * (1 + (a - 1) * self.k_a)
        if padding_mask is not None:
            w = ops.where(padding_mask, w, -1e9)
        if rnn_mode:
            rwkv7_op = rnn_generalized_delta_rule
        else:
            rwkv7_op = self.RWKV7_OP

        def reshape_and_cast(x, new_shape, dtype="float32"):
            x = ops.reshape(x, new_shape)
            if rnn_mode:
                return x
            return ops.cast(x, dtype)

        x, finnal_state = rwkv7_op(
            reshape_and_cast(r, (B, T, self.n_head, self.head_size)),
            reshape_and_cast(w, (B, T, self.n_head, self.head_size)),
            reshape_and_cast(k, (B, T, self.n_head, self.head_size)),
            reshape_and_cast(v, (B, T, self.n_head, self.head_size)),
            reshape_and_cast(-kk, (B, T, self.n_head, self.head_size)),
            reshape_and_cast(kk * a, (B, T, self.n_head, self.head_size)),
            initial_state=ops.cast(initial_state, "float32")
            if initial_state is not None
            else None,
        )
        x = reshape_and_cast(x, (B, T, C), self.compute_dtype)

        x = ops.reshape(self.ln_x(ops.reshape(x, (B * T, C))), ops.shape(x))

        x = ops.reshape(x, (B, T, C))
        r = ops.reshape(r, (B, T, H, -1))
        k = ops.reshape(k, (B, T, H, -1))
        v = ops.reshape(v, (B, T, C))

        rwkv = ops.sum(r * k * self.r_k, axis=-1, keepdims=True) * ops.reshape(
            v, (B, T, H, -1)
        )

        x = x + ops.reshape(rwkv, (B, T, C))
        x = self.output_layer(x * g)
        if train_mode:
            return x, v_first
        return x, v_first, last_cache_x, finnal_state

    def compute_output_shape(self, input_shape):
        output_shapes = [
            [None, None, self.hidden_size],
            [None, None, self.hidden_size],
        ]
        return output_shapes

    def normalize(
        self,
        x,
        eps: float = 1e-12,
    ):
        # F.normalize like api
        if keras.config.backend() == "torch":
            import torch.nn.functional as F

            return F.normalize(x, dim=-1, p=2.0)
        square_sum = ops.sum(ops.square(x), axis=-1, keepdims=True)
        inv_norm = ops.rsqrt(square_sum + eps)
        inv_norm = ops.maximum(inv_norm, eps)
        return x * inv_norm

    def get_config(self):
        config = {
            "hidden_size": self.hidden_size,
            "head_size": self.head_size,
            "gate_lora": self.gate_lora,
            "mv_lora": self.mv_lora,
            "aaa_lora": self.aaa_lora,
            "decay_lora": self.decay_lora,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RWKV7_Block(Layer):
    def __init__(
        self,
        hidden_size,
        head_size,
        intermediate_dim,
        gate_lora=128,
        mv_lora=32,
        aaa_lora=64,
        decay_lora=64,
        use_initial_norm=False,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.head_size = head_size
        self.hidden_size = hidden_size
        self.gate_lora = gate_lora
        self.mv_lora = mv_lora
        self.aaa_lora = aaa_lora
        self.decay_lora = decay_lora
        self.intermediate_dim = intermediate_dim
        self.use_initial_norm = use_initial_norm
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        super().build(input_shape)
        if self.use_initial_norm:
            self.ln0 = keras.layers.LayerNormalization(epsilon=1e-5, name="init_norm")
            self.ln0.build(input_shape)

        self.ln1 = keras.layers.LayerNormalization(epsilon=1e-5, name="att_norm")
        self.ln1.build(input_shape)

        self.ln2 = keras.layers.LayerNormalization(epsilon=1e-5, name="ffn_norm")
        self.ln2.build(input_shape)

        self.att = RWKV7_TimeMix(
            self.hidden_size,
            self.head_size,
            self.gate_lora,
            self.mv_lora,
            self.aaa_lora,
            self.decay_lora,
            name="RWKV_TIME_MIX",
            kernel_initializer=self.kernel_initializer,
        )
        self.att.build(input_shape)

        self.ffn = RWKV7_ChannelMix(
            self.intermediate_dim,
            name="RWKV_CMIX",
            kernel_initializer=self.kernel_initializer,
        )
        self.ffn.build(input_shape)

    def call(
        self,
        x,
        v_first=None,
        padding_mask=None,
        cache_state=None,
        cache_tmix_x=None,
        cache_cmix_x=None,
        rnn_mode=False,
        train_mode=True,
    ):
        if padding_mask is not None:
            padding_mask = ops.cast(padding_mask, x.dtype)
            padding_mask = ops.expand_dims(padding_mask, axis=-1)
        if self.use_initial_norm:
            x = self.ln0(x)
        if train_mode:
            xx, v_first = self.att(
                self.ln1(x),
                v_first=v_first,
                padding_mask=padding_mask,
                train_mode=train_mode,
            )
            x = x + xx
            xx = self.ln2(x)
            if padding_mask is not None:
                xx = xx * padding_mask
            x = x + self.ffn(xx, train_mode=train_mode)
            return x, v_first
        else:
            xx, v_first, cache_tmix_x, cache_state = self.att.call(
                self.ln1(x),
                v_first=v_first,
                padding_mask=padding_mask,
                last_cache_x=cache_tmix_x,
                cache_state=cache_state,
                rnn_mode=rnn_mode,
                train_mode=train_mode,
            )
            x = x + xx
            xx = self.ln2(x)
            if padding_mask is not None:
                xx = xx * padding_mask
            xx, cache_cmix_x = self.ffn(xx, cache_cmix_x, train_mode=train_mode)
            x = x + xx
            return x, v_first, cache_state, cache_tmix_x, cache_cmix_x

    def compute_output_shape(self, input_shape):
        output_shapes = [
            [None, None, self.hidden_size],
            [None, None, self.hidden_size],
        ]
        return output_shapes

    def get_config(self):
        config = {
            "hidden_size": self.hidden_size,
            "head_size": self.head_size,
            "gate_lora": self.gate_lora,
            "mv_lora": self.mv_lora,
            "aaa_lora": self.aaa_lora,
            "decay_lora": self.decay_lora,
            "intermediate_dim": self.intermediate_dim,
            "use_initial_norm": self.use_initial_norm,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
