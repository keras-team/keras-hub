import math

import torch
from keras import layers
from keras import ops

from keras_hub.src.layers.modeling.rms_normalization import RMSNormalization

attn_impl = "absorb"
rank = 0
block_size = 128


class Embedding(layers.Layer):
    """
    Embedding layer with parallelism support across distributed processes.

    Args:
        vocab_size (int): Vocabulary size.
        dim (int): Embedding dimension.
    """

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.embedding = layers.Embedding(self.vocab_size, self.dim)

    def call(self, x):
        return self.embedding(x)


# This can probably be replaced by a Dense layer.
# Does it transpose the weight?
def linear(x, weight, bias=None):
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.

    Args:
        x: The input tensor.
        weight: The weight tensor.
        bias: The bias tensor to be added. Default is None.

    Returns:
        The result of the linear transformation.

    """
    x = ops.matmul(x, ops.transpose(weight))
    if bias:
        x = x + bias

    return x


class Linear(layers.Layer):
    """
    Custom linear layer with support for optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer.
    """

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = self.add_weight(
            shape=(out_features, in_features), trainable=True, name="weight"
        )

        if bias:
            self.bias = self.add_weight(
                shape=(out_features,), trainable=True, name="bias"
            )
        else:
            self.bias = None

        # Scale was None if weight's element size was != 1
        # It's only =1 in the case of uint8.
        scale_out_features = (out_features + block_size - 1) // block_size
        scale_in_features = (in_features + block_size - 1) // block_size
        self.scale = self.add_weight(
            shape=(scale_out_features, scale_in_features),
            trainable=True,
            name="scale",
        )

    def call(self, x):
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """

    Args:
        in_features (int): Number of input features.
        out_features (int): Total number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        super().__init__(in_features, out_features, bias)

    def call(self, x):
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    """

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias)

    def call(self, x):
        y = linear(x, self.weight)
        if self.bias is not None:
            y += self.bias
        return y


def precompute_freqs_cis(
    qk_rope_head_dim,
    max_seq_len,
    beta_fast,
    beta_slow,
    rope_theta,
    rope_factor,
    original_seq_len,
):
    """
    Precomputes frequency-based complex exponential values for
        rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional
             embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values
            for positional embeddings.
    """
    dim = qk_rope_head_dim
    seqlen = max_seq_len
    beta_fast = beta_fast
    beta_slow = beta_slow
    base = int(rope_theta)
    factor = rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return (
            dim
            * ops.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * ops.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = ops.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = ops.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (ops.arange(dim) - min) / (max - min)
        ramp_func = ops.clip(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (ops.arange(0, dim, 2) / dim))
    if seqlen > original_seq_len:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = ops.arange(seqlen)
    freqs = ops.outer(t, freqs)
    freqs_cis = ops.polar(ops.ones_like(freqs), freqs)
    return freqs_cis


def _view_as_complex(tensor):
    """
    Tensor shaped (M, N, 2) ->
    """
    real_part = tensor[..., 0]
    imag_part = tensor[..., 1]
    return real_part + 1j * imag_part


def _view_as_real(x):
    # Assuming x is a complex tensor, we extract the real and imaginary parts
    real_part = ops.real(x)
    imag_part = ops.imag(x)

    # Stack the real and imaginary parts along the last axis
    return ops.stack((real_part, imag_part), axis=-1)


def apply_rotary_emb(x, freqs_cis):
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional
            embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values
          for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = _view_as_complex(ops.reshape(x, [*x.shape[:-1], -1, 2]))
    freqs_cis = ops.reshape(freqs_cis, [1, x.shape[1], 1, x.shape[-1]])
    y = _view_as_real(x * freqs_cis)
    # flatten on axis 3
    y = ops.reshape(y, x.shape[:3] + (-1,))
    return y


class MLA(layers.Layer):
    """
    Multi-Headed Attention Layer (MLA).

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """

    def __init__(
        self,
        dim,
        n_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        max_seq_len,
        original_seq_len,
        mscale,
        rope_factor,
        max_batch_size,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_local_heads = n_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.max_seq_len = max_seq_len
        self.original_seq_len = original_seq_len
        self.mscale = mscale
        self.rope_factor = rope_factor
        self.max_batch_size = max_batch_size

        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(
                self.dim, self.n_heads * self.qk_head_dim
            )
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNormalization(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(
                self.q_lora_rank, self.n_heads * self.qk_head_dim
            )
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNormalization(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
        )
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim**-0.5
        if self.max_seq_len > self.original_seq_len:
            mscale = 0.1 * self.mscale * math.log(self.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

            if attn_impl == "naive":
                self.k_cache = ops.zeros(
                    shape=(
                        self.max_batch_size,
                        self.max_seq_len,
                        self.n_local_heads,
                        self.qk_head_dim,
                    ),
                )

                self.v_cache = ops.zeros(
                    shape=(
                        self.max_batch_size,
                        self.max_seq_len,
                        self.n_local_heads,
                        self.v_head_dim,
                    ),
                )

            else:
                self.kv_cache = ops.zeros(
                    shape=(
                        self.max_batch_size,
                        self.max_seq_len,
                        self.kv_lora_rank,
                    ),
                )

                self.pe_cache = ops.zeros(
                    shape=(
                        self.max_batch_size,
                        self.max_seq_len,
                        self.qk_rope_head_dim,
                    ),
                )

    def call(self, x, start_pos, freqs_cis, mask):
        """
        Forward pass for the Multi-Headed Attention Layer (MLA).

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis: Precomputed complex exponential
                values for rotary embeddings.
            mask: Mask tensor to
                 exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = ops.shape(x)
        end_pos = start_pos + seqlen
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))

        q = ops.reshape(q, (bsz, seqlen, self.n_local_heads, self.qk_head_dim))
        q_nope, q_pe = ops.split(q, [self.qk_nope_head_dim], axis=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = ops.split(kv, [self.kv_lora_rank], axis=-1)
        k_pe = apply_rotary_emb(ops.expand_dims(k_pe, axis=2), freqs_cis)

        if attn_impl == "naive":
            q = ops.concat([q_nope, q_pe], axis=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = ops.reshape(
                kv,
                [
                    bsz,
                    seqlen,
                    self.n_local_heads,
                    self.qk_nope_head_dim + self.v_head_dim,
                ],
            )
            k_nope, v = ops.split(
                kv, [self.qk_nope_head_dim, self.v_head_dim], axis=-1
            )
            k = ops.concat(
                [k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], axis=-1
            )
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            scores = (
                ops.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos])
                * self.softmax_scale
            )
        else:
            wkv_b = self.wkv_b.weight
            wkv_b = ops.reshape(
                wkv_b, [self.n_local_heads, -1, self.kv_lora_rank]
            )
            q_nope = ops.einsum(
                "bshd,hdc->bshc", q_nope, wkv_b[:, : self.qk_nope_head_dim]
            )
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            scores = (
                ops.einsum(
                    "bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]
                )
                + ops.einsum(
                    "bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos]
                )
            ) * self.softmax_scale
        if mask is not None:
            # scores += mask.unsqueeze(1)
            scores += ops.expand_dims(mask, 1)

        scores = ops.softmax(scores, axis=-1)
        if attn_impl == "naive":
            x = ops.einsum(
                "bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos]
            )
        else:
            x = ops.einsum(
                "bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos]
            )
            x = ops.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim :])
        x = self.wo(ops.reshape(x, [x.shape[0], x.shape[1], -1]))
        return x


class MLP(layers.Layer):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """

    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim)
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def call(self, x):
        return self.w2(ops.silu(self.w1(x)) * self.w3(x))


class Gate(layers.Layer):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight: Learnable weights for the gate.
        bias: Optional bias term for the gate.
    """

    def __init__(
        self,
        dim,
        n_activated_experts,
        n_routed_experts,
        n_expert_groups,
        n_limited_groups,
        score_func,
        route_scale,
    ):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = dim
        self.topk = n_activated_experts
        self.n_groups = n_expert_groups
        self.topk_groups = n_limited_groups
        self.score_func = score_func
        self.route_scale = route_scale

        self.weight = self.add_weight(
            shape=(n_routed_experts, dim),
            trainable=True,
            name="weight",
        )

        if self.dim == 7168:
            self.bias = self.add_weight(
                shape=(n_routed_experts,), trainable=True, name="bias"
            )
        else:
            self.bias = None

    def call(self, x):
        scores = linear(x, self.weight)
        if self.score_func == "softmax":
            scores = ops.softmax(scores, axis=-1)
        else:
            scores = ops.sigmoid(scores)
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = ops.reshape(scores, [x.shape[0], self.n_groups, -1])
            if self.bias is None:
                group_scores = ops.max(scores, axis=-1)
            else:
                # This used to be on dim=-1 - does it make a difference here?
                # TODO: Check.
                group_scores = ops.top_k(scores, 2)[0]
                group_scores = ops.sum(group_scores, axis=-1)

            # This used to be on dim=-1 - does it make a difference here?
            # TODO: Check.
            indices = ops.top_k(group_scores, self.topk_groups)[1]

            mask = ops.ones(shape=[x.shape[0], self.n_groups])
            mask = ops.take_along_axis(scores, indices, axis=1)

            scores = ops.where(ops.expand_dims(mask, -1), scores, float("inf"))
            scores = ops.reshape(scores, [scores.shape[0], -1])
        indices = ops.top_k(scores, self.topk, dim=-1)[1]
        weights = ops.take_along_axis(original_scores, indices, axis=1)
        if self.score_func == "sigmoid":
            weights /= ops.sum(weights, axis=-1, keepdims=True)
        weights *= self.route_scale
        return weights, indices


class Expert(layers.Layer):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1: Linear layer for input-to-hidden transformation.
        w2: Linear layer for hidden-to-output transformation.
        w3: Additional linear layer for feature transformation.
    """

    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def call(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(ops.silu(self.w1(x)) * self.w3(x))


class MoE(layers.Layer):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """

    def __init__(
        self,
        dim,
        n_routed_experts,
        n_activated_experts,
        n_expert_groups,
        n_limited_groups,
        score_func,
        route_scale,
        moe_inter_dim,
        n_shared_experts,
    ):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.n_local_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(
            dim,
            n_activated_experts,
            n_routed_experts,
            n_expert_groups,
            n_limited_groups,
            score_func,
            route_scale,
        )
        self.experts = [
            Expert(dim, moe_inter_dim)
            if self.experts_start_idx <= i < self.experts_end_idx
            else None
            for i in range(self.n_routed_experts)
        ]
        self.shared_experts = MLP(dim, n_shared_experts * moe_inter_dim)

    def call(self, x):
        shape = ops.shape(x)
        x = ops.reshape(x, [-1, self.dim])
        weights, indices = self.gate(x)
        y = ops.zeros_like(x)
        counts = ops.bincount(
            ops.reshape(
                indices,
                [
                    -1,
                ],
            ),
            minlength=self.n_routed_experts,
        ).tolist()  # Does this work for all backends?
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        res = y + z
        return ops.reshape(res, shape)


class Block(layers.Layer):
    """
    Transformer block combining attention and feed-forward layers.

    Attributes:
        attn: Attention layer (MLA).
        ffn: Feed-forward network (MLP or MoE).
        attn_norm: Layer normalization for attention.
        ffn_norm: Layer normalization for feed-forward network.
    """

    def __init__(
        self,
        layer_id: int,
        dim,
        n_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        inter_dim,
        n_dense_layers,
        n_routed_experts,
        n_activated_experts,
        n_expert_groups,
        n_limited_groups,
        score_func,
        route_scale,
        moe_inter_dim,
        n_shared_experts,
        max_seq_len,
        original_seq_len,
        mscale,
        rope_factor,
        max_batch_size,
    ):
        super().__init__()
        self.attn = MLA(
            dim,
            n_heads,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            max_seq_len,
            original_seq_len,
            mscale,
            rope_factor,
            max_batch_size,
        )
        self.ffn = (
            MLP(dim, inter_dim)
            if layer_id < n_dense_layers
            else MoE(
                dim,
                n_routed_experts,
                n_activated_experts,
                n_expert_groups,
                n_limited_groups,
                score_func,
                route_scale,
                moe_inter_dim,
                n_shared_experts,
            )
        )
        self.attn_norm = RMSNormalization(input_dim=dim)
        self.ffn_norm = RMSNormalization(input_dim=dim)

    def call(self, x, start_pos: int, freqs_cis, mask):
        """
        Forward pass for the Transformer block.

        Args:
            x: Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis: Precomputed complex exponential
                values for rotary embeddings.
            mask: Mask tensor to exclude certain
                positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        x = x + self.attn(
            self.attn_norm(x),
            start_pos=start_pos,
            freqs_cis=freqs_cis,
            mask=mask,
        )
        x = x + self.ffn(self.ffn_norm(x))
        return x
