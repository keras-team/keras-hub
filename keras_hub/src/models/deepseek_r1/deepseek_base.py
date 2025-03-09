import logging
import math
import time
from dataclasses import dataclass
from typing import Literal

# from keras.layers import RMSNormalization
import keras
import torch
from keras import layers
from keras import ops
from tqdm import tqdm

# TODO: Replace with keras.layers.RMSNormalization
# when https://github.com/keras-team/keras/pull/20911 is merged
from keras_hub.src.layers.modeling.rms_normalization import RMSNormalization

logging.basicConfig(level=logging.INFO)


world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"


@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """

    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    # n_layers: int = 27
    n_layers: int = 1
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.0
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0


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
    This function supports specialized implementations based on quantization
    and tensor formats.

    Args:
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor. It may be quantized and
            requires dequantization for certain cases.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.

    Returns:
        torch.Tensor: The result of the linear transformation, which may involve
        quantization-aware computations depending on the input parameters.

    Notes:
        - If `weight` is quantized (e.g., `element_size() == 1`), a dequantized version
          is used for computation.
        - If `gemm_impl == "bf16"`, dequantization and a `bf16` GEMM operation are applied.
        - For other cases, the function applies quantization to `x` and uses `fp8_gemm` for computation.
    """
    # return F.linear(x, weight, bias)
    x = ops.matmul(x, ops.transpose(weight))
    if bias:
        x = x + bias

    return x


class Linear(layers.Layer):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
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
        self.scale = None

    def call(self, x):
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """
    Linear layer with column parallelism, splitting output features across distributed processes.

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
        dtype=None,
    ):
        assert out_features % world_size == 0, (
            f"Output features must be divisible by world size (world_size={world_size})"
        )
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias)

    def call(self, x):
        """
        Forward pass for column parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with column-parallel computation.
        """
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    """
    Linear layer with row parallelism, splitting input features across distributed processes.

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        assert in_features % world_size == 0, (
            f"Input features must be divisible by world size (world_size={world_size})"
        )
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias)

    def call(self, x):
        """
        Forward pass for row parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """
        y = linear(x, self.weight)
        if self.bias is not None:
            y += self.bias
        return y


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
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
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

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
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, args.original_seq_len
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

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

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
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

            if attn_impl == "naive":
                self.kv_cache = ops.zeros(
                    shape=(
                        args.max_batch_size,
                        args.max_seq_len,
                        self.n_local_heads,
                        self.qk_head_dim,
                    ),
                )

                self.pe_cache = ops.zeros(
                    shape=(
                        args.max_batch_size,
                        args.max_seq_len,
                        self.n_local_heads,
                        self.v_head_dim,
                    ),
                )

            else:
                self.kv_cache = ops.zeros(
                    shape=(
                        args.max_batch_size,
                        args.max_seq_len,
                        self.kv_lora_rank,
                    ),
                )

                self.pe_cache = ops.zeros(
                    shape=(
                        args.max_batch_size,
                        args.max_seq_len,
                        self.qk_rope_head_dim,
                    ),
                )

    def call(self, x, start_pos, freqs_cis, mask):
        """
        Forward pass for the Multi-Headed Attention Layer (MLA).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential
                values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to
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
            # wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
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
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
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
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """

    def __init__(self, args: ModelArgs):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale

        self.weight = self.add_weight(
            shape=(args.n_routed_experts, args.dim),
            trainable=True,
            name="weight",
        )

        if self.dim == 7168:
            self.bias = self.add_weight(
                shape=(args.n_routed_experts,), trainable=True, name="bias"
            )
        else:
            self.bias = None

    def call(self, x):
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        scores = linear(x, self.weight)
        if self.score_func == "softmax":
            scores = ops.softmax(scores, axis=-1)
        else:
            scores = ops.sigmoid(scores)
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            # scores = scores.view(x.size(0), self.n_groups, -1)
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

            # mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            mask = ops.ones(shape=[x.shape[0], self.n_groups])
            mask = ops.take_along_axis(scores, indices, axis=1)

            # scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
            scores = ops.where(ops.expand_dims(mask, -1), scores, float("inf"))
            scores = ops.reshape(scores, [scores.shape[0], -1])  # flatten(1)
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
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
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
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
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

    def __init__(self, args: ModelArgs):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, (
            f"Number of experts must be divisible by world size (world_size={world_size})"
        )
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts = [
            Expert(args.dim, args.moe_inter_dim)
            if self.experts_start_idx <= i < self.experts_end_idx
            else None
            for i in range(self.n_routed_experts)
        ]
        self.shared_experts = MLP(
            args.dim, args.n_shared_experts * args.moe_inter_dim
        )

    def call(self, x):
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
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
        ).tolist()  # would this work?
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
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelArgs): Model arguments containing block parameters.
        """
        super().__init__()
        self.attn = MLA(args)
        self.ffn = (
            MLP(args.dim, args.inter_dim)
            if layer_id < args.n_dense_layers
            else MoE(args)
        )
        self.attn_norm = RMSNormalization(input_dim=args.dim)
        self.ffn_norm = RMSNormalization(input_dim=args.dim)

    def call(self, x, start_pos: int, freqs_cis, mask):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential
                values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain
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


class Transformer(keras.Model):
    """
    Transformer model with positional embeddings, multiple layers,
        and output projection.

    Attributes:
        max_seq_len (int): Maximum sequence length for the transformer.
        embed (nn.Module): Embedding layer for input tokens.
        layers (torch.nn.ModuleList): List of transformer blocks.
        norm (nn.Module): Layer normalization applied after all blocks.
        head (nn.Module): Output projection layer mapping to vocabulary size.
        freqs_cis (torch.Tensor): Precomputed complex
            exponential values for rotary embeddings.
    """

    def __init__(self, args: ModelArgs, name="DeepSeekV3"):
        """
        Initializes the Transformer model.

        Args:
            args (ModelArgs): Model arguments containing transformer parameters.
        """
        global world_size, rank
        world_size = 1
        rank = 0
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.embed = Embedding(args.vocab_size, args.dim)
        self.blocks = []
        for layer_id in range(args.n_layers):
            logging.info(f"Layer {layer_id}")
            self.blocks.append(Block(layer_id, args))
        self.norm = RMSNormalization(args.dim)
        self.head = ColumnParallelLinear(args.dim, args.vocab_size)
        self.freqs_cis = precompute_freqs_cis(args)

    def call(self, tokens, start_pos: int = 0):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape
                (batch_size, seq_len).
            start_pos (int, optional): Starting position in the sequence
                for rotary embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        seqlen = ops.shape(tokens)[1]
        h = self.embed(tokens)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        mask = None
        if seqlen > 1:
            mask = ops.full((seqlen, seqlen), float("-inf"))
            mask = ops.triu(mask, k=1)
        for layer in self.blocks:
            h = layer(h, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask)
        h = self.norm(h)[:, -1]
        logits = self.head(h)
        return logits


if __name__ == "__main__":
    keras.config.set_dtype_policy("mixed_float16")
    args = ModelArgs()
    x = keras.random.randint((1, 128), 0, args.vocab_size)
    logging.info("Creating model...")
    model = Transformer(args)
    logging.info("Running dummy input...")
    outs = model(x)
    logging.info(f"{model.summary()}")
    logging.info(
        f"Output size for dummy input (shape of (1, 128)): {outs.size()}"
    )

    total_tokens_generated = 0
    total_generation_time = 0.0

    steps = 10
    logging.info(f"Generating {steps} tokens sequentially")
    x = keras.random.randint((1, 128), 0, args.vocab_size, seed=42)

    for i in tqdm(range(steps)):
        start_time = time.time()
        outs = model(x)
        res_token = outs.argmax(1).unsqueeze(0)
        x = ops.concatenate([x, res_token], 1)
        end_time = time.time() - start_time
        total_generation_time += end_time
        total_tokens_generated += 1

    tokens_per_second = total_tokens_generated / total_generation_time
    logging.info(f"Total tokens generated: {total_tokens_generated}")
    logging.info(f"Total generation time: {total_generation_time:.2f} seconds")
    logging.info(f"Tokens per second: {tokens_per_second:.2f}")
