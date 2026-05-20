"""
Llama3 model — pure PyTorch, no external inference library needed.

This script loads a converted Llama3 checkpoint (produced by
`export_llama3_to_torch_xla.py`) onto an XLA device and runs text generation.

`transformers` is required only for the BPE tokenizer:
`pip install torch_xla transformers`

Ensure that your installed `torch_xla` and `torch` versions match.

Sample usage:

```
python tools/llama3/run_llama3_xla.py \
    --checkpoint_dir llama3_xla \
    --prompt "The capital of France is"
```

This should produce something like:
```
======================================
PROMPT:  The capital of France is
RESULT:  Paris. The city is home to
======================================
```
"""

import json
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import app
from absl import flags

try:
    from transformers import PreTrainedTokenizerFast
except ImportError as e:
    raise ImportError(
        "transformers is required for the Llama3 BPE tokenizer: "
        "pip install transformers"
    ) from e

try:
    import torch_xla
    import torch_xla.core.xla_model as xm

    # torch_xla >= 2.6 exposes torch_xla.device() / torch_xla.sync();
    # fall back to the legacy xm helpers on older versions.
    _xla_device = getattr(torch_xla, "device", xm.xla_device)
    _xla_sync = getattr(torch_xla, "sync", xm.mark_step)
except ImportError as e:
    raise ImportError(
        "torch_xla is required but could not be imported. "
        "Ensure torch and torch_xla versions match:\n"
        "  pip install torch_xla[tpu]==<torch_version> "
        "-f https://storage.googleapis.com/libtpu-releases/index.html"
    ) from e


# ------------------------------------------------------------------ #
# Llama3 model — pure PyTorch, no external inference library needed
# ------------------------------------------------------------------ #


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        return norm.type_as(x) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dim into the first half."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _build_rope_cache(
    max_seq_len: int,
    head_dim: int,
    rope_theta: float,
    device,
    rope_frequency_adjustment_factor: float = None,
    low_freq_factor: float = None,
    high_freq_factor: float = None,
    pretraining_seq_len: int = None,
):
    """Pre-compute cos/sin tables for Rotary Position Embeddings.

    Supports standard RoPE (Llama 3.0) and the scaled RoPE variant used
    by Llama 3.1+ (rope_type="llama3").
    """
    inv_freq = 1.0 / (
        rope_theta
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
            / head_dim
        )
    )

    # Llama 3.1+ scaled RoPE
    if rope_frequency_adjustment_factor is not None and pretraining_seq_len:
        low_freq_wavelen = pretraining_seq_len / low_freq_factor
        high_freq_wavelen = pretraining_seq_len / high_freq_factor
        wavelens = 2 * math.pi / inv_freq

        new_inv_freq = []
        for wl, freq in zip(wavelens.tolist(), inv_freq.tolist()):
            if wl > low_freq_wavelen:
                new_inv_freq.append(freq / rope_frequency_adjustment_factor)
            elif wl < high_freq_wavelen:
                new_inv_freq.append(freq)
            else:
                # Interpolate smoothly between the two regimes
                smooth = (pretraining_seq_len / wl - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                new_inv_freq.append(
                    (1 - smooth) * freq / rope_frequency_adjustment_factor
                    + smooth * freq
                )
        inv_freq = torch.tensor(
            new_inv_freq, dtype=torch.float32, device=device
        )

    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim // 2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, head_dim)
    return emb.cos(), emb.sin()


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads  # GQA repeat factor

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,  # (batch, 1, dim)
        cos: torch.Tensor,  # (1, head_dim) — RoPE for current position
        sin: torch.Tensor,  # (1, head_dim)
        k_cache: torch.Tensor,  # (batch, max_seq_len, n_kv_heads, head_dim)
        v_cache: torch.Tensor,
        pos: int,  # current sequence position index
    ) -> torch.Tensor:
        bsz = x.shape[0]

        q = self.wq(x).view(bsz, 1, self.n_heads, self.head_dim)
        k = self.wk(x).view(bsz, 1, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, 1, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        cos = cos.view(1, 1, 1, self.head_dim)
        sin = sin.view(1, 1, 1, self.head_dim)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        # Write new k, v into cache
        k_cache[:, pos : pos + 1] = k
        v_cache[:, pos : pos + 1] = v

        # Attend to all cached positions [0 : pos+1]
        k_full = k_cache[:, : pos + 1]  # (batch, pos+1, n_kv_heads, head_dim)
        v_full = v_cache[:, : pos + 1]

        # Expand kv heads for GQA
        if self.n_rep > 1:
            k_full = k_full.repeat_interleave(self.n_rep, dim=2)
            v_full = v_full.repeat_interleave(self.n_rep, dim=2)

        # (batch, n_heads, 1, head_dim) @ (batch, n_heads, head_dim, pos+1)
        q = q.transpose(1, 2)
        k_full = k_full.transpose(1, 2)
        v_full = v_full.transpose(1, 2)

        attn = torch.matmul(q, k_full.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )  # (batch, n_heads, 1, pos+1)
        attn = F.softmax(attn.float(), dim=-1).type_as(q)

        out = torch.matmul(attn, v_full)  # (batch, n_heads, 1, head_dim)
        out = out.transpose(1, 2).contiguous().view(bsz, 1, -1)
        return self.wo(out)  # (batch, 1, dim)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        hidden_dim: int,
        norm_eps: float,
    ):
        super().__init__()
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.attention = Attention(dim, n_heads, n_kv_heads, head_dim)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.feed_forward = FeedForward(dim, hidden_dim)

    def forward(self, x, cos, sin, k_cache, v_cache, pos):
        x = x + self.attention(
            self.attention_norm(x), cos, sin, k_cache, v_cache, pos
        )
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class LlamaModel(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        dim = params["dim"]
        n_heads = params["n_heads"]
        self.n_layers = params["n_layers"]

        self.tok_embeddings = nn.Embedding(params["vocab_size"], dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=params["n_kv_heads"],
                    head_dim=params.get("head_dim", dim // n_heads),
                    hidden_dim=params["hidden_dim"],
                    norm_eps=params.get("norm_eps", 1e-5),
                )
                for _ in range(self.n_layers)
            ]
        )
        self.norm = RMSNorm(dim, eps=params.get("norm_eps", 1e-5))
        self.output = nn.Linear(dim, params["vocab_size"], bias=False)

    def forward(
        self,
        token_id: torch.Tensor,  # (batch,)
        pos: int,
        cos: torch.Tensor,  # (1, head_dim)
        sin: torch.Tensor,
        k_caches: list,
        v_caches: list,
    ) -> torch.Tensor:
        x = self.tok_embeddings(token_id).unsqueeze(1)  # (batch, 1, dim)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, k_caches[i], v_caches[i], pos)
        return self.output(self.norm(x[:, -1, :]))  # (batch, vocab_size)


# ------------------------------------------------------------------ #
# Flags
# ------------------------------------------------------------------ #

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "checkpoint_dir",
    "llama3_xla",
    "Directory containing `llama3.ckpt`, `params.json`, and `tokenizer.json`.",
)
flags.DEFINE_string(
    "prompt",
    "The capital of France is",
    "A test prompt for verifying functionality of the converted model.",
)
flags.DEFINE_integer("output_len", 10, "Number of new tokens to generate.")
flags.DEFINE_float(
    "temperature", 0.0, "Sampling temperature. Use 0.0 for greedy decoding."
)
flags.DEFINE_integer(
    "max_seq_len",
    512,
    "Maximum sequence length (prompt + output). KV caches are pre-allocated "
    "to this size.",
)


# ------------------------------------------------------------------ #
# Generation
# ------------------------------------------------------------------ #


def generate(
    index: int,
    checkpoint_dir: str,
    prompt: str,
    output_len: int,
    temperature: float,
    max_seq_len: int,
):
    """Generate text on an XLA device from a converted Llama3 checkpoint."""
    device = _xla_device()

    if index > 0:
        sys.stdout = open(os.devnull, "w")

    # 1. Load params and build model
    with open(os.path.join(checkpoint_dir, "params.json")) as f:
        params = json.load(f)

    model = LlamaModel(params).to(torch.float32).eval()
    state = torch.load(
        os.path.join(checkpoint_dir, "llama3.ckpt"), map_location="cpu"
    )
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    _xla_sync()

    # 2. Tokenise using the BPE tokenizer saved as tokenizer.json
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(checkpoint_dir, "tokenizer.json")
    )
    prompt_ids = tokenizer.encode(prompt)

    # 3. Pre-allocate KV caches
    n_kv_heads = params["n_kv_heads"]
    head_dim = params.get("head_dim", params["dim"] // params["n_heads"])
    k_caches = [
        torch.zeros(1, max_seq_len, n_kv_heads, head_dim, device=device)
        for _ in range(params["n_layers"])
    ]
    v_caches = [
        torch.zeros(1, max_seq_len, n_kv_heads, head_dim, device=device)
        for _ in range(params["n_layers"])
    ]

    # 4. Pre-compute RoPE tables (with optional Llama3.1+ scaling)
    rope_theta = params.get("rope_theta", 500000.0)
    cos_table, sin_table = _build_rope_cache(
        max_seq_len,
        head_dim,
        rope_theta,
        device,
        rope_frequency_adjustment_factor=params.get(
            "rope_frequency_adjustment_factor"
        ),
        low_freq_factor=params.get("rope_low_freq_factor"),
        high_freq_factor=params.get("rope_high_freq_factor"),
        pretraining_seq_len=params.get("rope_pretraining_sequence_length"),
    )
    _xla_sync()

    # 5. Phase 1 — Prefill: feed prompt tokens one by one into the KV cache
    logits = None
    for pos, tok_id in enumerate(prompt_ids):
        tok = torch.tensor([tok_id], dtype=torch.long, device=device)
        cos = cos_table[pos : pos + 1]
        sin = sin_table[pos : pos + 1]
        with torch.no_grad():
            logits = model(tok, pos, cos, sin, k_caches, v_caches)
        _xla_sync()

    # 6. Phase 2 — Decode: generate `output_len` new tokens
    if temperature == 0.0:
        next_tok = logits.argmax(dim=-1)  # (1,) — first generated token
    else:
        probs = F.softmax(logits / temperature, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)
    _xla_sync()

    generated_ids = []
    pos = len(prompt_ids)
    for _ in range(output_len):
        tok_id = next_tok.item()  # sync point — unavoidable for EOS check
        generated_ids.append(tok_id)
        # Llama3 EOS token ids: 128001 (<|end_of_text|>), 128009 (<|eot_id|>)
        if tok_id in (128001, 128009):
            break

        cos = cos_table[pos : pos + 1]
        sin = sin_table[pos : pos + 1]
        with torch.no_grad():
            logits = model(next_tok, pos, cos, sin, k_caches, v_caches)
        _xla_sync()
        pos += 1

        if temperature == 0.0:
            next_tok = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)
        _xla_sync()

    # 7. Decode and print
    result = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print("=" * 38)
    print(f"PROMPT:  {prompt}")
    print(f"RESULT:  {result}")
    print("=" * 38)


def flag_error_handler():
    for required in ("llama3.ckpt", "params.json", "tokenizer.json"):
        path = os.path.join(FLAGS.checkpoint_dir, required)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: `{path}`. "
                "Please run `export_llama3_to_torch_xla.py` first."
            )


def main(_):
    flag_error_handler()
    generate(
        0,
        FLAGS.checkpoint_dir,
        FLAGS.prompt,
        FLAGS.output_len,
        FLAGS.temperature,
        FLAGS.max_seq_len,
    )


if __name__ == "__main__":
    app.run(main)
