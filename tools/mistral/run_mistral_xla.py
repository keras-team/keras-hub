"""
This script loads a converted Mistral checkpoint (produced by
`export_mistral_to_torch_xla.py`) onto an XLA device and runs text generation.

`mistral-inference` is NOT required.  Only `torch_xla` and `sentencepiece`
are needed:

`pip install torch_xla sentencepiece`

Ensure that your installed `torch_xla` and `torch` versions match.

Sample usage:

```
python tools/mistral/run_mistral_xla.py \
  --checkpoint_dir mistral_xla \
  --prompt "Inception is about"
```

This should produce something like:
```
======================================
PROMPT: Inception is about
RESULT: a thief who steals corporate secrets through the use of
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
    import sentencepiece as spm
except ImportError as e:
    raise ImportError(
        "sentencepiece is required: pip install sentencepiece"
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
# Mistral model — pure PyTorch, no external inference library needed
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
    max_seq_len: int, head_dim: int, rope_theta: float, device
):
    """Pre-compute cos/sin tables for Rotary Position Embeddings."""
    inv_freq = 1.0 / (
        rope_theta
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
            / head_dim
        )
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


class MistralModel(nn.Module):
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
    "mistral_xla",
    "Directory containing `mistral.ckpt`, `params.json`, and"
    " `tokenizer.model`.",
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
    """Generate text on an XLA device from a converted Mistral checkpoint."""
    device = _xla_device()

    if index > 0:
        sys.stdout = open(os.devnull, "w")

    # 1. Load params and build model
    with open(os.path.join(checkpoint_dir, "params.json")) as f:
        params = json.load(f)

    model = MistralModel(params).to(torch.float32).eval()
    state = torch.load(
        os.path.join(checkpoint_dir, "mistral.ckpt"), map_location="cpu"
    )
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    _xla_sync()

    # 2. Tokenise
    sp = spm.SentencePieceProcessor()
    sp.Load(os.path.join(checkpoint_dir, "tokenizer.model"))
    prompt_ids = sp.Encode(prompt, out_type=int)

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

    # 4. Pre-compute RoPE tables
    rope_theta = params.get("rope_theta", 10000.0)
    cos_table, sin_table = _build_rope_cache(
        max_seq_len, head_dim, rope_theta, device
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
        if tok_id == sp.eos_id():
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
    result = sp.Decode(generated_ids)
    print("=" * 38)
    print(f"PROMPT: {prompt}")
    print(f"RESULT: {result}")
    print("=" * 38)


def flag_error_handler():
    for required in ("mistral.ckpt", "params.json", "tokenizer.model"):
        path = os.path.join(FLAGS.checkpoint_dir, required)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: `{path}`. "
                "Please run `export_mistral_to_torch_xla.py` first."
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
