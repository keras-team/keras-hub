import os

import numpy as np
import requests
import torch
from absl import app
from absl import flags

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KERNEL_TYPE"] = "native"


import types

import torch.nn as nn
import torch.nn.functional as F
from keras import ops  # noqa: E402
from modelscope import snapshot_download

from keras_hub.src.models.rwkv7.rwkv7_backbone import RWKV7Backbone
from keras_hub.src.models.rwkv7.rwkv7_causal_lm import RWKV7CausalLM
from keras_hub.src.models.rwkv7.rwkv7_tokenizer import VOCAB_FILENAME
from keras_hub.src.models.rwkv7.rwkv7_tokenizer import RWKVTokenizer

PRESET_MAP = {
    "RWKV7_G1a_0.1B": "rwkv7-g1a-0.1b-20250728-ctx4096.pth",
    "RWKV7_G1a_0.3B": "rwkv7-g1a-0.4b-20250905-ctx4096.pth",
    "RWKV7_G1b_1.5B": "rwkv7-g1b-1.5b-20251202-ctx8192.pth",
    "RWKV7_G1c_2.9B": "rwkv7-g1c-2.9b-20251231-ctx8192.pth",
    "RWKV7_G1c_7.2B": "rwkv7-g1c-7.2b-20251231-ctx8192.pth",
    "RWKV7_G1c_13B": "rwkv7-g1c-13.3b-20251231-ctx8192.pth",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)


# RWKV-v7 official PyTorch implementation
# From https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_demo.py
HEAD_SIZE = 64


def RWKV7_OP(r, w, k, v, a, b):
    """
    Official RWKV-7 core operator.
    Performs the time-mix recurrence with delta-rule based learning.
    """
    DTYPE = r.dtype
    B, T, C = r.size()
    H = C // HEAD_SIZE
    N = HEAD_SIZE
    r = r.view(B, T, H, N).float()
    k = k.view(B, T, H, N).float()
    v = v.view(B, T, H, N).float()
    a = a.view(B, T, H, N).float()
    b = b.view(B, T, H, N).float()

    # Compute decay factor (log-space)
    w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))
    out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)
    state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float)

    # Recurrent inference loop over time
    for t in range(T):
        kk = k[:, t, :].view(B, H, 1, N)
        rr = r[:, t, :].view(B, H, N, 1)
        vv = v[:, t, :].view(B, H, N, 1)
        aa = a[:, t, :].view(B, H, N, 1)
        bb = b[:, t, :].view(B, H, 1, N)
        # State update: decay + delta-rule + residual
        state = state * w[:, t, :, None, :] + state @ aa @ bb + vv @ kk
        # Read-out for current position
        out[:, t, :] = (state @ rr).view(B, H, N)
    return out.view(B, T, C).to(DTYPE)


# RWKV Time-Mix Layer (Attention)
class RWKV_Tmix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        H, N, C = self.n_head, self.head_size, args.n_embd

        # Low-rank adaptation & shift scalars
        self.x_r = nn.Parameter(torch.empty(1, 1, C))
        self.x_w = nn.Parameter(torch.empty(1, 1, C))
        self.x_k = nn.Parameter(torch.empty(1, 1, C))
        self.x_v = nn.Parameter(torch.empty(1, 1, C))
        self.x_a = nn.Parameter(torch.empty(1, 1, C))
        self.x_g = nn.Parameter(torch.empty(1, 1, C))

        # Decay (w) modulation
        self.w0 = nn.Parameter(torch.empty(1, 1, C))
        self.w1 = nn.Parameter(torch.empty(C, D_DECAY_LORA))
        self.w2 = nn.Parameter(torch.empty(D_DECAY_LORA, C))

        # In-context learning rate (a) modulation
        self.a0 = nn.Parameter(torch.empty(1, 1, C))
        self.a1 = nn.Parameter(torch.empty(C, D_AAA_LORA))
        self.a2 = nn.Parameter(torch.empty(D_AAA_LORA, C))

        # Value residual modulation
        self.v0 = nn.Parameter(torch.empty(1, 1, C))
        self.v1 = nn.Parameter(torch.empty(C, D_MV_LORA))
        self.v2 = nn.Parameter(torch.empty(D_MV_LORA, C))

        # Gate modulation
        self.g1 = nn.Parameter(torch.empty(C, D_GATE_LORA))
        self.g2 = nn.Parameter(torch.empty(D_GATE_LORA, C))

        # Normalization & positional factors
        self.k_k = nn.Parameter(torch.empty(1, 1, C))
        self.k_a = nn.Parameter(torch.empty(1, 1, C))
        self.r_k = nn.Parameter(torch.empty(H, N))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        # GroupNorm with very small epsilon for numerical stability
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

    def forward(self, x, v_first=None):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x  # Difference token shift

        # Apply token-shift to each branch
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = (
            -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        )  # Clamp
        k = self.key(xk)
        v = self.value(xv)

        # Value residual: only active on non-first layers
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(
                self.v0 + (xv @ self.v1) @ self.v2
            )

        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)  # In-context LR
        g = torch.sigmoid(xg @ self.g1) @ self.g2  # Gate

        # Normalize keys for stability
        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        # Core recurrence
        x = RWKV7_OP(r, w, k, v, -kk, kk * a).to(r.dtype)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        # Additional local mix (receptance * key * r_k) * value
        x = x + (
            (r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(
                dim=-1, keepdim=True
            )
            * v.view(B, T, H, -1)
        ).view(B, T, C)
        x = self.output(x * g)
        return x, v_first


# RWKV Channel-Mix Layer (Feed-Forward)
class RWKV_CMix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        with torch.no_grad():
            self.x_k = nn.Parameter(torch.empty(1, 1, args.n_embd))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    def forward(self, x):
        xx = self.time_shift(x) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2  # Squared ReLU
        return self.value(k)


# RWKV Building Block (Time-Mix + Channel-Mix + Norms)
class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.ln0 = nn.LayerNorm(args.n_embd) if layer_id == 0 else None
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)

    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)
        xx, v_first = self.att(self.ln1(x), v_first)
        x = x + xx
        x = x + self.ffn(self.ln2(x))
        return x, v_first


# Full RWKV Model
class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        args.dim_att = args.n_embd
        args.dim_ffn = args.n_embd * 4
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList(
            [Block(args, i) for i in range(args.n_layer)]
        )
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def forward(self, idx):
        x = self.emb(idx)
        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first)
        x = self.ln_out(x)
        x = self.head(x)
        return x


def convert_cmix(my_chnnal_mix, weights, i):
    my_chnnal_mix.set_weights(
        [
            weights.pop("blocks.%d.ffn.x_k" % i).reshape([-1]),
            weights.pop("blocks.%d.ffn.key.weight" % i).T,
            weights.pop("blocks.%d.ffn.value.weight" % i).T,
        ]
    )


def convert_tmix(my_time_mix, weights, i):
    if my_time_mix.add_v_first:
        weights_list = [
            weights.pop("blocks.%d.att.x_r" % i).reshape([-1]),
            weights.pop("blocks.%d.att.x_w" % i).reshape([-1]),
            weights.pop("blocks.%d.att.x_k" % i).reshape([-1]),
            weights.pop("blocks.%d.att.x_v" % i).reshape([-1]),
            weights.pop("blocks.%d.att.x_a" % i).reshape([-1]),
            weights.pop("blocks.%d.att.x_g" % i).reshape([-1]),
            weights.pop("blocks.%d.att.w0" % i).reshape([-1]),
            weights.pop("blocks.%d.att.w1" % i),
            weights.pop("blocks.%d.att.w2" % i),
            weights.pop("blocks.%d.att.a0" % i).reshape([-1]),
            weights.pop("blocks.%d.att.a1" % i),
            weights.pop("blocks.%d.att.a2" % i),
            weights.pop("blocks.%d.att.v0" % i).reshape([-1]),
            weights.pop("blocks.%d.att.v1" % i),
            weights.pop("blocks.%d.att.v2" % i),
            weights.pop("blocks.%d.att.g1" % i),
            weights.pop("blocks.%d.att.g2" % i),
            weights.pop("blocks.%d.att.k_k" % i).reshape([-1]),
            weights.pop("blocks.%d.att.k_a" % i).reshape([-1]),
            weights.pop("blocks.%d.att.r_k" % i),
            weights.pop("blocks.%d.att.receptance.weight" % i).T,
            weights.pop("blocks.%d.att.key.weight" % i).T,
            weights.pop("blocks.%d.att.value.weight" % i).T,
            weights.pop("blocks.%d.att.output.weight" % i).T,
            weights.pop("blocks.%d.att.ln_x.weight" % i),
            weights.pop("blocks.%d.att.ln_x.bias" % i),
        ]
    else:
        weights_list = [
            weights.pop("blocks.%d.att.x_r" % i).reshape([-1]),
            weights.pop("blocks.%d.att.x_w" % i).reshape([-1]),
            weights.pop("blocks.%d.att.x_k" % i).reshape([-1]),
            weights.pop("blocks.%d.att.x_v" % i).reshape([-1]),
            weights.pop("blocks.%d.att.x_a" % i).reshape([-1]),
            weights.pop("blocks.%d.att.x_g" % i).reshape([-1]),
            weights.pop("blocks.%d.att.w0" % i).reshape([-1]),
            weights.pop("blocks.%d.att.w1" % i),
            weights.pop("blocks.%d.att.w2" % i),
            weights.pop("blocks.%d.att.a0" % i).reshape([-1]),
            weights.pop("blocks.%d.att.a1" % i),
            weights.pop("blocks.%d.att.a2" % i),
            weights.pop("blocks.%d.att.g1" % i),
            weights.pop("blocks.%d.att.g2" % i),
            weights.pop("blocks.%d.att.k_k" % i).reshape([-1]),
            weights.pop("blocks.%d.att.k_a" % i).reshape([-1]),
            weights.pop("blocks.%d.att.r_k" % i),
            weights.pop("blocks.%d.att.receptance.weight" % i).T,
            weights.pop("blocks.%d.att.key.weight" % i).T,
            weights.pop("blocks.%d.att.value.weight" % i).T,
            weights.pop("blocks.%d.att.output.weight" % i).T,
            weights.pop("blocks.%d.att.ln_x.weight" % i),
            weights.pop("blocks.%d.att.ln_x.bias" % i),
        ]

    my_time_mix.set_weights(weights_list)


def convert_layernorm(myln, weights, ln_id, layer_id):
    myln.set_weights(
        [
            weights.pop("blocks.%d.ln%d.weight" % (layer_id, ln_id)),
            weights.pop("blocks.%d.ln%d.bias" % (layer_id, ln_id)),
        ]
    )


def convert_block(my_block, weights, i):
    convert_cmix(my_block.ffn, weights, i)
    convert_tmix(my_block.att, weights, i)
    if my_block.use_initial_norm:
        convert_layernorm(my_block.ln0, weights, 0, i)
    convert_layernorm(my_block.ln1, weights, 1, i)
    convert_layernorm(my_block.ln2, weights, 2, i)


def convert_backbone(my_backbone, standard_RWKV):
    for i in range(my_backbone.num_layers):
        convert_block(my_backbone.rwkv_layers[i], standard_RWKV.blocks[i])
    my_backbone.token_embedding.set_weights(
        [standard_RWKV.emb.weight.detach().cpu()]
    )
    convert_layernorm(my_backbone.output_layer_norm, standard_RWKV.ln_out)


def convert_rwkv7_checkpoints(weights_path):
    weights = torch.load(weights_path, map_location="cpu")
    weights = {k: v.float().numpy() for k, v in weights.items()}
    w = weights
    global D_DECAY_LORA, D_AAA_LORA, D_MV_LORA, D_GATE_LORA
    D_DECAY_LORA = weights["blocks.0.att.w1"].shape[1]
    D_AAA_LORA = weights["blocks.0.att.a1"].shape[1]
    D_MV_LORA = weights["blocks.1.att.v1"].shape[1]
    D_GATE_LORA = weights["blocks.1.att.g1"].shape[1]
    n_layer = 0
    for k in w.keys():
        layer_id = int(k.split(".")[1]) if ("blocks." in k) else 0
        n_layer = max(n_layer, layer_id + 1)

    config = {
        "hidden_size": w["emb.weight"].shape[1],
        "num_layers": n_layer,
        "intermediate_dim": w["blocks.0.ffn.key.weight"].shape[0],
        "vocabulary_size": 65536,
        "head_size": 64,
        "gate_lora": D_GATE_LORA,
        "mv_lora": D_MV_LORA,
        "aaa_lora": D_AAA_LORA,
        "decay_lora": D_DECAY_LORA,
    }
    my_backbone = RWKV7Backbone(**config)

    # Copy layer-1 value-residual params to layer-0 (compatibility)
    weights["blocks.0.att.v0"] = weights["blocks.1.att.v0"]
    weights["blocks.0.att.v1"] = weights["blocks.1.att.v1"]
    weights["blocks.0.att.v2"] = weights["blocks.1.att.v2"]

    my_backbone.get_layer("token_embedding").set_weights(
        [weights.pop("emb.weight")]
    )
    for i in range(config["num_layers"]):
        my_block = my_backbone.get_layer(f"rwkv_layer_{i}")
        convert_block(my_block, weights, i)

    my_backbone.output_layer_norm.set_weights(
        [
            weights.pop("ln_out.weight"),
            weights.pop("ln_out.bias"),
        ]
    )
    model = RWKV7CausalLM(my_backbone)
    my_backbone.head.set_weights([weights.pop("head.weight").T])
    return model


url = "https://raw.githubusercontent.com/BlinkDL/RWKV-LM/main/RWKV-v7/rwkv_vocab_v20230424.txt"


def main(_):
    if not os.path.exists(FLAGS.preset):
        os.makedirs(FLAGS.preset)

    source_model_name = PRESET_MAP[FLAGS.preset]
    # Download vocabulary file

    vocabs = requests.get(url, timeout=30).text
    with open(
        os.path.join(FLAGS.preset, VOCAB_FILENAME), "w", encoding="utf-8"
    ) as f:
        f.write(vocabs)
    tokenizer = RWKVTokenizer()
    tokenizer.load_assets(FLAGS.preset)

    # Download checkpoint
    download_path = snapshot_download(
        repo_id="RWKV/rwkv7-g1",
        allow_patterns=source_model_name,
    )

    weights_path = os.path.join(download_path, source_model_name)

    # Convert to Keras format
    my_model = convert_rwkv7_checkpoints(weights_path)

    # Re-build PyTorch reference model
    args = types.SimpleNamespace()
    args.n_layer = my_model.backbone.num_layers
    args.n_embd = my_model.backbone.hidden_size
    args.vocab_size = my_model.backbone.vocabulary_size
    args.head_size_a = 64
    args.dim_att = args.n_embd
    args.dim_ffn = my_model.backbone.intermediate_dim

    if os.environ["CUDA_VISIBLE_DEVICES"] != "-1":
        standard_model = RWKV(args).cuda()
    else:
        standard_model = RWKV(args)

    weights = torch.load(weights_path, map_location="cpu")
    # Some parameters are not present in the weights, but this does not matter.
    # This is because these parameters are not used
    standard_model.load_state_dict(weights, strict=False)

    # Sanity check: tokenize & compare outputs
    x = tokenizer(["i love u"])
    x = np.reshape(x, [1, -1])
    keras_input = ops.convert_to_tensor(x, "int32")
    my_output = my_model.predict_on_batch(
        {"token_ids": keras_input, "padding_mask": ops.ones_like(keras_input)}
    )
    xx = torch.from_numpy(x).int()
    if torch.cuda.is_available():
        xx = xx.cuda()
    standard_output = standard_model(xx)

    standard_output = standard_output.cpu().float().detach().numpy()
    my_output = ops.convert_to_numpy(ops.cast(my_output, "float32"))

    try:
        np.testing.assert_allclose(my_output, standard_output, atol=1e-4)
        print("Successfully passed the numerical verification! 🎯✅📊")
    except AssertionError as err:
        print("\n")
        print(err.args[0])
        print("\n")

    # Export final Keras model
    my_model.backbone.save_to_preset(f"./{FLAGS.preset}")
    tokenizer.save_to_preset(f"./{FLAGS.preset}")


# Entry Guard

if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
