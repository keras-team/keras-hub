"""Convert diffusers `FluxTransformer2DModel` checkpoints to `FluxBackbone`.

The weights are read from the `transformer/` folder of a diffusers pipeline
repo, such as `hf://black-forest-labs/FLUX.1-schnell`, through
`DiffusersPresetLoader`. The porting helpers here are shared with FLUX.2
(`convert_flux2.py`).
"""

import collections

import numpy as np

from keras_hub.src.models.flux.flux_backbone import FluxBackbone
from keras_hub.src.utils.preset_utils import SAFETENSOR_FILE
from keras_hub.src.utils.preset_utils import get_file

try:
    import safetensors
except ImportError:
    safetensors = None

backbone_cls = FluxBackbone


def checkpoint_keys(loader):
    """Return the names of all tensors in the checkpoint read by `loader`."""
    if loader.safetensor_config is not None:
        return set(loader.safetensor_config["weight_map"])
    path = get_file(loader.preset, loader.fname or SAFETENSOR_FILE)
    with safetensors.safe_open(path, framework="np") as file:
        return set(file.keys())


class CheckedLoader:
    """Reads tensors by exact name and checks that a conversion is complete.

    `SafetensorLoader` resolves a missing name to any name that ends with
    it, so a missing `transformer_blocks.0.attn.to_q.weight` would silently
    load `single_transformer_blocks.0.attn.to_q.weight`. This wrapper only
    reads names that exist verbatim, and `check_complete` fails unless every
    weight of the model was assigned exactly once and every checkpoint tensor
    was read, so a missing, renamed or extra tensor fails the conversion
    instead of leaving a randomly initialized weight behind.

    Args:
        loader: A `SafetensorLoader`.
    """

    def __init__(self, loader):
        self.loader = loader
        self.keys = checkpoint_keys(loader)
        self.read = set()
        self.assigned = collections.Counter()

    def get_tensor(self, key):
        if key not in self.keys:
            raise ValueError(f"The checkpoint has no tensor `{key}`.")
        self.read.add(key)
        return self.loader.get_tensor(key)

    def assign(self, variable, value):
        # Checkpoints are usually bfloat16; cast on the host.
        variable.assign(np.asarray(value).astype(variable.dtype))
        self.assigned[id(variable)] += 1

    def check_complete(self, model, ignore=()):
        """Fail unless the mapping was complete and one-to-one."""
        never = [w.path for w in model.weights if id(w) not in self.assigned]
        twice = [w.path for w in model.weights if self.assigned[id(w)] > 1]
        unused = sorted(self.keys - self.read - set(ignore))
        problems = []
        if never:
            problems.append(f"{len(never)} weights never assigned: {never[:8]}")
        if twice:
            problems.append(f"{len(twice)} weights assigned twice: {twice[:8]}")
        if unused:
            problems.append(
                f"{len(unused)} checkpoint tensors never read: {unused[:8]}"
            )
        if problems:
            raise ValueError(
                f"The checkpoint does not match `{type(model).__name__}`:\n"
                + "\n".join(f"  - {problem}" for problem in problems)
            )


def port_dense(loader, dense, key):
    """Port a PyTorch `nn.Linear` into a Keras `Dense`.

    PyTorch stores `[out_features, in_features]`; Keras wants
    `[in_features, out_features]`.
    """
    loader.assign(dense.kernel, loader.get_tensor(f"{key}.weight").T)
    if dense.bias is not None:
        loader.assign(dense.bias, loader.get_tensor(f"{key}.bias"))


def port_fused_dense(loader, dense, keys):
    """Port several `nn.Linear`s, stacked along the output, into one `Dense`.

    diffusers keeps `to_q`, `to_k` and `to_v` (and, for FLUX.1 single-stream
    blocks, `proj_mlp`) separate, where FLUX fuses them into one projection.
    """
    kernels = [loader.get_tensor(f"{key}.weight").T for key in keys]
    loader.assign(dense.kernel, np.concatenate(kernels, axis=-1))
    if dense.bias is not None:
        biases = [loader.get_tensor(f"{key}.bias") for key in keys]
        loader.assign(dense.bias, np.concatenate(biases, axis=-1))


def port_norm_out(loader, dense, key):
    """Port the projection of diffusers' `AdaLayerNormContinuous`.

    diffusers splits its output as `(scale, shift)`; the FLUX final layers
    split it as `(shift, scale)`, so the two halves are swapped.
    """

    def swap_halves(x):
        scale, shift = np.split(x, 2, axis=-1)
        return np.concatenate([shift, scale], axis=-1)

    kernel = loader.get_tensor(f"{key}.weight").T
    loader.assign(dense.kernel, swap_halves(kernel))
    if dense.bias is not None:
        loader.assign(dense.bias, swap_halves(loader.get_tensor(f"{key}.bias")))


def port_qk_norm(loader, qk_norm, query_key, key_key):
    """Port the RMSNorm scales applied to the queries and keys."""
    query_scale = loader.get_tensor(f"{query_key}.weight")
    loader.assign(qk_norm.query_norm.scale, query_scale)
    loader.assign(
        qk_norm.key_norm.scale, loader.get_tensor(f"{key_key}.weight")
    )


def port_mlp_embedder(loader, embedder, key):
    """Port a diffusers `TimestepEmbedding` or `PixArtAlphaTextProjection`."""
    port_dense(loader, embedder.input_layer, f"{key}.linear_1")
    port_dense(loader, embedder.output_layer, f"{key}.linear_2")


def convert_backbone_config(hf_config):
    """Convert a diffusers `FluxTransformer2DModel` config to kwargs.

    Structural keys are read directly, so a missing one fails here, with its
    name, instead of building a model of the wrong shape.
    """
    if hf_config["patch_size"] != 1:
        raise ValueError(
            "`FluxBackbone` takes latents already packed into 2x2 patches, "
            "so the transformer must have `patch_size=1`. Received: "
            f"patch_size={hf_config['patch_size']}"
        )
    input_channels = hf_config["in_channels"]
    num_heads = hf_config["num_attention_heads"]
    return {
        "input_channels": input_channels,
        "hidden_size": num_heads * hf_config["attention_head_dim"],
        # diffusers hardcodes the MLP ratio, the RoPE base frequency and
        # the attention biases of FLUX.1.
        "mlp_ratio": 4.0,
        "num_heads": num_heads,
        "depth": hf_config["num_layers"],
        "depth_single_blocks": hf_config["num_single_layers"],
        # Configs written before diffusers added this key, including
        # FLUX.1 [schnell]'s, use diffusers' default.
        "axes_dim": list(hf_config.get("axes_dims_rope", (16, 56, 56))),
        "theta": 10_000,
        "use_bias": True,
        "guidance_embed": hf_config["guidance_embeds"],
        # Sequence axes stay dynamic so a preset built from this config can
        # run at any resolution and prompt length.
        "image_shape": (None, input_channels),
        "text_shape": (None, hf_config["joint_attention_dim"]),
        "image_ids_shape": (None, 3),
        "text_ids_shape": (None, 3),
        "y_shape": (hf_config["pooled_projection_dim"],),
    }


def _port_double_block(loader, block, prefix):
    port_dense(
        loader, block.image_mod.linear_projection, f"{prefix}.norm1.linear"
    )
    port_dense(
        loader,
        block.text_mod.linear_projection,
        f"{prefix}.norm1_context.linear",
    )
    attn = f"{prefix}.attn"
    port_fused_dense(
        loader,
        block.image_qkv,
        [f"{attn}.to_q", f"{attn}.to_k", f"{attn}.to_v"],
    )
    port_fused_dense(
        loader,
        block.text_qkv,
        [f"{attn}.add_q_proj", f"{attn}.add_k_proj", f"{attn}.add_v_proj"],
    )
    port_qk_norm(
        loader, block.image_attn_norm, f"{attn}.norm_q", f"{attn}.norm_k"
    )
    port_qk_norm(
        loader,
        block.text_attn_norm,
        f"{attn}.norm_added_q",
        f"{attn}.norm_added_k",
    )
    port_dense(loader, block.image_attn_proj, f"{attn}.to_out.0")
    port_dense(loader, block.text_attn_proj, f"{attn}.to_add_out")
    # `image_mlp` and `text_mlp` are `Sequential(Dense, GELU, Dense)`.
    port_dense(loader, block.image_mlp.layers[0], f"{prefix}.ff.net.0.proj")
    port_dense(loader, block.image_mlp.layers[2], f"{prefix}.ff.net.2")
    port_dense(
        loader, block.text_mlp.layers[0], f"{prefix}.ff_context.net.0.proj"
    )
    port_dense(loader, block.text_mlp.layers[2], f"{prefix}.ff_context.net.2")


def _port_single_block(loader, block, prefix):
    port_dense(
        loader, block.modulation.linear_projection, f"{prefix}.norm.linear"
    )
    # `linear1` is `[q, k, v, mlp]`.
    attn = f"{prefix}.attn"
    port_fused_dense(
        loader,
        block.linear1,
        [f"{attn}.to_q", f"{attn}.to_k", f"{attn}.to_v", f"{prefix}.proj_mlp"],
    )
    port_qk_norm(loader, block.norm, f"{attn}.norm_q", f"{attn}.norm_k")
    port_dense(loader, block.linear2, f"{prefix}.proj_out")


def convert_weights(backbone, loader):
    """Port diffusers `FluxTransformer2DModel` weights into `backbone`.

    Fails unless every backbone weight is assigned exactly once and every
    checkpoint tensor is read.

    Args:
        backbone: A `FluxBackbone`.
        loader: A `SafetensorLoader` over the diffusers `transformer/`
            weights.
    """
    if not isinstance(backbone, FluxBackbone):
        raise ValueError(
            "The provided backbone must be an instance of FluxBackbone. "
            f"Received: {type(backbone)}"
        )
    loader = CheckedLoader(loader)
    port_dense(loader, backbone.image_input_embedder, "x_embedder")
    port_dense(loader, backbone.text_input_embedder, "context_embedder")
    embedders = "time_text_embed"
    port_mlp_embedder(
        loader,
        backbone.time_input_embedder,
        f"{embedders}.timestep_embedder",
    )
    port_mlp_embedder(
        loader, backbone.vector_embedder, f"{embedders}.text_embedder"
    )
    if backbone.guidance_embed:
        port_mlp_embedder(
            loader,
            backbone.guidance_input_embedder,
            f"{embedders}.guidance_embedder",
        )
    for i, block in enumerate(backbone.double_blocks):
        _port_double_block(loader, block, f"transformer_blocks.{i}")
    for i, block in enumerate(backbone.single_blocks):
        _port_single_block(loader, block, f"single_transformer_blocks.{i}")
    # `adaLN_modulation` is `Sequential(SiLU, Dense)`.
    port_norm_out(
        loader,
        backbone.final_layer.adaLN_modulation.layers[1],
        "norm_out.linear",
    )
    port_dense(loader, backbone.final_layer.linear, "proj_out")
    loader.check_complete(backbone)
