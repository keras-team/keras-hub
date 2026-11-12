import ml_dtypes  # noqa: F401
import numpy as np

from keras_hub.src.models.unlimited_ocr.unlimited_ocr_backbone import (
    UnlimitedOCRBackbone,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_vision_encoder import (
    UnlimitedOCRVisionEncoder,
)
from keras_hub.src.utils.preset_utils import check_file_exists
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = UnlimitedOCRBackbone

_DEFAULT_IMAGE_TOKEN_ID = 128815
_DEFAULT_ROPE_THETA = 10000.0


def _get_text_config(transformers_config):
    # The native class writes `text_config`. The original checkpoint writes
    # `language_config`.
    if "text_config" in transformers_config:
        return transformers_config["text_config"]
    return transformers_config["language_config"]


def _get_vision_configs(transformers_config):
    """Returns the SAM and CLIP settings in one flat form.

    The native class writes `vision_config.sam_config` and
    `vision_config.encoder_config`. The original checkpoint writes
    `vision_config.width.{sam_vit_b,clip-l-14-224}` and hard-codes the other
    values in the modeling code.
    """
    vision_config = transformers_config["vision_config"]
    if "sam_config" in vision_config:
        sam = vision_config["sam_config"]
        clip = vision_config["encoder_config"]
        return {
            "sam_hidden_dim": sam["hidden_size"],
            "sam_num_layers": sam["num_hidden_layers"],
            "sam_num_heads": sam["num_attention_heads"],
            "sam_mlp_dim": sam["mlp_dim"],
            "sam_output_channels": sam["output_channels"],
            "sam_patch_size": sam["patch_size"],
            "sam_image_size": sam["image_size"],
            "sam_window_size": sam["window_size"],
            "sam_global_attn_indexes": tuple(sam["global_attn_indexes"]),
            "sam_downsample_channels": tuple(sam["downsample_channels"]),
            "clip_hidden_dim": clip["hidden_size"],
            "clip_num_layers": clip["num_hidden_layers"],
            "clip_num_heads": clip["num_attention_heads"],
            "clip_intermediate_dim": clip["intermediate_size"],
            "clip_image_size": clip["image_size"],
            "clip_patch_size": clip["patch_size"],
        }

    width = vision_config["width"]
    sam = width["sam_vit_b"]
    clip = width["clip-l-14-224"]
    return {
        "sam_hidden_dim": sam["width"],
        "sam_num_layers": sam["layers"],
        "sam_num_heads": sam["heads"],
        # The modeling code hard-codes these SAM values (`deepencoder.py`).
        "sam_mlp_dim": sam["width"] * 4,
        "sam_output_channels": 256,
        "sam_patch_size": 16,
        "sam_image_size": vision_config["image_size"],
        "sam_window_size": 14,
        "sam_global_attn_indexes": tuple(sam["global_attn_indexes"]),
        "sam_downsample_channels": tuple(sam["downsample_channels"]),
        "clip_hidden_dim": clip["width"],
        "clip_num_layers": clip["layers"],
        "clip_num_heads": clip["heads"],
        "clip_intermediate_dim": clip["width"] * 4,
        "clip_image_size": clip["image_size"],
        "clip_patch_size": clip["patch_size"],
    }


def _get_num_dense_layers(text_config):
    mlp_layer_types = text_config.get("mlp_layer_types")
    if mlp_layer_types is not None:
        num_dense_layers = 0
        for layer_type in mlp_layer_types:
            if layer_type != "dense":
                break
            num_dense_layers += 1
        return num_dense_layers
    return text_config.get("first_k_dense_replace", 1)


def _get_rope_theta(text_config):
    rope_theta = text_config.get("rope_parameters", {}).get("rope_theta")
    if rope_theta is None:
        rope_theta = text_config.get("rope_theta", _DEFAULT_ROPE_THETA)
    return rope_theta


def _get_sliding_window(transformers_config, text_config):
    for config in (text_config, transformers_config):
        for key in ("sliding_window", "sliding_window_size"):
            if config.get(key) is not None:
                return config[key]
    return 128


def convert_backbone_config(transformers_config):
    text_config = _get_text_config(transformers_config)
    vision_encoder = UnlimitedOCRVisionEncoder(
        **_get_vision_configs(transformers_config)
    )
    # The tile size is not in the model config. The backbone default of 640
    # matches the released checkpoint. Pass `local_image_size` to
    # `from_preset()` for another tile size.
    global_image_size = vision_encoder.sam_image_size
    return {
        "vocabulary_size": text_config["vocab_size"],
        "num_layers": text_config["num_hidden_layers"],
        "num_query_heads": text_config["num_attention_heads"],
        "hidden_dim": text_config["hidden_size"],
        "intermediate_dim": text_config["intermediate_size"],
        "moe_intermediate_dim": text_config["moe_intermediate_size"],
        "num_shared_experts": text_config["n_shared_experts"],
        "num_experts": text_config["n_routed_experts"],
        "top_k": text_config["num_experts_per_tok"],
        "vision_encoder": vision_encoder,
        "num_dense_layers": _get_num_dense_layers(text_config),
        "routed_scaling_factor": text_config.get("routed_scaling_factor", 1.0),
        "rope_max_wavelength": _get_rope_theta(text_config),
        "layer_norm_epsilon": text_config.get("rms_norm_eps", 1e-6),
        "sliding_window": _get_sliding_window(transformers_config, text_config),
        "image_token_id": transformers_config.get(
            "image_token_id", _DEFAULT_IMAGE_TOKEN_ID
        ),
        "global_image_size": global_image_size,
    }


def load_image_converter_config(preset, transformers_config):
    preprocessor_config = load_json(preset, "preprocessor_config.json")
    config = {
        "global_size": preprocessor_config["size"]["height"],
        "tile_size": preprocessor_config["tile_size"],
        "min_tiles": preprocessor_config["min_patches"],
        "max_tiles": preprocessor_config["max_patches"],
        "crop_to_patches": preprocessor_config["crop_to_patches"],
        "background_color": preprocessor_config["background_color"],
        "image_mean": preprocessor_config["image_mean"],
        "image_std": preprocessor_config["image_std"],
        "rescale_factor": preprocessor_config["rescale_factor"],
    }
    if check_file_exists(preset, "processor_config.json"):
        processor_config = load_json(preset, "processor_config.json")
        config["patch_size"] = processor_config["patch_size"]
        config["downsample_ratio"] = processor_config["downsample_ratio"]
    return config


def convert_weights(backbone, loader, transformers_config):
    def to_float32(hf_tensor):
        return hf_tensor.astype(np.float32)

    def port_plain(keras_variable, hf_weight_key):
        loader.port_weight(
            keras_variable=keras_variable,
            hf_weight_key=hf_weight_key,
            hook_fn=lambda hf_tensor, _: to_float32(hf_tensor),
        )

    def port_linear(keras_dense, hf_prefix, bias=True):
        # `nn.Linear` stores `(out, in)`. The Keras kernel is `(in, out)`.
        loader.port_weight(
            keras_variable=keras_dense.kernel,
            hf_weight_key=f"{hf_prefix}.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(to_float32(hf_tensor)),
        )
        if bias:
            port_plain(keras_dense.bias, f"{hf_prefix}.bias")

    def port_conv(keras_conv, hf_prefix, bias=True):
        # `nn.Conv2d` stores `(O, I, H, W)`. The Keras kernel is `(H, W, I, O)`.
        loader.port_weight(
            keras_variable=keras_conv.kernel,
            hf_weight_key=f"{hf_prefix}.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                to_float32(hf_tensor), axes=(2, 3, 1, 0)
            ),
        )
        if bias:
            port_plain(keras_conv.bias, f"{hf_prefix}.bias")

    def port_norm(keras_norm, hf_prefix):
        port_plain(keras_norm.gamma, f"{hf_prefix}.weight")
        port_plain(keras_norm.beta, f"{hf_prefix}.bias")

    def reshape_kernel(hf_tensor, keras_shape):
        return np.reshape(np.transpose(to_float32(hf_tensor)), keras_shape)

    # === Text decoder ===
    token_embedding = backbone.get_layer("token_embedding")
    port_plain(token_embedding.embeddings, "model.embed_tokens.weight")
    # The checkpoint does not tie the output projection to the embeddings.
    loader.port_weight(
        keras_variable=token_embedding.reverse_embeddings,
        hf_weight_key="lm_head.weight",
        hook_fn=lambda hf_tensor, _: np.transpose(to_float32(hf_tensor)),
    )

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"transformer_layer_{i}")
        prefix = f"model.layers.{i}"

        port_plain(
            decoder_layer._self_attention_layernorm.scale,
            f"{prefix}.input_layernorm.weight",
        )
        port_plain(
            decoder_layer._feedforward_layernorm.scale,
            f"{prefix}.post_attention_layernorm.weight",
        )

        attention = decoder_layer._self_attention_layer
        for keras_dense, name in (
            (attention.query_dense, "q_proj"),
            (attention.key_dense, "k_proj"),
            (attention.value_dense, "v_proj"),
            (attention._output_dense, "o_proj"),
        ):
            loader.port_weight(
                keras_variable=keras_dense.kernel,
                hf_weight_key=f"{prefix}.self_attn.{name}.weight",
                hook_fn=reshape_kernel,
            )

        mlp = decoder_layer.mlp
        if not decoder_layer.sparse:
            port_linear(
                mlp._feedforward_gate_dense,
                f"{prefix}.mlp.gate_proj",
                bias=False,
            )
            port_linear(
                mlp._feedforward_intermediate_dense,
                f"{prefix}.mlp.up_proj",
                bias=False,
            )
            port_linear(
                mlp._feedforward_output_dense,
                f"{prefix}.mlp.down_proj",
                bias=False,
            )
            continue

        # The router weight is `(experts, hidden)` in the checkpoint.
        loader.port_weight(
            keras_variable=mlp.router_kernel,
            hf_weight_key=f"{prefix}.mlp.gate.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(to_float32(hf_tensor)),
        )
        # The checkpoint holds one tensor per expert and projection. The
        # Keras expert bank stacks them. The gate comes before the up
        # projection on the last axis.
        gate_up_experts = []
        down_experts = []
        for expert in range(backbone.num_experts):
            expert_prefix = f"{prefix}.mlp.experts.{expert}"
            gate_proj = loader.get_tensor(f"{expert_prefix}.gate_proj.weight")
            up_proj = loader.get_tensor(f"{expert_prefix}.up_proj.weight")
            gate_up_experts.append(
                np.concatenate(
                    [
                        np.transpose(to_float32(gate_proj)),
                        np.transpose(to_float32(up_proj)),
                    ],
                    axis=-1,
                )
            )
            down_proj = loader.get_tensor(f"{expert_prefix}.down_proj.weight")
            down_experts.append(np.transpose(to_float32(down_proj)))
        mlp.expert_bank._expert_feedforward_gate_dense.assign(
            np.stack(gate_up_experts, axis=0)
        )
        mlp.expert_bank._expert_feedforward_output_dense.assign(
            np.stack(down_experts, axis=0)
        )
        del gate_up_experts, down_experts

        # The shared experts have no gate weight in the checkpoint.
        shared = mlp.shared_expert_dense
        port_linear(
            shared._feedforward_gate_dense,
            f"{prefix}.mlp.shared_experts.gate_proj",
            bias=False,
        )
        port_linear(
            shared._feedforward_intermediate_dense,
            f"{prefix}.mlp.shared_experts.up_proj",
            bias=False,
        )
        port_linear(
            shared._feedforward_output_dense,
            f"{prefix}.mlp.shared_experts.down_proj",
            bias=False,
        )

    port_plain(
        backbone.get_layer("sequence_output_layernorm").scale,
        "model.norm.weight",
    )

    # === Projector and merge vectors ===
    port_linear(backbone.multimodal_projector, "model.projector.layers")
    merger = backbone.image_text_embedding_merger
    port_plain(merger.image_newline, "model.image_newline")
    # The checkpoint key keeps a typo.
    port_plain(merger.view_separator, "model.view_seperator")

    # === SAM tower ===
    vision_encoder = backbone.vision_encoder
    sam_prefix = "model.sam_model"
    port_conv(
        vision_encoder.sam_patch_embedding.projection,
        f"{sam_prefix}.patch_embed.proj",
    )
    port_plain(
        vision_encoder.sam_position_embedding.pos_embed,
        f"{sam_prefix}.pos_embed",
    )
    for i, block in enumerate(vision_encoder.sam_blocks):
        block_prefix = f"{sam_prefix}.blocks.{i}"
        port_norm(block.layer_norm1, f"{block_prefix}.norm1")
        port_norm(block.layer_norm2, f"{block_prefix}.norm2")
        port_linear(block.attention.qkv, f"{block_prefix}.attn.qkv")
        port_linear(block.attention.projection, f"{block_prefix}.attn.proj")
        relative_pe = block.attention.add_decomposed_relative_pe
        port_plain(relative_pe.rel_pos_h, f"{block_prefix}.attn.rel_pos_h")
        port_plain(relative_pe.rel_pos_w, f"{block_prefix}.attn.rel_pos_w")
        # The MLP is a `Sequential` of dense, activation and dense layers.
        mlp_layers = block.mlp_block.dense_net.layers
        port_linear(mlp_layers[0], f"{block_prefix}.mlp.lin1")
        port_linear(mlp_layers[2], f"{block_prefix}.mlp.lin2")
    port_conv(
        vision_encoder.sam_neck_conv_1, f"{sam_prefix}.neck.0", bias=False
    )
    port_norm(vision_encoder.sam_neck_norm_1, f"{sam_prefix}.neck.1")
    port_conv(
        vision_encoder.sam_neck_conv_2, f"{sam_prefix}.neck.2", bias=False
    )
    port_norm(vision_encoder.sam_neck_norm_2, f"{sam_prefix}.neck.3")
    port_conv(vision_encoder.sam_net_2, f"{sam_prefix}.net_2", bias=False)
    port_conv(vision_encoder.sam_net_3, f"{sam_prefix}.net_3", bias=False)

    # === CLIP tower ===
    # The checkpoint also holds `embeddings.patch_embedding.weight`. The
    # model never reads it, because the SAM features act as the patches. The
    # converter skips it.
    clip_prefix = "model.vision_model"
    port_plain(
        vision_encoder.clip_embedding.class_embedding,
        f"{clip_prefix}.embeddings.class_embedding",
    )
    port_plain(
        vision_encoder.clip_embedding.position_embedding,
        f"{clip_prefix}.embeddings.position_embedding.weight",
    )
    port_norm(vision_encoder.clip_pre_layer_norm, f"{clip_prefix}.pre_layrnorm")
    num_heads = vision_encoder.clip_num_heads
    head_dim = vision_encoder.clip_hidden_dim // num_heads
    for i, layer in enumerate(vision_encoder.clip_layers):
        layer_prefix = f"{clip_prefix}.transformer.layers.{i}"
        port_norm(layer.layer_norm_1, f"{layer_prefix}.layer_norm1")
        port_norm(layer.layer_norm_2, f"{layer_prefix}.layer_norm2")

        # The checkpoint fuses the query, key and value projections on the
        # output axis, in this order.
        for index, keras_dense in enumerate(
            (
                layer.attention.query_dense,
                layer.attention.key_dense,
                layer.attention.value_dense,
            )
        ):
            loader.port_weight(
                keras_variable=keras_dense.kernel,
                hf_weight_key=f"{layer_prefix}.self_attn.qkv_proj.weight",
                hook_fn=lambda hf_tensor, keras_shape, index=index: np.reshape(
                    np.transpose(
                        np.split(to_float32(hf_tensor), 3, axis=0)[index]
                    ),
                    keras_shape,
                ),
            )
            loader.port_weight(
                keras_variable=keras_dense.bias,
                hf_weight_key=f"{layer_prefix}.self_attn.qkv_proj.bias",
                hook_fn=lambda hf_tensor, keras_shape, index=index: np.reshape(
                    np.split(to_float32(hf_tensor), 3, axis=0)[index],
                    keras_shape,
                ),
            )
        loader.port_weight(
            keras_variable=layer.attention.output_dense.kernel,
            hf_weight_key=f"{layer_prefix}.self_attn.out_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(to_float32(hf_tensor)),
                (num_heads, head_dim, keras_shape[-1]),
            ),
        )
        port_plain(
            layer.attention.output_dense.bias,
            f"{layer_prefix}.self_attn.out_proj.bias",
        )
        port_linear(layer.dense_1, f"{layer_prefix}.mlp.fc1")
        port_linear(layer.dense_2, f"{layer_prefix}.mlp.fc2")

    return backbone


def convert_tokenizer(cls, preset, **kwargs):
    tokenizer_config = load_json(preset, "tokenizer.json")
    vocab = tokenizer_config["model"]["vocab"]
    merges = tokenizer_config["model"]["merges"]

    # Newer `tokenizers` versions store each merge as a pair of strings.
    if merges and isinstance(merges[0], list):
        merges = [" ".join(merge) for merge in merges]

    # The tokenizer class picks its unsplittable tokens. Every added token
    # needs a vocabulary entry to keep its id.
    for token in tokenizer_config["added_tokens"]:
        vocab[token["content"]] = token["id"]

    return cls(vocabulary=vocab, merges=merges, **kwargs)
