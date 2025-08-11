import numpy as np

from keras_hub.src.models.llama3.llama3_backbone import Llama3Backbone
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = Llama3Backbone


def convert_backbone_config(transformers_config):
    backbone_config = {
        "vocabulary_size": transformers_config["vocab_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "num_query_heads": transformers_config["num_attention_heads"],
        "hidden_dim": transformers_config["hidden_size"],
        "intermediate_dim": transformers_config["intermediate_size"],
        "num_key_value_heads": transformers_config["num_key_value_heads"],
        "tie_word_embeddings": transformers_config["tie_word_embeddings"],
        "rope_max_wavelength": transformers_config["rope_theta"],
    }

    if transformers_config.get("rope_scaling", None) is not None:
        if transformers_config["rope_scaling"]["rope_type"] != "llama3":
            raise ValueError("The config should be a valid llama3 config.")
        backbone_config["rope_frequency_adjustment_factor"] = (
            transformers_config["rope_scaling"]["factor"]
        )
        backbone_config["rope_low_freq_factor"] = transformers_config[
            "rope_scaling"
        ]["low_freq_factor"]
        backbone_config["rope_high_freq_factor"] = transformers_config[
            "rope_scaling"
        ]["high_freq_factor"]
        backbone_config["rope_pretraining_sequence_length"] = (
            transformers_config["rope_scaling"][
                "original_max_position_embeddings"
            ]
        )
    return backbone_config


def convert_weights(backbone, loader, transformers_config):
    loader.port_weight(
        keras_variable=backbone.get_layer("token_embedding").embeddings,
        hf_weight_key="model.embed_tokens.weight",
    )
    if not backbone.tie_word_embeddings:
        loader.port_weight(
            keras_variable=backbone.get_layer(
                "token_embedding"
            ).reverse_embeddings,
            hf_weight_key="lm_head.weight",
            # rearrange_pattern="b a -> a b",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )

    def transpose_and_reshape(x, shape):
        return np.reshape(np.transpose(x), shape)

    # Attention blocks
    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"transformer_layer_{i}")
        # Norm layers
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layernorm.scale,
            hf_weight_key=f"model.layers.{i}.input_layernorm.weight",
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_layernorm.scale,
            hf_weight_key=f"model.layers.{i}.post_attention_layernorm.weight",
        )

        # Attention layers
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._query_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.q_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._key_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.k_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._value_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.v_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._output_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.o_proj.weight",
            # rearrange_patterns="c (a b) -> a b c",
            # rearrange_dims={"a": backbone.num_query_heads},
            hook_fn=transpose_and_reshape,
        )

        # MLP layers
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_gate_dense.kernel,
            hf_weight_key=f"model.layers.{i}.mlp.gate_proj.weight",
            # rearrange_patterns="b a -> a b",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_intermediate_dense.kernel,
            hf_weight_key=f"model.layers.{i}.mlp.up_proj.weight",
            # rearrange_patterns="b a -> a b",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_output_dense.kernel,
            hf_weight_key=f"model.layers.{i}.mlp.down_proj.weight",
            # rearrange_patterns="b a -> a b",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )

    # Final normalization layer
    loader.port_weight(
        keras_variable=backbone.get_layer("sequence_output_layernorm").scale,
        hf_weight_key="model.norm.weight",
    )

    return backbone


def convert_tokenizer(cls, preset, **kwargs):
    tokenizer_config = load_json(preset, "tokenizer.json")
    vocab = tokenizer_config["model"]["vocab"]
    merges = tokenizer_config["model"]["merges"]
    
    # Debug: Print the structure of merges
    print(f"Type of merges: {type(merges)}")
    if merges:
        print(f"Length of merges: {len(merges)}")
        print(f"Type of first merge: {type(merges[0])}")
        print(f"First few merges: {merges[:3]}")
    
    # Handle different merge formats
    if not merges:
        merges = []
    elif isinstance(merges[0], str):
        # Standard format: list of strings like ["Ġ a", "Ġ b", ...]
        merges = [tuple(merge.split()) for merge in merges]
    elif isinstance(merges[0], list) and len(merges[0]) == 2:
        # Alternative format: list of lists like [["Ġ", "a"], ["Ġ", "b"], ...]
        merges = [tuple(merge) for merge in merges]
    elif isinstance(merges[0], (list, tuple)) and len(merges[0]) == 1:
        # Another possible format: nested single-element lists/tuples
        # This might be the issue - convert to pairs
        print("Warning: Merges appear to be in single-element format, attempting to reconstruct pairs")
        # This is a fallback - we might need to handle this differently
        merges = []
    elif not isinstance(merges[0], tuple):
        print(f"Unexpected merge format: {type(merges[0])}")
        print(f"Sample merge: {merges[0]}")
        # Try to convert whatever format it is
        try:
            merges = [tuple(merge) if hasattr(merge, '__iter__') and not isinstance(merge, str) else (merge, '') for merge in merges]
        except:
            print("Failed to convert merges, using empty list")
            merges = []
    
    print(f"Final merges length: {len(merges)}")
    if merges:
        print(f"Sample final merge: {merges[0]}")

    # Load all special tokens with the exception of "reserved" ones.
    special_tokens = set()
    for token in tokenizer_config["added_tokens"]:
        if not token["content"].startswith("<|reserved_special_token_"):
            vocab[token["content"]] = token["id"]
            special_tokens.add(token["content"])

    # Load text start and stop tokens from the config.
    # Llama3 uses the <|end_of_text|> end token for regular models
    # but uses <|eot_id|> for instruction-tuned  variants.
    tokenizer_config2 = load_json(preset, "tokenizer_config.json")
    bos_token = tokenizer_config2["bos_token"]
    eos_token = tokenizer_config2["eos_token"]

    kwargs.update(
        {
            "bos_token": bos_token,
            "eos_token": eos_token,
            "misc_special_tokens": special_tokens,
        }
    )

    return cls(vocabulary=vocab, merges=merges, **kwargs)
