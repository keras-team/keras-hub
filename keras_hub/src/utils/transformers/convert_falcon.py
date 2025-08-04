import numpy as np

from keras_hub.src.models.falcon import FalconBackbone
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = FalconBackbone


def convert_backbone_config(transformers_config):
    return {
            "vocabulary_size": transformers_config["vocab_size"],
            "num_layers": transformers_config["num_hidden_layers"],
            "hidden_dim": transformers_config["hidden_size"],
            "num_attention_heads": transformers_config["num_attention_heads"],
            "intermediate_dim": transformers_config.get("intermediate_size", 
                            transformers_config["hidden_size"] * 4)
        }


def convert_weights(backbone, loader, transformers_config):
    # Embeddings
    loader.port_weight(
        keras_variable=backbone.get_layer("token_embedding").embeddings,
        hf_weight_key="transformer.word_embeddings.weight",
    )

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"transformer_layer_{i}")

        # Norm layer
        loader.port_weight(
            keras_variable=decoder_layer.input_layernorm.gamma,
            hf_weight_key=f"transformer.h.{i}.input_layernorm.weight",
        )

        
        # Attention layers
        hidden_dim = transformers_config["hidden_size"]

        loader.port_weight(
            keras_variable=decoder_layer.attention_layer.output_dense.kernel,
            hf_weight_key=f"transformer.h.{i}.self_attention.dense.weight",
        )
        input_dim = decoder_layer.attention_layer.query_dense.kernel.shape[0]
        attention_dim = decoder_layer.attention_layer.query_dense.kernel.shape[1]

        hf_tensor= loader.get_tensor(f'transformer.h.{i}.self_attention.query_key_value.weight')
        hf_tensor= np.reshape(hf_tensor, (3,hidden_dim, hidden_dim))
        hf_tensor= np.transpose(hf_tensor,(0,2,1))
        query_weight, key_weight, value_weight = hf_tensor

        loader.port_weight(decoder_layer.attention_layer.query_dense.kernel, query_weight)
        loader.port_weight(decoder_layer.attention_layer.key_dense.kernel, key_weight)
        loader.port_weight(decoder_layer.attention_layer.value_dense.kernel, value_weight)


        # MLP dense layers
        loader.port_weight(
            keras_variable=decoder_layer.dense_h_to_4h.kernel,
            hf_weight_key=f"h.{i}.mlp.dense_h_to_4h.weight",
        )
        loader.port_weight(
            keras_variable=decoder_layer.dense_4h_to_h.kernel,
            hf_weight_key=f"h.{i}.mlp.dense_4h_to_h.weight",
        )

    # Final layernorm
    loader.port_weight(
        keras_variable=backbone.get_layer("final_layernorm").gamma,
        hf_weight_key="transformer.ln_f.weight",
    )
    loader.port_weight(
        keras_variable=backbone.get_layer("final_layernorm").beta,
        hf_weight_key="transformer.ln_f.bias",
    )
        

def convert_tokenizer(cls, preset, **kwargs):
    tokenizer_data = load_json(preset, "tokenizer.json")
    vocab = tokenizer_data["model"]["vocab"]
    merges = tokenizer_data["model"].get("merges", None)

    tokenizer_config2 = load_json(preset, "tokenizer_config.json")
    bos_token = tokenizer_config2["bos_token"]
    eos_token = tokenizer_config2["eos_token"]

    tokenizer_kwargs = {"vocabulary": vocab}

    if merges is not None:
        tokenizer_kwargs["merges"] = merges
    if bos_token is not None:
        tokenizer_kwargs["bos_token"] = bos_token
    if eos_token is not None:
        tokenizer_kwargs["eos_token"] = eos_token

    tokenizer_kwargs.update(kwargs) 

    return cls(**tokenizer_kwargs)
