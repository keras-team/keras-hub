import numpy as np

from keras_hub.src.models.falcon import FalconBackbone
from keras_hub.src.utils.preset_utils import HF_TOKENIZER_CONFIG_FILE
from keras_hub.src.utils.preset_utils import get_file
from keras_hub.src.utils.preset_utils import load_json

backbone_cls= FalconBackbone

def convert_backbone_config(transformers_config):
    return {
        "vocabulary_size": transformers_config["vocab_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "num_attention_heads": transformers_config["num_attention_heads"],
        "hidden_dim": transformers_config["hidden_size"],
        "intermediate_dim": 32*4,
    }

def transpose_and_reshape(x, shape):
        return np.reshape(np.transpose(x), shape)

def convert_weights(backbone, loader, transformers_config):
        # Embeddings
        loader.port_weight(keras_variable= backbone.get_layer('token_embedding').embeddings,
                           hf_weight_key = "word_embeddings.weight")


        for i in range(backbone.num_layers):
            decoder_layer = backbone.get_layer(f"transformer_layer_{i}")
            
            # Norm layer 
            loader.port_weight(keras_variable=decoder_layer.input_layernorm.gamma, 
                            hf_weight_key=f'h.{i}.input_layernorm.weight')
            
            # Attention layers
            loader.port_weight(keras_variable=decoder_layer.attention_layer.output_dense.kernel,
                            hf_weight_key= f'h.{i}.self_attention.dense.weight')
            
            loader.port_weight(keras_variable= decoder_layer.post_attention_layernorm.gamma,
                               hf_weight_key=f'h.{i}.self_attention.query_key_value.weight',
                               hook_fn=lambda hf_tensor, keras_shape: np.mean(np.reshape(hf_tensor, (-1, keras_shape[0])), axis=0))


def convert_tokenizer(cls, preset, **kwargs):
    tokenizer_config = load_json(preset, 'tokenizer_config.json')
    tokenizer_data = load_json(preset, 'tokenizer.json')
    vocab = tokenizer_data["model"]["vocab"]
    merges = tokenizer_data["model"].get("merges", None)  

    tokenizer_kwargs = {
        "vocabulary": vocab,
        "merges": merges
    }
    return cls(**tokenizer_kwargs)