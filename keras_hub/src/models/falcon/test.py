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


# def convert_weights(backbone, transformers_config):
    
#     for i in range(backbone.num_layers):
#         decoder_layer = backbone.get_layer(f"transformer_layer_{i}")
#         print(decoder_layer)
#         # print(decoder_layer.input_layernorm.weights)
#         # print('======================')
#         # print(decoder_layer.post_attention_layernorm.weights)


# transformers_config = {
#     "vocab_size": 50304,
#     "num_hidden_layers": 32,
#     "hidden_size": 4544,
#     "num_attention_heads": 71,
# }
# keras_config = convert_backbone_config(transformers_config)

# backbone = FalconBackbone(**keras_config)

# convert_weights(backbone, transformers_config)


