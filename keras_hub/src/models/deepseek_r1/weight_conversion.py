import torch
import keras
from keras_hub.src.models.deepseek_r1.deepseek_base import Transformer, ModelArgs
import logging
import time
from tqdm import tqdm
keras.config.set_dtype_policy("mixed_float16")
import numpy as np

logging.basicConfig(level=logging.INFO)

logging.info("Load torch weights...")
torch_weights = torch.load('keras_hub/src/models/deepseek_r1/deepseek_weights.pt')

logging.info(torch_weights.keys())

logging.info("PyTorch weights:")
for key, weight in torch_weights.items():
    logging.info(f"{key}, {weight.shape}")


"""
(['embed.weight', 'layers.0.attn.wq.weight', 'layers.0.attn.wkv_a.weight', 
'layers.0.attn.kv_norm.weight', 'layers.0.attn.wkv_b.weight', 
'layers.0.attn.wo.weight', 'layers.0.ffn.w1.weight', 
'layers.0.ffn.w2.weight', 'layers.0.ffn.w3.weight', 
'layers.0.attn_norm.weight', 'layers.0.ffn_norm.weight', 
'norm.weight', 'head.weight'])
"""

args = ModelArgs()
logging.info("Initializing model...")
model = Transformer(args)

logging.info("Running dummy input...")
x = keras.random.randint((1, 128), 0, args.vocab_size)
model(x)

# print keras weights
logging.info("Keras weights:")
#for layer in model.layers: logging.info(layer.get_config(), layer.get_weights())
for layer in model.layers: logging.info(f"{layer.name}, {layer.get_weights()[0].shape}")


def convert_block(keras_block, torch_weights, index):
    print("Weights and shapes")
    for i, w in enumerate(keras_block.weights):
        print(i, w.path, w.shape)
    print()
    for i, w in enumerate(torch_weights):
        if f"layers.{index-1}" in w:
            print(i-1, w, torch_weights[w].shape)
            
            
    keras_block.weights[0].assign(torch_weights[f"layers.{index-1}.attn.wq.weight"])
    keras_block.weights[1].assign(torch_weights[f"layers.{index-1}.attn.wkv_a.weight"])
    keras_block.weights[2].assign(torch_weights[f"layers.{index-1}.attn.kv_norm.weight"])
    keras_block.weights[3].assign(torch_weights[f"layers.{index-1}.attn.wkv_b.weight"])
    keras_block.weights[4].assign(torch_weights[f"layers.{index-1}.attn.wo.weight"])
    keras_block.weights[5].assign(torch_weights[f"layers.{index-1}.ffn.w1.weight"])
    keras_block.weights[6].assign(torch_weights[f"layers.{index-1}.ffn.w2.weight"])
    keras_block.weights[7].assign(torch_weights[f"layers.{index-1}.ffn.w3.weight"])
    keras_block.weights[8].assign(torch_weights[f"layers.{index-1}.attn_norm.weight"])
    keras_block.weights[9].assign(torch_weights[f"layers.{index-1}.ffn_norm.weight"])


logging.info("Converting embedding")
model.layers[0].set_weights(weights=[torch_weights["embed.weight"]])
logging.info(model.layers[0].weights)

logging.info("Converting head")
model.layers[-1].set_weights(weights=[torch_weights['head.weight']])
logging.info(model.layers[-1].weights)

logging.info("Converting head norm")
model.layers[-2].set_weights(weights=[torch_weights['norm.weight']])
logging.info(model.layers[-2].weights)

index = 1
convert_block(model.layers[index], torch_weights, index)

# verify outputs

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
    x = keras.ops.concatenate([x, res_token], 1)
    print(x)
    end_time = time.time() - start_time
    total_generation_time += end_time
    total_tokens_generated += 1

tokens_per_second = total_tokens_generated / total_generation_time
logging.info(f"Total tokens generated: {total_tokens_generated}")
logging.info(f"Total generation time: {total_generation_time:.2f} seconds")
logging.info(f"Tokens per second: {tokens_per_second:.2f}")
