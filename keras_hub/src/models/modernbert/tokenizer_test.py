import os
os.environ["KERAS_BACKEND"] = "tensorflow" # "tensorflow" , "jax" "torch" 

import keras_hub
import numpy as np
from modernbert_backbone import ModernBertBackbone
from modernbert_tokenizer import ModernBertTokenizer

# tokenizer 
vocab = {"<|padding|>": 0, "<|endoftext|>": 1, "<mask|>": 2, "the": 3, "fox": 4}
merges = ["t h", "th e", "f o", "fo x"]
tokenizer = ModernBertTokenizer(vocabulary=vocab, merges=merges)

# backbone
backbone = ModernBertBackbone(
    vocabulary_size=5,
    hidden_dim=32,
    intermediate_dim=64,
    num_layers=2,
    num_heads=2,
)
text = "the fox"
token_ids = tokenizer(text)

# batch dimension
token_ids = np.expand_dims(token_ids, 0)
padding_mask = token_ids != tokenizer.pad_token_id

output = backbone({"token_ids": token_ids, "padding_mask": padding_mask})

print(f"Backend: {os.environ['KERAS_BACKEND']}")
print(f"Input: {text}")
print(f"Output Shape: {output.shape}") # Should be (1, 2, 32)

