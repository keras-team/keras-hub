import numpy as np
import keras
from modernbert_tokenizer import ModernBertTokenizer
from modernbert_preprocessor import ModernBertMaskedLMPreprocessor
from modernbert_backbone import ModernBertBackbone

tokenizer = ModernBertTokenizer(vocabulary="keras_hub/src/models/modernbert/vocab.json", 
                                merges="keras_hub/src/models/modernbert/merges.txt")

# Preprocessor
preprocessor = ModernBertMaskedLMPreprocessor(
    tokenizer=tokenizer,
    sequence_length=128
)

# Backbone
backbone = ModernBertBackbone(
    vocabulary_size=50368,
    hidden_dim=256,
    intermediate_dim=512,
    num_layers=4,
    num_heads=8,
)

raw_data = ["The quick brown fox jumps over the lazy dog."]


# {'token_ids', 'padding_mask', 'mask_positions'}
x, y, sw = preprocessor(raw_data)

# Extract what backbone needs
backbone_inputs = {
    "token_ids": x["token_ids"],
    "padding_mask": x["padding_mask"] 
}

print(f"Token IDs shape: {backbone_inputs['token_ids'].shape}") # (1, 128)
print(f"Padding Mask shape: {backbone_inputs['padding_mask'].shape}") # (1, 128)

backbone_output = backbone(backbone_inputs)

print(f"Preprocessor Output Keys: {x.keys()}")
print(f"Backbone Output Shape: {backbone_output.shape}") # (Batch, 128, 256)
