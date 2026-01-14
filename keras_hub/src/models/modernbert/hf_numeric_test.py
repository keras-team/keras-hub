import numpy as np
import torch
import keras
import keras_hub
from transformers import AutoModel, AutoTokenizer
from modernbert_backbone import ModernBertBackbone
# from keras_hub.src.models.modernbert_backbone import ModernBertBackbone

text = "Keras Hub is modernized with ModernBERT!"
model_id = "answerdotai/ModernBERT-base"

hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
hf_model = AutoModel.from_pretrained(model_id)
hf_inputs = hf_tokenizer(text, return_tensors="pt")

with torch.no_grad():
    hf_outputs = hf_model(**hf_inputs).last_hidden_state.numpy()

# Ensure to initialize Keras model with the same config as HF
keras_backbone = ModernBertBackbone(
    vocabulary_size=50368,
    num_layers=22,
    num_heads=12,
    hidden_dim=768,
    intermediate_dim=1152, # ModernBERT uses GeGLU (dim * 1.5)
    local_attention_window= 128,
)
keras_backbone.load_weights("model_weight.h5")

# Mock comparison 
keras_inputs = {
    "token_ids": hf_inputs["input_ids"].numpy(),
    "padding_mask": hf_inputs["attention_mask"].numpy(),
}
keras_outputs = keras_backbone.predict(keras_inputs)

# Compare
diff = np.abs(hf_outputs - keras_outputs)
print(f"Max absolute difference: {np.max(diff)}")
print(f"Mean absolute difference: {np.mean(diff)}")

# Assertion (usually < 1e-5 for float32)
# np.testing.assert_allclose(hf_outputs, keras_outputs, atol=1e-5)