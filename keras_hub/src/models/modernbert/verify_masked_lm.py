import tensorflow as tf
from modernbert_tokenizer import ModernBertTokenizer
from modernbert_preprocessor import ModernBertMaskedLMPreprocessor
from modernbert_backbone import ModernBertBackbone
from modernbert_masked_lm import ModernBertMaskedLM

tokenizer = ModernBertTokenizer(
    vocabulary="keras_hub/src/models/modernbert/vocab.json", 
    merges="keras_hub/src/models/modernbert/merges.txt"
)

preprocessor = ModernBertMaskedLMPreprocessor(
    tokenizer, 
    sequence_length=128
)

backbone = ModernBertBackbone(
    vocabulary_size=50368, 
    hidden_dim=256, 
    intermediate_dim=512, 
    num_layers=4, 
    num_heads=8
)

model = ModernBertMaskedLM(backbone, preprocessor)

raw_data = ["The quick brown fox jumps over the lazy dog."]
dataset = tf.data.Dataset.from_tensor_slices(raw_data)

dataset = dataset.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(1)

#  ({'token_ids', 'padding_mask', 'mask_positions'}, labels, weights)
preds = model.predict(dataset)

print(f"MLM Prediction Shape: {preds.shape}") 
# Expected Output: (1, 96, 50368)