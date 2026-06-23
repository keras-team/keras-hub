import torch
import keras
import keras_hub
from keras_hub.src.vllm.adapter import KerasVLLMAdapter

def test_forward_step_with_cache():
    print("Initializing dummy model...")
    # Use a tiny preset or create from config to avoid download
    backbone = keras_hub.models.GemmaBackbone(
        vocabulary_size=256,
        num_layers=2,
        num_heads=4,
        num_query_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        hidden_dim=64,
        intermediate_dim=128,
        max_sequence_length=128,
    )
    model = keras_hub.models.GemmaCausalLM(backbone=backbone)
    
    adapter = KerasVLLMAdapter(config=None)
    adapter.model = model
    
    batch_size = 1
    seq_len = 5
    
    input_ids = torch.randint(0, 256, (batch_size, seq_len))
    positions = torch.arange(seq_len).unsqueeze(0)
    
    # Keras Hub expects cache shape: (batch_size, 2, max_len, num_heads, head_dim)
    # Let's see what happens if we pass dummy kv_caches
    kv_caches = [
        torch.zeros((batch_size, 2, 128, 2, 64), dtype=torch.float32)
        for _ in range(2)
    ]
    
    print("Calling forward_step...")
    output = adapter.forward_step(
        input_ids=input_ids,
        positions=positions,
        kv_caches=kv_caches,
        attention_metadata=None,
    )
    
    print("Output shape:", output.shape)
    
if __name__ == "__main__":
    test_forward_step_with_cache()
