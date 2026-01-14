import keras
from keras import ops
import numpy as np
from modernbert_layers import ModernBertAttention 

def test_attention_serialization():
    hidden_dim = 128
    num_heads = 4
    
    rotary_emb = keras.layers.Dense(32) 
    
    original_layer = ModernBertAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        rotary_embedding=rotary_emb,
        local_attention_window=64,
        dropout=0.1,
        name="test_attention"
    )

    # Serialization
    config = original_layer.get_config()
    print("--- Layer Config Generated ---")
    print(config)

    # Reconstruct from configuration (Deserialization)
    reconstructed_layer = ModernBertAttention.from_config(config)

    # properties
    assert reconstructed_layer.hidden_dim == original_layer.hidden_dim
    assert reconstructed_layer.num_heads == original_layer.num_heads
    assert reconstructed_layer.local_attention_window == original_layer.local_attention_window
    print("\nVerification Successful: Properties Match!")

    # Functional equality
    x = ops.ones((1, 16, hidden_dim))

    _ = original_layer(x)
    _ = reconstructed_layer(x)
    
    # Sync weights to ensure they produce the same output
    reconstructed_layer.set_weights(original_layer.get_weights())
    
    y_orig = original_layer(x, training=False)
    y_recon = reconstructed_layer(x, training=False)
    
    np.testing.assert_allclose(ops.convert_to_numpy(y_orig), 
                               ops.convert_to_numpy(y_recon), atol=1e-5)
    print("Verification Successful: Outputs Match!")

if __name__ == "__main__":
    test_attention_serialization()