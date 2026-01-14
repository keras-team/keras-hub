import os
import numpy as np
import keras
from keras import ops
from modernbert_backbone import ModernBertBackbone

def test_modernbert_backbone_full_flow():
    vocab_size = 1000
    hidden_dim = 64
    intermediate_dim = 128
    num_layers = 4  # Includes 1 global layer if global_attn_every_n_layers=3
    num_heads = 4

    print("Initializing ModernBERT Backbone")
    backbone = ModernBertBackbone(
        vocabulary_size=vocab_size,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        local_attention_window=32,
        global_attn_every_n_layers=3,
    )

    # Dummy input
    batch_size = 2
    seq_len = 64
    input_data = {
        "token_ids": np.random.randint(0, vocab_size, (batch_size, seq_len)),
        "padding_mask": np.ones((batch_size, seq_len)),
    }

    # Test Forward Pass
    print("Testing Forward Pass...")
    output = backbone(input_data)
    
    # Assert output shape: (batch, seq, hidden)
    expected_shape = (batch_size, seq_len, hidden_dim)
    assert output.shape == expected_shape
    print(f"Forward Pass Success! Shape: {output.shape}")

    # Serialization
    print("Testing Full Model Saving/Loading (.keras)...")
    model_path = "modernbert_test.keras"
    backbone.save(model_path)
    
    # Load it back

    reloaded_model = keras.models.load_model(model_path)
    
    # Compare outputs of original and reloaded
    y_orig = backbone(input_data)
    y_reloaded = reloaded_model(input_data)
    
    np.testing.assert_allclose(
        ops.convert_to_numpy(y_orig), 
        ops.convert_to_numpy(y_reloaded), 
        atol=1e-5
    )
    print("Serialization Round-trip Success!")

    # Cleanup
    if os.path.exists(model_path):
        os.remove(model_path)

if __name__ == "__main__":
    test_modernbert_backbone_full_flow()