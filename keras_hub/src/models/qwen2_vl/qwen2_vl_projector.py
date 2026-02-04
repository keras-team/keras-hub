import keras
from keras import layers
from keras import ops

class Qwen2VLProjector(layers.Layer):
    """
    Projector layer for Qwen2-VL.
    
    This layer downsamples vision features by merging 2x2 neighboring patches 
    into a single token and projecting them to the LLM's hidden size.
    """
    def __init__(self, hidden_size, output_hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.output_hidden_size = output_hidden_size
        
        self.merger = layers.Sequential([
            layers.Dense(output_hidden_size, name="merger_proj"),
            layers.Activation("gelu", name="activation"),
            layers.Dense(output_hidden_size, name="output_proj")
        ], name="merger")

    def call(self, x):
        # x shape: (Batch, Height, Width, Channels)
        
        input_shape = ops.shape(x)
        H, W, C = input_shape[1], input_shape[2], input_shape[3]
        
        # Reshape to isolate 2x2 blocks
        # Shape: (B, H/2, 2, W/2, 2, C)
        x = ops.reshape(x, (-1, H // 2, 2, W // 2, 2, C))
        
        # Permute to bring the 2x2 blocks together
        # Shape: (B, H/2, W/2, 2, 2, C)
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
        
        # Flatten the 2x2xC block into a single vector
        # Shape: (B, H/2, W/2, 4*C)
        x = ops.reshape(x, (-1, H // 2, W // 2, 4 * C))
        
        x = self.merger(x)
        
        return x