import keras
import numpy as np

from .layers import gelu


# Step 1
# Create and return the CLIP Embeddings
class OpenCLIPTextTransformer(keras.models.Model):
    def __init__(self, maxLength=77, vocabularySize=49408):
        super().__init__()

        # Create embeddings -> Step 2
        self.embeddings = OpenCLIPTextEmbeddings(
            maxLength=maxLength, vocabularySize=vocabularySize
        )

        # Create encoder -> Step 3
        self.encoder = OpenCLIPEncoder()

        self.final_layer_norm = keras.layers.LayerNormalization(
            epsilon=1e-5, name="FinalLayerNormalization"
        )
        self.causal_attention_mask = keras.initializers.Constant(
            np.triu(np.ones((1, 1, 77, 77), dtype="float32") * -np.inf, k=1)
        )

    def call(self, inputs):
        input_ids, position_ids = inputs
        x = self.embeddings([input_ids, position_ids])
        x = self.encoder([x, self.causal_attention_mask])
        return self.final_layer_norm(x)


# Step 2
# Create and return word and position embeddings
class OpenCLIPTextEmbeddings(keras.layers.Layer):
    def __init__(self, maxLength=77, vocabularySize=49408, embeddingSize=1024):
        super().__init__()
        # Token Embedding Layer - Representing a sequence of tokens (words)
        self.token_embedding_layer = keras.layers.Embedding(
            vocabularySize, embeddingSize, name="token_embedding"
        )
        # Position Embedding layer - Where is the word in the sentence? What does it mean in the context of the sentence?
        self.position_embedding_layer = keras.layers.Embedding(
            maxLength, embeddingSize, name="position_embedding"
        )

    def call(self, inputs):
        input_ids, position_ids = inputs
        word_embeddings = self.token_embedding_layer(input_ids)
        position_embeddings = self.position_embedding_layer(position_ids)
        return word_embeddings + position_embeddings


# Step 3
# Create and return the hidden states (aka hidden size)
class OpenCLIPEncoder(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.layers = [OpenCLIPEncoderLayer() for i in range(24)]

    def call(self, inputs):
        [hidden_states, causal_attention_mask] = inputs
        for l in self.layers:
            hidden_states = l([hidden_states, causal_attention_mask])
        return hidden_states


# Step 4 (also creatd in step 3)
# Create the layers
class OpenCLIPEncoderLayer(keras.layers.Layer):
    def __init__(self, intermediateSize=4096, embeddingSize=1024):
        super().__init__()
        self.layer_norm1 = keras.layers.LayerNormalization(
            epsilon=1e-5, name="LayerNormalization01"
        )  # Layer Normalization 1
        self.self_attn = OpenCLIPAttention()  # Attention Layers
        self.layer_norm2 = keras.layers.LayerNormalization(
            epsilon=1e-5, name="LayerNormalization02"
        )  # Layer Normalization 2
        self.fc1 = keras.layers.Dense(intermediateSize, name="FC1")  # MLP layer?
        self.fc2 = keras.layers.Dense(embeddingSize, name="FC2")  # ???

    def call(self, inputs):
        hidden_states, causal_attention_mask = inputs
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn([hidden_states, causal_attention_mask])
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)

        # MLP Steps
        hidden_states = self.fc1(hidden_states)
        hidden_states = gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)

        return residual + hidden_states


class OpenCLIPAttention(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.embed_dim = 1024
        self.num_heads = 16
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = keras.layers.Dense(
            self.embed_dim, name="QueryState"
        )  # Query states, the given word
        self.k_proj = keras.layers.Dense(
            self.embed_dim, name="KeyState"
        )  # Key states, all other words
        self.v_proj = keras.layers.Dense(
            self.embed_dim, name="ValueState"
        )  # Value states, the sentence
        self.out_proj = keras.layers.Dense(
            self.embed_dim, name="OutProjection"
        )  # Out Projection?

    def _shape(self, tensor, seq_len: int, bsz: int):
        # Keys
        a = keras.ops.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim))
        return keras.layers.Permute((2, 1, 3))(a)  # bs , n_head , seq_len , head_dim

    def call(self, inputs):
        hidden_states, causal_attention_mask = inputs
        bsz, tgt_len, embed_dim = hidden_states.shape
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), tgt_len, -1)
        value_states = self._shape(self.v_proj(hidden_states), tgt_len, -1)

        proj_shape = (-1, tgt_len, self.head_dim)
        query_states = self._shape(query_states, tgt_len, -1)
        query_states = keras.ops.reshape(query_states, proj_shape)
        key_states = keras.ops.reshape(key_states, proj_shape)

        src_len = tgt_len
        value_states = keras.ops.reshape(value_states, proj_shape)
        attn_weights = query_states @ keras.layers.Permute((2, 1))(key_states)

        attn_weights = keras.ops.reshape(
            attn_weights, (-1, self.num_heads, tgt_len, src_len)
        )
        attn_weights = attn_weights + causal_attention_mask
        attn_weights = keras.ops.reshape(attn_weights, (-1, tgt_len, src_len))

        attn_weights = keras.ops.softmax(attn_weights)
        attn_output = attn_weights @ value_states

        attn_output = keras.ops.reshape(
            attn_output, (-1, self.num_heads, tgt_len, self.head_dim)
        )
        attn_output = keras.layers.Permute((2, 1, 3))(attn_output)
        attn_output = keras.ops.reshape(attn_output, (-1, tgt_len, embed_dim))

        return self.out_proj(attn_output)
