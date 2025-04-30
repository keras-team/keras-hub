"""LayoutLMv3 backbone model implementation.

This module implements the LayoutLMv3 model architecture as described in
"LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking"
(https://arxiv.org/abs/2204.08387).

The LayoutLMv3 model is a multimodal transformer that combines text, layout, and
visual information for document understanding tasks. It uses a unified architecture
to process both text and image inputs, with special attention to spatial relationships
in documents.

Example:
```python
# Initialize backbone from preset
backbone = LayoutLMv3Backbone.from_preset("layoutlmv3_base")

# Process document image and text
outputs = backbone({
    "input_ids": input_ids,  # Shape: (batch_size, seq_length)
    "bbox": bbox,  # Shape: (batch_size, seq_length, 4)
    "attention_mask": attention_mask,  # Shape: (batch_size, seq_length)
    "image": image  # Shape: (batch_size, height, width, channels)
})
```

References:
- [LayoutLMv3 Paper](https://arxiv.org/abs/2204.08387)
- [LayoutLMv3 GitHub](https://github.com/microsoft/unilm/tree/master/layoutlmv3)
"""

import os
from typing import Dict, List, Optional, Tuple, Union

from keras import backend, layers, ops
from keras.saving import register_keras_serializable
from keras.utils import register_keras_serializable
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.api_export import keras_hub_export

from .layoutlmv3_tokenizer import LayoutLMv3Tokenizer
from .layoutlmv3_presets import backbone_presets
from .layoutlmv3_transformer import LayoutLMv3TransformerLayer

@keras_hub_export("keras_hub.models.LayoutLMv3Backbone")
@register_keras_serializable(package="keras_hub")
class LayoutLMv3Backbone(Backbone):
    """LayoutLMv3 backbone model for document understanding tasks.

    This class implements the LayoutLMv3 model architecture for joint text and layout
    understanding in document AI tasks. It processes both text and image inputs while
    maintaining spatial relationships in documents.

    Args:
        vocab_size: int. Size of the vocabulary. Defaults to 30522.
        hidden_size: int. Size of the hidden layers. Defaults to 768.
        num_hidden_layers: int. Number of transformer layers. Defaults to 12.
        num_attention_heads: int. Number of attention heads. Defaults to 12.
        intermediate_size: int. Size of the intermediate layer. Defaults to 3072.
        hidden_act: str. Activation function for the hidden layers. Defaults to "gelu".
        hidden_dropout_prob: float. Dropout probability for hidden layers. Defaults to 0.1.
        attention_probs_dropout_prob: float. Dropout probability for attention layers. Defaults to 0.1.
        max_position_embeddings: int. Maximum sequence length. Defaults to 512.
        type_vocab_size: int. Size of the token type vocabulary. Defaults to 2.
        initializer_range: float. Range for weight initialization. Defaults to 0.02.
        layer_norm_eps: float. Epsilon for layer normalization. Defaults to 1e-12.
        pad_token_id: int. ID of the padding token. Defaults to 0.
        position_embedding_type: str. Type of position embedding. Defaults to "absolute".
        use_cache: bool. Whether to use caching. Defaults to True.
        classifier_dropout: float. Dropout probability for classifier. Defaults to None.
        patch_size: int. Size of image patches. Defaults to 16.
        num_channels: int. Number of image channels. Defaults to 3.
        qkv_bias: bool. Whether to use bias in QKV projection. Defaults to True.
        use_abs_pos: bool. Whether to use absolute position embeddings. Defaults to True.
        use_rel_pos: bool. Whether to use relative position embeddings. Defaults to True.
        rel_pos_bins: int. Number of relative position bins. Defaults to 32.
        max_rel_pos: int. Maximum relative position. Defaults to 128.
        spatial_embedding_dim: int. Dimension of spatial embeddings. Defaults to 64.

    References:
        - [LayoutLMv3 Paper](https://arxiv.org/abs/2204.08387)
        - [LayoutLMv3 GitHub](https://github.com/microsoft/unilm/tree/master/layoutlmv3)
    """
    
    presets = backbone_presets

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        position_embedding_type: str = "absolute",
        use_cache: bool = True,
        classifier_dropout: Optional[float] = None,
        patch_size: int = 16,
        num_channels: int = 3,
        qkv_bias: bool = True,
        use_abs_pos: bool = True,
        use_rel_pos: bool = True,
        rel_pos_bins: int = 32,
        max_rel_pos: int = 128,
        spatial_embedding_dim: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        
        # Input layers
        self.input_ids = layers.Input(shape=(None,), dtype="int32", name="input_ids")
        self.bbox = layers.Input(shape=(None, 4), dtype="int32", name="bbox")
        self.attention_mask = layers.Input(shape=(None,), dtype="int32", name="attention_mask")
        self.image = layers.Input(shape=(None, None, None, num_channels), dtype="float32", name="image")
        
        # Embeddings
        self.word_embeddings = layers.Embedding(
            vocab_size, hidden_size, name="embeddings.word_embeddings"
        )
        self.position_embeddings = layers.Embedding(
            max_position_embeddings, hidden_size, name="embeddings.position_embeddings"
        )
        self.x_position_embeddings = layers.Embedding(1024, spatial_embedding_dim, name="embeddings.x_position_embeddings")
        self.y_position_embeddings = layers.Embedding(1024, spatial_embedding_dim, name="embeddings.y_position_embeddings")
        self.h_position_embeddings = layers.Embedding(1024, spatial_embedding_dim, name="embeddings.h_position_embeddings")
        self.w_position_embeddings = layers.Embedding(1024, spatial_embedding_dim, name="embeddings.w_position_embeddings")
        self.token_type_embeddings = layers.Embedding(
            type_vocab_size, hidden_size, name="embeddings.token_type_embeddings"
        )
        
        # Layer normalization
        self.embeddings_LayerNorm = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="embeddings.LayerNorm"
        )
        self.norm = layers.LayerNormalization(epsilon=layer_norm_eps, name="norm")
        
        # Spatial embedding projections
        self.x_proj = layers.Dense(hidden_size, name="x_proj")
        self.y_proj = layers.Dense(hidden_size, name="y_proj")
        self.h_proj = layers.Dense(hidden_size, name="h_proj")
        self.w_proj = layers.Dense(hidden_size, name="w_proj")
        
        # Transformer encoder layers
        self.encoder_layers = [
            LayoutLMv3TransformerLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                initializer_range=initializer_range,
                layer_norm_eps=layer_norm_eps,
                qkv_bias=qkv_bias,
                use_rel_pos=use_rel_pos,
                rel_pos_bins=rel_pos_bins,
                max_rel_pos=max_rel_pos,
                name=f"encoder.layer.{i}",
            )
            for i in range(num_hidden_layers)
        ]
        
        # Image processing
        self.patch_embed = layers.Conv2D(
            hidden_size,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            name="patch_embed.proj",
        )
        self.patch_embed_layer_norm = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="LayerNorm"
        )
        
        # CLS token
        self.cls_token = self.add_weight(
            shape=(1, 1, hidden_size),
            initializer="random_normal",
            trainable=True,
            name="cls_token",
        )
        
        # Pooler
        self.pooler = layers.Dense(hidden_size, activation="tanh", name="pooler")
        
    def call(self, inputs: Dict[str, backend.Tensor]) -> Dict[str, backend.Tensor]:
        """Process text and image inputs through the LayoutLMv3 model.

        Args:
            inputs: Dictionary containing:
                - input_ids: Int tensor of shape (batch_size, sequence_length)
                - bbox: Int tensor of shape (batch_size, sequence_length, 4)
                - attention_mask: Int tensor of shape (batch_size, sequence_length)
                - image: Float tensor of shape (batch_size, height, width, channels)

        Returns:
            Dictionary containing:
                - sequence_output: Float tensor of shape (batch_size, sequence_length, hidden_size)
                - pooled_output: Float tensor of shape (batch_size, hidden_size)
                - hidden_states: List of tensors of shape (batch_size, sequence_length, hidden_size)

        Example:
        ```python
        outputs = backbone({
            "input_ids": input_ids,
            "bbox": bbox,
            "attention_mask": attention_mask,
            "image": image
        })
        sequence_output = outputs["sequence_output"]
        pooled_output = outputs["pooled_output"]
        ```
        """
        input_ids = inputs["input_ids"]
        bbox = inputs["bbox"]
        attention_mask = inputs["attention_mask"]
        image = inputs["image"]
        
        # Get sequence length
        seq_length = backend.shape(input_ids)[1]
        
        # Create position IDs
        position_ids = backend.arange(seq_length, dtype="int32")
        position_embeddings = self.position_embeddings(position_ids)
        
        # Get spatial embeddings
        x_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        y_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 2])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 3])
        
        # Project spatial embeddings to hidden size
        x_position_embeddings = self.x_proj(x_position_embeddings)
        y_position_embeddings = self.y_proj(y_position_embeddings)
        h_position_embeddings = self.h_proj(h_position_embeddings)
        w_position_embeddings = self.w_proj(w_position_embeddings)
        
        # Get word embeddings and token type embeddings
        word_embeddings = self.word_embeddings(input_ids)
        token_type_ids = backend.zeros_like(input_ids[:, 0:1])
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        token_type_embeddings = backend.broadcast_to(
            token_type_embeddings,
            [backend.shape(input_ids)[0], backend.shape(input_ids)[1], self.hidden_size],
        )
        
        # Combine all embeddings
        text_embeddings = (
            word_embeddings
            + position_embeddings
            + x_position_embeddings
            + y_position_embeddings
            + h_position_embeddings
            + w_position_embeddings
            + token_type_embeddings
        )
        
        # Process image
        patch_embeddings = self.patch_embed(image)
        batch_size = backend.shape(patch_embeddings)[0]
        patch_embeddings_shape = backend.shape(patch_embeddings)
        num_patches = patch_embeddings_shape[1] * patch_embeddings_shape[2]
        patch_embeddings = backend.reshape(
            patch_embeddings, [batch_size, num_patches, self.hidden_size]
        )
        patch_embeddings = self.patch_embed_layer_norm(patch_embeddings)
        
        # Combine text and image embeddings
        x = backend.concatenate([text_embeddings, patch_embeddings], axis=1)
        
        # Add CLS token
        cls_tokens = backend.broadcast_to(
            self.cls_token, [backend.shape(x)[0], 1, self.hidden_size]
        )
        x = backend.concatenate([cls_tokens, x], axis=1)
        
        # Apply layer normalization
        x = self.embeddings_LayerNorm(x)
        
        # Create attention mask
        new_seq_length = backend.shape(x)[1]
        extended_attention_mask = backend.ones(
            (backend.shape(input_ids)[0], new_seq_length), dtype="int32"
        )
        extended_attention_mask = backend.cast(
            extended_attention_mask[:, None, None, :],
            dtype="float32",
        )
        extended_attention_mask = backend.broadcast_to(
            extended_attention_mask,
            [
                backend.shape(input_ids)[0],
                1,
                new_seq_length,
                new_seq_length,
            ],
        )
        
        # Apply transformer layers
        hidden_states = []
        for layer in self.encoder_layers:
            x = layer(x, extended_attention_mask)
            hidden_states.append(x)
        
        # Get sequence output and pooled output
        sequence_output = x
        pooled_output = self.pooler(sequence_output[:, 0])
        
        return {
            "sequence_output": sequence_output,
            "pooled_output": pooled_output,
            "hidden_states": hidden_states,
        }
    
    def get_config(self) -> Dict:
        """Get the model configuration.

        Returns:
            Dictionary containing the model configuration.
        """
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_act": self.hidden_act,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "max_position_embeddings": self.max_position_embeddings,
            "type_vocab_size": self.type_vocab_size,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "pad_token_id": self.pad_token_id,
            "position_embedding_type": self.position_embedding_type,
            "use_cache": self.use_cache,
            "classifier_dropout": self.classifier_dropout,
            "patch_size": self.patch_size,
            "num_channels": self.num_channels,
            "qkv_bias": self.qkv_bias,
            "use_abs_pos": self.use_abs_pos,
            "use_rel_pos": self.use_rel_pos,
            "rel_pos_bins": self.rel_pos_bins,
            "max_rel_pos": self.max_rel_pos,
            "spatial_embedding_dim": self.spatial_embedding_dim,
        })
        return config 