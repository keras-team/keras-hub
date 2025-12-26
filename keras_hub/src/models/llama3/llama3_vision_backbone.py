"""Llama 3.2 Vision Backbone with Cross-Attention fusion."""

import keras
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.llama3.llama3_backbone import Llama3Backbone
from keras_hub.src.models.llama3.llama3_vision_cross_attention import (
    Llama3VisionCrossAttention,
)
from keras_hub.src.models.llama3.llama3_vision_encoder import (
    Llama3VisionEncoder,
)
from keras_hub.src.models.llama3.llama3_vision_projector import (
    Llama3VisionProjector,
)

# Default cross-attention layer positions (as per Meta's architecture)
# These are the layers where visual features are injected via cross-attention
DEFAULT_CROSS_ATTENTION_LAYERS = [3, 8, 13, 18, 23, 28, 33, 38]


@keras_hub_export("keras_hub.models.Llama3VisionBackbone")
class Llama3VisionBackbone(Backbone):
    """Llama 3.2 Vision Backbone model with Cross-Attention fusion.

    This model implements the Llama 3.2 Vision architecture which uses
    cross-attention layers to progressively inject visual features into
    the language model at specific decoder layer positions.

    Unlike early-fusion approaches that concatenate image and text tokens,
    this architecture keeps vision and text processing separate, using
    gated cross-attention at specific layers (default: [3,8,13,18,23,28,33,38])
    to integrate visual information.

    Args:
        config: `Llama3VisionConfig` instance containing:
            - vision_encoder_config: Configuration for the vision encoder.
            - text_config: Configuration for the text backbone.
            - cross_attention_layers: List of layer indices for cross-attention.
            - dtype: Data type for computations.

    Example:
    ```python
    from keras_hub.models import Llama3VisionConfig, Llama3VisionBackbone

    config = Llama3VisionConfig(
        vision_encoder_config={
            "hidden_dim": 1152,
            "num_layers": 27,
            "num_heads": 16,
            "patch_size": 14,
            "image_size": 560,
        },
        text_config={
            "vocabulary_size": 128256,
            "num_layers": 40,
            "hidden_dim": 4096,
            "num_query_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_dim": 14336,
        },
        cross_attention_layers=[3, 8, 13, 18, 23, 28, 33, 38],
    )

    backbone = Llama3VisionBackbone(config)
    ```
    """

    def __init__(self, config, **kwargs):
        # 1. Validate Configs
        if config.vision_encoder_config is None:
            raise ValueError("`vision_encoder_config` must be provided.")
        if config.text_config is None:
            raise ValueError("`text_config` must be provided.")

        # Get cross-attention layer positions
        cross_attention_layers = getattr(
            config, "cross_attention_layers", DEFAULT_CROSS_ATTENTION_LAYERS
        )

        # 2. Define Inputs (Symbolic Tensors)
        # Images: (Batch, Height, Width, Channels)
        image_input = keras.Input(
            shape=(
                config.vision_encoder_config.image_size,
                config.vision_encoder_config.image_size,
                3,
            ),
            name="images",
            dtype=config.dtype,
        )
        # Tokens: (Batch, Sequence_Length)
        token_id_input = keras.Input(
            shape=(None,),
            name="token_ids",
            dtype="int32",
        )
        # Padding Mask: (Batch, Sequence_Length)
        padding_mask_input = keras.Input(
            shape=(None,),
            name="padding_mask",
            dtype="bool",
        )

        # 3. Instantiate Sub-Components
        # Vision Encoder (SigLIP-based ViT)
        vision_encoder = Llama3VisionEncoder(
            hidden_dim=config.vision_encoder_config.hidden_dim,
            num_layers=config.vision_encoder_config.num_layers,
            num_heads=config.vision_encoder_config.num_heads,
            intermediate_dim=config.vision_encoder_config.intermediate_dim,
            patch_size=config.vision_encoder_config.patch_size,
            image_size=config.vision_encoder_config.image_size,
            num_channels=config.vision_encoder_config.num_channels,
            local_layers=getattr(
                config.vision_encoder_config, "local_layers", None
            ),
            global_layers=getattr(
                config.vision_encoder_config, "global_layers", None
            ),
            activation=config.vision_encoder_config.activation,
            dropout=config.vision_encoder_config.dropout,
            attention_dropout=config.vision_encoder_config.attention_dropout,
            layer_norm_epsilon=config.vision_encoder_config.layer_norm_epsilon,
            dtype=config.dtype,
            name="vision_encoder",
        )

        # Extract text config parameters
        if hasattr(config.text_config, "hidden_dim"):
            text_hidden_dim = config.text_config.hidden_dim
            text_num_query_heads = getattr(
                config.text_config, "num_query_heads", 32
            )
            text_num_kv_heads = getattr(
                config.text_config, "num_key_value_heads", 8
            )
            text_layer_norm_eps = getattr(
                config.text_config, "layer_norm_epsilon", 1e-5
            )
        elif isinstance(config.text_config, dict):
            text_hidden_dim = config.text_config.get("hidden_dim", 4096)
            text_num_query_heads = config.text_config.get("num_query_heads", 32)
            text_num_kv_heads = config.text_config.get("num_key_value_heads", 8)
            text_layer_norm_eps = config.text_config.get(
                "layer_norm_epsilon", 1e-5
            )
        else:
            text_hidden_dim = 4096
            text_num_query_heads = 32
            text_num_kv_heads = 8
            text_layer_norm_eps = 1e-5

        # Vision Projector (projects vision features to text embedding space)
        vision_projector = Llama3VisionProjector(
            hidden_dim=config.vision_encoder_config.hidden_dim,
            output_dim=text_hidden_dim,
            activation="gelu",
            dtype=config.dtype,
            name="vision_projector",
        )

        # Text Backbone
        if hasattr(config.text_config, "get_config"):
            text_config_dict = config.text_config.get_config()
        elif isinstance(config.text_config, dict):
            text_config_dict = config.text_config.copy()
        else:
            raise ValueError(
                "text_config must be either a dict or have a "
                "get_config() method"
            )

        # Remove dtype to avoid duplicate argument
        text_config_dict.pop("dtype", None)
        text_backbone = Llama3Backbone(
            **text_config_dict,
            dtype=config.dtype,
            name="text_backbone",
        )

        # Cross-Attention layers (placed at specific decoder positions)
        cross_attention_blocks = {}
        for layer_idx in cross_attention_layers:
            if layer_idx < len(text_backbone.transformer_layers):
                cross_attention_blocks[layer_idx] = Llama3VisionCrossAttention(
                    hidden_dim=text_hidden_dim,
                    num_heads=text_num_query_heads,
                    num_key_value_heads=text_num_kv_heads,
                    layer_norm_epsilon=text_layer_norm_eps,
                    dtype=config.dtype,
                    name=f"cross_attention_{layer_idx}",
                )

        # ----------------------------------------------------------------------
        # 4. Build the Graph (Functional Composition with Cross-Attention)
        # ----------------------------------------------------------------------

        # A. Vision Path: Encode and project to text dimension
        vision_features = vision_encoder(image_input)
        vision_features = vision_projector(vision_features)
        # Shape: (B, num_patches, text_hidden_dim)

        # Create vision mask (all True since all patches are valid)
        def create_vision_mask(vision_feats):
            batch_size = ops.shape(vision_feats)[0]
            num_patches = ops.shape(vision_feats)[1]
            return ops.ones((batch_size, num_patches), dtype="bool")

        vision_mask = layers.Lambda(
            create_vision_mask, name="create_vision_mask"
        )(vision_features)

        # B. Text Path: Embed tokens
        x = text_backbone.token_embedding(token_id_input)
        # Shape: (B, seq_len, text_hidden_dim)

        # C. Process through decoder with cross-attention at specific layers
        for i, transformer_layer in enumerate(text_backbone.transformer_layers):
            # Standard self-attention + FFN
            x = transformer_layer(x, decoder_padding_mask=padding_mask_input)

            # Cross-attention injection at specific layers
            if i in cross_attention_blocks:
                x = cross_attention_blocks[i](
                    hidden_states=x,
                    vision_features=vision_features,
                    vision_mask=vision_mask,
                )

        # D. Final layer norm
        x = text_backbone.layer_norm(x)

        # 5. Initialize the Functional Model
        model_dtype = kwargs.pop("dtype", config.dtype)

        super().__init__(
            inputs={
                "images": image_input,
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
            },
            outputs=x,
            dtype=model_dtype,
            **kwargs,
        )

        # 6. Store references for access and serialization
        self.config = config
        self.vision_encoder = vision_encoder
        self.vision_projector = vision_projector
        self.text_backbone = text_backbone
        self.cross_attention_blocks = cross_attention_blocks
        self.cross_attention_layers = cross_attention_layers

    def get_config(self):
        config = super().get_config()
        config.update({"config": self.config.get_config()})
        return config

    @classmethod
    def from_config(cls, config):
        from keras_hub.src.models.llama3.llama3_vision_config import (
            Llama3VisionConfig,
        )

        config_data = config.pop("config")
        vision_config = Llama3VisionConfig(**config_data)
        return cls(config=vision_config, **config)

    # =========================================================================
    # Fine-Tuning Utilities
    # =========================================================================

    def freeze_vision_encoder(self):
        """Freeze the entire vision encoder.

        Useful for fine-tuning only the language model and cross-attention
        while keeping pretrained vision features fixed.

        Example:
        ```python
        backbone = Llama3VisionBackbone(config)
        backbone.freeze_vision_encoder()
        # Train only text backbone and cross-attention
        ```
        """
        self.vision_encoder.trainable = False
        self.vision_projector.trainable = False

    def freeze_text_backbone(self):
        """Freeze the text backbone (Llama3).

        Useful for adapter-style training where you fine-tune only
        the vision encoder and cross-attention.
        """
        self.text_backbone.trainable = False

    def freeze_cross_attention(self):
        """Freeze all cross-attention layers.

        Useful when you want to fine-tune vision and text separately
        without modifying the fusion mechanism.
        """
        for ca_block in self.cross_attention_blocks.values():
            ca_block.trainable = False

    def freeze_for_vision_adapter_training(self):
        """Freeze everything except cross-attention layers.

        This is the recommended fine-tuning strategy for Llama 3.2 Vision:
        - Vision encoder: FROZEN (pretrained SigLIP)
        - Text backbone: FROZEN (pretrained Llama 3.1)
        - Cross-attention: TRAINABLE (adapter)
        - Vision projector: TRAINABLE (adapter)

        Example:
        ```python
        backbone = Llama3VisionBackbone(config)
        backbone.freeze_for_vision_adapter_training()
        # Only cross-attention and vision projector are trainable
        ```
        """
        self.vision_encoder.trainable = False
        self.text_backbone.trainable = False
        self.vision_projector.trainable = True

        for ca_block in self.cross_attention_blocks.values():
            ca_block.trainable = True

    def freeze_for_lora_training(self):
        """Freeze all weights for LoRA-style training.

        Call this before adding LoRA adapters to the model.
        """
        self.vision_encoder.trainable = False
        self.vision_projector.trainable = False
        self.text_backbone.trainable = False

        for ca_block in self.cross_attention_blocks.values():
            ca_block.trainable = False

    def unfreeze_all(self):
        """Unfreeze all model weights.

        Restores all layers to trainable state for full fine-tuning.
        """
        self.vision_encoder.trainable = True
        self.vision_projector.trainable = True
        self.text_backbone.trainable = True

        for ca_block in self.cross_attention_blocks.values():
            ca_block.trainable = True

        # Also unfreeze vision encoder sublayers
        if hasattr(self.vision_encoder, "unfreeze_all"):
            self.vision_encoder.unfreeze_all()

    def get_trainable_summary(self):
        """Get a summary of trainable components.

        Returns:
            Dict with component trainability status.
        """

        def count_params(layer, trainable_only=True):
            if trainable_only:
                return sum(
                    int(np.prod(w.shape)) for w in layer.trainable_weights
                )
            return sum(int(np.prod(w.shape)) for w in layer.weights)

        import numpy as np

        summary = {
            "vision_encoder": {
                "trainable": self.vision_encoder.trainable,
                "params": count_params(
                    self.vision_encoder, trainable_only=False
                ),
            },
            "vision_projector": {
                "trainable": self.vision_projector.trainable,
                "params": count_params(
                    self.vision_projector, trainable_only=False
                ),
            },
            "text_backbone": {
                "trainable": self.text_backbone.trainable,
                "params": count_params(
                    self.text_backbone, trainable_only=False
                ),
            },
            "cross_attention": {
                "trainable_layers": sum(
                    1
                    for ca in self.cross_attention_blocks.values()
                    if ca.trainable
                ),
                "total_layers": len(self.cross_attention_blocks),
            },
        }

        total_trainable = sum(
            int(np.prod(w.shape)) for w in self.trainable_weights
        )
        total_params = self.count_params()

        summary["total"] = {
            "trainable_params": total_trainable,
            "total_params": total_params,
            "trainable_ratio": f"{total_trainable/total_params*100:.1f}%",
        }

        return summary

    def print_trainable_summary(self):
        """Print a formatted summary of trainable components."""
        summary = self.get_trainable_summary()

        print("\n" + "=" * 50)
        print("TRAINABLE PARAMETERS SUMMARY")
        print("=" * 50)

        for component, info in summary.items():
            if component == "total":
                continue
            if isinstance(info, dict) and "trainable" in info:
                status = "✓ TRAINABLE" if info["trainable"] else "✗ FROZEN"
                params = info.get("params", "N/A")
                print(f"  {component:20s}: {status:15s} ({params:,} params)")
            elif isinstance(info, dict) and "trainable_layers" in info:
                trainable_str = (
                    f"{info['trainable_layers']}/{info['total_layers']} "
                    "layers trainable"
                )
                print(f"  {component:20s}: {trainable_str}")

        print("-" * 50)
        total = summary["total"]
        trainable_str = (
            f"{total['trainable_params']:,} / {total['total_params']:,}"
        )
        print(f"  Total trainable: {trainable_str}")
        print(f"  Trainable ratio: {total['trainable_ratio']}")
        print("=" * 50)
