from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.modern_bert.modern_bert_backbone import (
    ModernBertBackbone,
)


@keras_hub_export("keras_hub.models.MMBertBackbone")
class MMBertBackbone(ModernBertBackbone):
    """mmBERT backbone model.

    mmBERT is a multilingual encoder that reuses the ModernBERT architecture:
    Rotary Positional Embeddings (RoPE), GeGLU activations, Layer Normalization
    and alternating local/global attention layers. The layers, the arguments
    and their defaults, and the computation are inherited unchanged from
    `keras_hub.models.ModernBertBackbone`.

    The differences are in the configuration, which the checkpoint supplies:

    * The vocabulary is the 256k Gemma-2 SentencePiece vocabulary, so
      `vocabulary_size` is `256000` (the ModernBERT default is `50368`).
    * The local RoPE table uses a max wavelength of `160000`, like the global
      one (the ModernBERT default for `local_rotary_max_wavelength` is
      `10000`).
    * The tokenizer is a byte-level BPE tokenizer instead of a WordPiece one;
      see `keras_hub.models.MMBertTokenizer`.

    The values used by the released mmBERT checkpoints are `vocabulary_size =
    256000`, `hidden_dim = 768`, `intermediate_dim = 1152`, `num_layers = 22`,
    `num_heads = 12`, `local_attention_window = 128`,
    `global_attn_every_n_layers = 3`, `rotary_max_wavelength = 160000`,
    `local_rotary_max_wavelength = 160000` and `layer_norm_epsilon = 1e-5`.

    The backbone takes a dict with `token_ids` and `padding_mask` and returns
    the hidden states of the final layer.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        hidden_dim: int. The size of the transformer hidden state.
        intermediate_dim: int. The output dimension of the GeGLU MLP.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads.
        local_attention_window: int. Window size for local attention layers.
        global_attn_every_n_layers: int. Frequency of global attention layers.
        rotary_max_wavelength: int. Max wavelength for RoPE.
        local_rotary_max_wavelength: int or None. Max wavelength for local
            RoPE. If `None`, uses `rotary_max_wavelength`.
        layer_norm_epsilon: float. Epsilon used by the Layer Normalization
            layers.
        dtype: string or `keras.DTypePolicy`. The dtype of the layers.
            Defaults to `None`.

    Examples:
    ```python
    import keras_hub
    import numpy as np

    # Instantiate an mmBERT backbone.
    backbone = keras_hub.models.MMBertBackbone(vocabulary_size=256000)

    # Prepare dummy input data.
    input_data = {
        "token_ids": np.random.randint(0, 256000, size=(2, 512), dtype="int32"),
        "padding_mask": np.ones((2, 512), dtype="int32"),
    }

    # Extract hidden states.
    outputs = backbone(input_data)
    ```
    """

    # mmBERT presets are added once the converted checkpoints are uploaded,
    # so clear the mapping inherited from `ModernBertBackbone`.
    presets = {}
