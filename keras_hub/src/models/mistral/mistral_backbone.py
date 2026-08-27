import keras
from keras import ops
from keras.layers import ReversibleEmbedding
from keras.layers import RMSNormalization

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.mistral.mistral_transformer_decoder import (
    MistralTransformerDecoder,
)
from keras_hub.src.models.mistral.mistral_vision_encoder import (
    Mistral3ImageFeatureExtractor,
)
from keras_hub.src.models.mistral.mistral_vision_encoder import (
    Mistral3ImageTextEmbeddingMerger,
)


def _mistral_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


@keras_hub_export("keras_hub.models.MistralBackbone")
class MistralBackbone(Backbone):
    """
    The Mistral Transformer core architecture with hyperparameters.

    This network implements a Transformer-based decoder network,
    Mistral, as described in
    ["Mistral 7B"](https://arxiv.org/pdf/2310.06825.pdf).
    It includes the embedding lookups and transformer layers.

    The default constructor gives a fully customizable, randomly initialized
    Mistral model with any number of layers, heads, and embedding
    dimensions. To load preset architectures and weights, use the `from_preset`
    constructor.

    Args:
        vocabulary_size (int): The size of the token vocabulary.
        num_layers (int): The number of transformer layers.
        num_query_heads (int): The number of query attention heads for
            each transformer.
        hidden_dim (int): The size of the transformer encoding and pooling
            layers.
        intermediate_dim (int): The output dimension of the first Dense layer
            in a three-layer feedforward network for each transformer.
        num_key_value_heads (int): The number of key and value attention heads
            for each transformer.
        rope_max_wavelength (int, optional): The maximum angular wavelength of
            the sine/cosine curves, for rotary embeddings. Defaults to `10000`.
        rope_scaling_factor (float, optional): The scaling factor for
            calculation of roatary embedding. Defaults to `1.0`.
        layer_norm_epsilon (float, optional): Epsilon for the layer
            normalization layers in the transformer decoder. Defaults to `1e-6`.
        sliding_window (int, optional): The sliding window for the mistral
            attention layers. This controls the maximum cache size for the
            attention layers in each transformer decoder. Only `sliding_window`
            number of tokens are saved in the cache and used to generate the
            next token. Defaults to `512`. Pass `None` to disable sliding
            window attention entirely (e.g. Magistral).
        head_dim (int, optional): The size of each attention head. When
            `None` (the default), falls back to `hidden_dim // num_query_heads`.
            Set explicitly when the model's head size is not equal to
            `hidden_dim // num_query_heads` — e.g. Magistral uses
            `head_dim=128` with `hidden_dim=5120` and `num_query_heads=32`.
        vision_encoder: A `keras_hub.models.Mistral3VisionEncoder` instance,
            or `None` (the default) for a text-only model. When set,
            `multimodal_projector` must also be provided, and `call()`
            accepts additional `"pixel_values"`, `"image_sizes"`, and
            `"placeholder_indices"` inputs. `"pixel_values"` has dynamic
            spatial dimensions (padded to the batch's largest image, as
            produced by HF's image processor), not a fixed canvas;
            `"image_sizes"` gives each image's true `(height, width)` for
            cropping after patchification. `"placeholder_indices"` gives
            the flat positions of image placeholder tokens in `token_ids`
            (into the flattened `batch * seq_length` sequence) that get
            replaced with projected image features — compute it with
            `mistral_vision_encoder.compute_image_placeholder_indices`
            before calling the model, since deriving it in-graph would be
            incompatible with `jax.jit` tracing.
        multimodal_projector: A `Mistral3MultiModalProjector` instance.
            Required when `vision_encoder` is set.
        image_token_index: int, optional. The token ID in `token_ids` that
            marks image placeholder positions. Defaults to `10`. Unused in
            text-only mode; used together with
            `compute_image_placeholder_indices` to build the
            `"placeholder_indices"` model input.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights. Note that some computations,
            such as softmax and layer normalization, will always be done at
            float32 precision regardless of dtype.

    Examples:

    ```python
    input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
    }

    # Pretrained Mistral decoder.
    model = keras_hub.models.MistralBackbone.from_preset("mistral7b_base_en")
    model(input_data)

    # Randomly initialized Mistral decoder with custom config.
    model = keras_hub.models.MistralBackbone(
        vocabulary_size=10,
        hidden_dim=512,
        num_layers=2,
        num_query_heads=32,
        num_key_value_heads=8,
        intermediate_dim=1024,
        sliding_window=512,
        layer_norm_epsilon=1e-6,
        dtype="float32"
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_query_heads,
        hidden_dim,
        intermediate_dim,
        num_key_value_heads,
        rope_max_wavelength=10000,
        rope_scaling_factor=1.0,
        layer_norm_epsilon=1e-6,
        sliding_window=512,
        head_dim=None,
        dropout=0,
        vision_encoder=None,
        multimodal_projector=None,
        image_token_index=10,
        dtype=None,
        **kwargs,
    ):
        if (vision_encoder is None) != (multimodal_projector is None):
            raise ValueError(
                "`vision_encoder` and `multimodal_projector` must be "
                "provided together for a multimodal `MistralBackbone`."
            )
        text_only_model = vision_encoder is None

        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=False,
            embeddings_initializer=_mistral_kernel_initializer(stddev=0.01),
            dtype=dtype,
            name="token_embedding",
        )
        self.transformer_layers = []
        for i in range(num_layers):
            layer = MistralTransformerDecoder(
                intermediate_dim=intermediate_dim,
                num_query_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads,
                rope_max_wavelength=rope_max_wavelength,
                rope_scaling_factor=rope_scaling_factor,
                layer_norm_epsilon=layer_norm_epsilon,
                activation=ops.silu,
                kernel_initializer=_mistral_kernel_initializer(stddev=0.02),
                sliding_window=sliding_window,
                head_dim=head_dim,
                dropout=dropout,
                dtype=dtype,
                name=f"transformer_layer_{i}",
            )
            self.transformer_layers.append(layer)
        self.layer_norm = RMSNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="sequence_output_layernorm",
        )
        self.vision_encoder = vision_encoder
        self.multimodal_projector = multimodal_projector
        if not text_only_model:
            self.image_text_embedding_merger = Mistral3ImageTextEmbeddingMerger(
                dtype=dtype,
                name="image_text_embedding_merger",
            )

        # === Functional Model ===
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        x = self.token_embedding(token_id_input)

        if not text_only_model:
            # `None` spatial dims: HF's `PixtralImageProcessor` pads each
            # batch to its own largest image, not to a fixed canvas, so the
            # input canvas size varies per call. `image_sizes` carries each
            # image's true (unpadded) `(height, width)` for cropping.
            pixel_values_input = keras.Input(
                shape=(vision_encoder.num_channels, None, None),
                name="pixel_values",
            )
            image_sizes_input = keras.Input(
                shape=(2,), dtype="int32", name="image_sizes"
            )
            # Flat positions of image placeholder tokens in the flattened
            # `(batch * seq_length,)` sequence. Computed on the host (e.g.
            # via `compute_image_placeholder_indices`) rather than derived
            # in-graph, since a `nonzero`-style lookup has a data-dependent
            # output shape that is incompatible with `jax.jit` tracing.
            placeholder_indices_input = keras.Input(
                shape=(None,),
                dtype="int32",
                name="placeholder_indices",
            )
            self.image_feature_extractor = Mistral3ImageFeatureExtractor(
                vision_encoder,
                multimodal_projector,
                dtype=dtype,
                name="image_feature_extractor",
            )
            image_features = self.image_feature_extractor(
                pixel_values_input,
                image_sizes_input,
            )
            x = self.image_text_embedding_merger(
                x, image_features, placeholder_indices_input
            )

        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, decoder_padding_mask=padding_mask_input)
        sequence_output = self.layer_norm(x)

        inputs = {
            "token_ids": token_id_input,
            "padding_mask": padding_mask_input,
        }
        if not text_only_model:
            inputs.update(
                {
                    "pixel_values": pixel_values_input,
                    "image_sizes": image_sizes_input,
                    "placeholder_indices": placeholder_indices_input,
                }
            )

        super().__init__(
            inputs=inputs,
            outputs=sequence_output,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.rope_max_wavelength = rope_max_wavelength
        self.num_key_value_heads = num_key_value_heads
        self.rope_scaling_factor = rope_scaling_factor
        self.sliding_window = sliding_window
        self.head_dim = head_dim
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout
        self.image_token_index = image_token_index
        self.text_only_model = text_only_model

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "num_key_value_heads": self.num_key_value_heads,
                "sliding_window": self.sliding_window,
                "head_dim": self.head_dim,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "image_token_index": self.image_token_index,
                "vision_encoder": None
                if self.vision_encoder is None
                else keras.layers.serialize(self.vision_encoder),
                "multimodal_projector": None
                if self.multimodal_projector is None
                else keras.layers.serialize(self.multimodal_projector),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.update(
            {
                "vision_encoder": None
                if config["vision_encoder"] is None
                else keras.layers.deserialize(config["vision_encoder"]),
                "multimodal_projector": None
                if config["multimodal_projector"] is None
                else keras.layers.deserialize(config["multimodal_projector"]),
            }
        )
        return super().from_config(config)
