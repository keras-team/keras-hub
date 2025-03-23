import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.gemma.rms_normalization import RMSNormalization
from keras_hub.src.models.gemma3.gemma3_decoder_block import Gemma3DecoderBlock
from keras_hub.src.models.gemma3.gemma3_interleave_embeddings import (
    Gemma3InterleaveEmbeddings,
)
from keras_hub.src.models.gemma3.gemma3_vit import Gemma3Vit


@keras_hub_export("keras_hub.models.Gemma3Backbone")
class Gemma3Backbone(Backbone):
    """Gemma3 core network with hyperparameters.

    This backbone implements the mixed-modality Gemma3 architecture. It
    contains a Visual Transformer network, as well as text token embedding
    layer, followed by a backend-agnostic concatenation operation to
    construct a sequence of representations of mixed type embeddings (visual
    and textual). Then, the concatenated sequence is passed through a series
    of Mixed Modality Decoder Blocks. The returned value from calling this model
    represents probabilistic values for output tokens.

    For a higher-level object for text-generation,
    see `keras_hub.models.Gemma3CausalLM`.

    The default constructor gives a fully customizable, randomly initialized
    Gemma3 model with any number of vit layers, heads, embedding
    dimensions, and equivalent configuration for Gemma3 Decoder layers. To
    load preset architectures and weights, use the `from_preset` constructor.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        image_size: int. The resolution of the image in both width and height.
            The input images must be square.
        num_layers: int. The number of transformer mixed decoder layers.
        num_query_heads: int. The number of heads for the query projections in
            the mixed decoder attention layer.
        num_key_value_heads: int. The number of heads for the key and value
            projections in the mixed decoder attention layers.
        hidden_dim: int. The size of the transformer hidden state at the end
            of each mixed transformer layer.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer decoder block.
        head_dim: int. The size of each attention head in the mixed decoder.
        vit_patch_size: int. The size of each square patch in the input image.
        vit_num_heads: int. The number of attention heads for the vision (image)
            transformer encoder.
        vit_hidden_dim: int. The size of the transformer hidden state at the end
            of each vision transformer layer.
        vit_num_layers: int. The number of vision transformer layers.
        vit_intermediate_dim: int. The output dimension of the first Dense layer
            in a two-layer feedforward network for vision transformer. Defaults
            to `4304`.
        vit_pool_size: int. Factor by which to downscale `(dim1, dim2)` in the
            average pooling layer. The same value is used for `"strides"`.
        vit_name: string. The name used for vision transformer layers.
        query_head_dim_normalize: boolean. If `True` normalize the query before
            attention with `head_dim`. If `False`, normalize the query with
            `hidden_dim / num_query_heads`. Defaults to `True`.
        use_query_key_norm: bool. If `True`, apply a RMS Norm layer to query and
            key before projecting them. Defaults to `True`.
        use_post_ffw_norm: boolean. Whether to normalize after the feedforward
            block. Defaults to `False`.
        use_post_attention_norm: boolean. Whether to normalize after the
            attention block. Defaults to `False`.
        attention_logit_soft_cap: `None` or int. Soft cap for the attention
            logits. Defaults to `None`.
        final_logit_soft_cap: `None` or int. Soft cap for the final logits.
            Defaults to `None`.
        use_sliding_window_attention: boolean. Whether to use sliding local
          window attention. Defaults to `False`.
        sliding_window_size: int. Size of the sliding local window. Defaults to
            `4096`.
        layer_norm_epsilon: float. The epsilon value user for every layer norm
            in all transformer blocks. Defaults to `1e-6`.
        dropout: float. Dropout probability for the Transformer decoder blocks.
            Defaults to `0`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the models computations and weights. Note that some
            computations, such as softmax and layer normalization will always
            be done a float32 precision regardless of dtype.

    Example:
    ```python
    input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "images": np.random.uniform(size=(1, 224, 224, 3)),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
    }

    # Pretrained Gemma3 decoder.
    model = keras_hub.models.Gemma3Backbone.from_preset("gemma3")
    model(input_data)

    # Randomly initialized Gemma3 decoder with custom config.
    model = keras_hub.models.Gemma3Backbone(
        vocabulary_size=50257,
        images_size=224,
        num_layers=12,
        num_query_heads=12,
        num_key_value_heads=1,
        hidden_dim=768,
        intermediate_dim=3072,
        head_dim=64,
        vit_patch_size=14,
        vit_num_heads=8,
        vit_hidden_dim=768,
        vit_intermediate_dim=3072,
        vit_num_layers=2,
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        image_size,
        num_layers,
        num_query_heads,
        num_key_value_heads,
        hidden_dim,
        intermediate_dim,
        head_dim,
        vit_patch_size,
        vit_num_heads,
        vit_hidden_dim,
        vit_num_layers,
        vit_intermediate_dim,
        vit_pool_size=None,
        vit_name=None,
        query_head_dim_normalize=True,
        use_query_key_norm=True,
        use_post_ffw_norm=False,
        use_post_attention_norm=False,
        attention_logit_soft_cap=None,
        final_logit_soft_cap=None,
        use_sliding_window_attention=False,
        sliding_window_size=4096,
        layer_norm_epsilon=1e-6,
        dropout=0,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=True,
            embeddings_initializer=keras.initializers.VarianceScaling(
                scale=1.0,
                mode="fan_in",
                distribution="untruncated_normal",
                seed=None,
            ),
            dtype=dtype,
            logit_soft_cap=final_logit_soft_cap,
            name="token_embedding",
        )

        self.vit_encoder = Gemma3Vit(
            image_size=image_size,
            patch_size=vit_patch_size,
            num_heads=vit_num_heads,
            hidden_dim=vit_hidden_dim,
            num_layers=vit_num_layers,
            intermediate_dim=vit_intermediate_dim,
            output_dim=hidden_dim,
            pool_size=vit_pool_size,
            dtype=dtype,
            name=vit_name,
        )

        self.interleave_embeddings = Gemma3InterleaveEmbeddings(
            num_vision_tokens_per_image=self.vit_encoder.num_vision_tokens_per_image,
            dtype=dtype,
            name="interleave_embeddings",
        )

        self.transformer_layers = []
        for i in range(num_layers):
            # 5 local, 1 global
            sliding_window = use_sliding_window_attention and (i % 6 < 5)
            rope_wavelength = 10_000.0 if sliding_window else 1_000_000.0
            rope_scaling_factor = 1.0 if sliding_window else 8.0
            layer = Gemma3DecoderBlock(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                head_dim=head_dim,
                num_query_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads,
                query_head_dim_normalize=query_head_dim_normalize,
                use_query_key_norm=use_query_key_norm,
                use_post_ffw_norm=use_post_ffw_norm,
                use_post_attention_norm=use_post_attention_norm,
                gate_dim_reduction=1,
                logit_soft_cap=attention_logit_soft_cap,
                use_sliding_window_attention=sliding_window,
                sliding_window_size=sliding_window_size,
                rope_wavelength=rope_wavelength,
                rope_scaling_factor=rope_scaling_factor,
                dropout=dropout,
                dtype=dtype,
                name=f"decoder_block_{i}",
            )
            self.transformer_layers.append(layer)
        self.layer_norm = RMSNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="final_normalization",
        )

        # === Functional Model ===

        # == Model inputs ==
        image_input = keras.Input(
            shape=(None, image_size, image_size, 3),
            name="images",
        )
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        response_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="response_mask"
        )
        vision_indices_input = keras.Input(
            shape=(None,), dtype="int32", name="vision_indices"
        )
        text_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="text_mask"
        )

        # == Image Embeddings ==
        img_embeddings = self.vit_encoder(image_input)

        # == Text embeddings ==
        text_embeddings = self.token_embedding(token_id_input)

        text_embeddings = text_embeddings * ops.cast(
            ops.sqrt(hidden_dim), text_embeddings.dtype
        )

        ## == Interleaving text and images ==
        # Place the image embeddings in the right position in `text_embeddings`.
        x = self.interleave_embeddings(
            image_embeddings=img_embeddings,
            text_embeddings=text_embeddings,
            vision_indices=vision_indices_input,
        )

        # == Decoder layers ==
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(
                x,
                padding_mask=padding_mask_input,
                response_mask=response_mask_input,
                text_mask=text_mask_input,
            )
        sequence_output = self.layer_norm(x)

        super().__init__(
            inputs={
                "images": image_input,
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
                "response_mask": response_mask_input,
                "vision_indices": vision_indices_input,
                "text_mask": text_mask_input,
            },
            outputs=sequence_output,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.image_size = image_size
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.head_dim = head_dim
        self.query_head_dim_normalize = query_head_dim_normalize
        self.use_query_key_norm = use_query_key_norm
        self.use_post_ffw_norm = use_post_ffw_norm
        self.use_post_attention_norm = use_post_attention_norm
        self.attention_logit_soft_cap = attention_logit_soft_cap
        self.final_logit_soft_cap = final_logit_soft_cap
        self.use_sliding_window_attention = use_sliding_window_attention
        self.sliding_window_size = sliding_window_size
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout

        # ViT params
        self.vit_patch_size = vit_patch_size
        self.vit_num_heads = vit_num_heads
        self.vit_hidden_dim = vit_hidden_dim
        self.vit_num_layers = vit_num_layers
        self.vit_intermediate_dim = vit_intermediate_dim
        self.vit_pool_size = vit_pool_size
        self.vit_name = vit_name

        # Keep the num_vision_tokens_per_image as a backbone property for easy
        # access.
        self.num_vision_tokens_per_image = (
            self.vit_encoder.num_vision_tokens_per_image
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "image_size": self.image_size,
                "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "head_dim": self.head_dim,
                "query_head_dim_normalize": self.query_head_dim_normalize,
                "use_query_key_norm": self.use_query_key_norm,
                "use_post_ffw_norm": self.use_post_ffw_norm,
                "use_post_attention_norm": self.use_post_attention_norm,
                "attention_logit_soft_cap": self.attention_logit_soft_cap,
                "final_logit_soft_cap": self.final_logit_soft_cap,
                "use_sliding_window_attention": (
                    self.use_sliding_window_attention
                ),
                "sliding_window_size": self.sliding_window_size,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                # ViT
                "vit_patch_size": self.vit_patch_size,
                "vit_num_heads": self.vit_num_heads,
                "vit_hidden_dim": self.vit_hidden_dim,
                "vit_num_layers": self.vit_num_layers,
                "vit_intermediate_dim": self.vit_intermediate_dim,
                "vit_pool_size": self.vit_pool_size,
                "vit_name": self.vit_name,
            }
        )
        return config
