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


@keras_hub_export("keras_hub.models.Gemma3Backbone")
class Gemma3Backbone(Backbone):
    """Gemma3 core network with hyperparameters.

    This backbone implements the Gemma3 model architecture. Gemma3 is a
    vision-language model (image-text in, text out). The text input is encoded
    using an embedding layer; images are encoded using a vision transformer
    (ViT). After encoding these two modalities, the image embeddings are placed
    in the correct position in the text embedding sequence. The mixed sequence
    of embeddings is then passed through transformer decoder layers.

    For a higher-level object for text-generation, see
    `keras_hub.models.Gemma3CausalLM`.

    The default constructor gives a fully customizable, randomly initialized
    Gemma3 model with any vision encoder, number of heads, embedding dimensions,
    and equivalent configuration for the decoder layers. To load preset
    architectures and weights, use the `from_preset` constructor.

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
        vision_encoder: A `Gemma3VisionEncoder` instance. `call()`
            takes in images and returns corresponding sequence of embeddings. If
            `None`, the model is a text-only model.
        layer_norm_epsilon: float. The epsilon value user for every layer norm
            in all transformer blocks. Defaults to `1e-6`.
        dropout: float. Dropout probability for the Transformer decoder blocks.
            Defaults to `0`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the models computations and weights. Note that some
            computations, such as softmax and layer normalization will always
            be done in float32 precision regardless of dtype. Defaults to
            `bfloat16`.

    Example:
    ```python
    # === Language Gemma3 model ===
    input_data = {}
    input_data["token_ids"] = np.ones(shape=(1, 300), dtype="int32")
    input_data["padding_mask"] = (
        np.expand_dims(np.array([1] * 280 + [0] * (300 - 280)), axis=0)
        .astype(bool)
    )

    # Pretrained Gemma3 decoder.
    model = keras_hub.models.Gemma3Backbone.from_preset(
        "gemma3_instruct_4b_text"
    )
    model(input_data)

    # Randomly initialized Gemma3 decoder with a custom config.
    model = keras_hub.models.Gemma3Backbone(
        vocabulary_size=262144,
        image_size=896,
        num_layers=34,
        num_query_heads=8,
        num_key_value_heads=4,
        hidden_dim=2560,
        intermediate_dim=10240,
        head_dim=256,
        query_head_dim_normalize=True,
        use_post_ffw_norm=True,
        use_post_attention_norm=True,
        final_logit_soft_cap=None,
        attention_logit_soft_cap=None,
        sliding_window_size=1024,
        use_sliding_window_attention=True,
        vision_encoder=None,
        layer_norm_epsilon=1e-06,
        dtype="bfloat16",
    )
    model(input_data)

    # === Vision + Language Gemma3 model ===
    input_data = {}
    input_data["images"] = np.ones(shape=(1, 1, 896, 896, 3))
    input_data["token_ids"] = np.ones(shape=(1, 300), dtype="int32")
    # images after the text part of the sequence.
    input_data["vision_mask"] = np.expand_dims(
        np.array([0] * 30 + [1] * 256 + [0] * 14),
        axis=0,
    ).astype(bool)
    input_data["vision_indices"] = (
        np.expand_dims(np.arange(30, 286), axis=0)
    )
    input_data["padding_mask"] = (
        np.expand_dims(np.array([1] * 286 + [0] * (300 - 286)), axis=0)
        .astype(bool)
    )

    # Pretrained Gemma3 decoder.
    model = keras_hub.models.Gemma3Backbone.from_preset("gemma3_instruct_4b")
    model(input_data)

    # Randomly initialized Gemma3 decoder with a custom config.
    vision_encoder = Gemma3VisionEncoder(
        image_size=896,
        patch_size=14,
        num_heads=16,
        hidden_dim=1152,
        num_layers=27,
        intermediate_dim=4304,
        output_dim=2560,
        pool_size=4,
        layer_norm_epsilon=1e-6,
        dtype="float32",
    )

    model = keras_hub.models.Gemma3Backbone(
        vocabulary_size=262144,
        image_size=896,
        num_layers=34,
        num_query_heads=8,
        num_key_value_heads=4,
        hidden_dim=2560,
        intermediate_dim=10240,
        head_dim=256,
        query_head_dim_normalize=True,
        use_post_ffw_norm=True,
        use_post_attention_norm=True,
        final_logit_soft_cap=None,
        attention_logit_soft_cap=None,
        sliding_window_size=1024,
        use_sliding_window_attention=True,
        vision_encoder=vision_encoder,
        layer_norm_epsilon=1e-06,
        dtype="bfloat16"
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
        query_head_dim_normalize=True,
        use_query_key_norm=True,
        use_post_ffw_norm=False,
        use_post_attention_norm=False,
        attention_logit_soft_cap=None,
        final_logit_soft_cap=None,
        use_sliding_window_attention=False,
        sliding_window_size=1024,
        local_rope_scaling_factor=1.0,
        global_rope_scaling_factor=1.0,
        vision_encoder=None,
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

        self.vision_encoder = vision_encoder
        text_only_model = True if vision_encoder is None else False
        if not text_only_model:
            self.interleave_embeddings = Gemma3InterleaveEmbeddings(
                num_vision_tokens_per_image=self.vision_encoder.num_vision_tokens_per_image,
                dtype=dtype,
                name="interleave_embeddings",
            )

        self.transformer_layers = []
        for i in range(num_layers):
            # 5 local, 1 global
            sliding_window = use_sliding_window_attention and (i % 6 < 5)
            rope_wavelength = 10_000.0 if sliding_window else 1_000_000.0
            rope_scaling_factor = (
                local_rope_scaling_factor
                if sliding_window
                else global_rope_scaling_factor
            )
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
        if not text_only_model:
            image_input = keras.Input(
                shape=(None, image_size, image_size, 3),
                name="images",
            )
            vision_indices_input = keras.Input(
                shape=(None,), dtype="int32", name="vision_indices"
            )
            # Truth be told, this is redundant, and we can infer this from
            # `vision_indices_input`. But it is easier to return this from
            # the preprocessor than to compute it here.
            vision_mask_input = keras.Input(
                shape=(None,), dtype="int32", name="vision_mask"
            )

        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        # == Text embeddings ==
        text_embeddings = self.token_embedding(token_id_input)

        text_embeddings = text_embeddings * ops.cast(
            ops.sqrt(hidden_dim), text_embeddings.dtype
        )

        # == Image Embeddings ==
        if not text_only_model:
            img_embeddings = self.vision_encoder(image_input)

            # == Interleaving text and images ==
            # Place image embeddings in the right position in
            # `text_embeddings`.
            x = self.interleave_embeddings(
                image_embeddings=img_embeddings,
                text_embeddings=text_embeddings,
                vision_indices=vision_indices_input,
            )
        else:
            x = text_embeddings

        # == Decoder layers ==
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(
                x,
                padding_mask=padding_mask_input,
                vision_mask=None if text_only_model else vision_mask_input,
            )
        sequence_output = self.layer_norm(x)

        inputs = {
            "token_ids": token_id_input,
            "padding_mask": padding_mask_input,
        }
        if not text_only_model:
            inputs.update(
                {
                    "images": image_input,
                    "vision_indices": vision_indices_input,
                    "vision_mask": vision_mask_input,
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
        self.local_rope_scaling_factor = local_rope_scaling_factor
        self.global_rope_scaling_factor = global_rope_scaling_factor
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout

        # Keep `num_vision_tokens_per_image` as a backbone property for easy
        # access.
        if not text_only_model:
            self.num_vision_tokens_per_image = (
                self.vision_encoder.num_vision_tokens_per_image
            )
        # Also, the `text_only_model`.
        self.text_only_model = text_only_model

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
                "local_rope_scaling_factor": self.local_rope_scaling_factor,
                "global_rope_scaling_factor": self.global_rope_scaling_factor,
                "vision_encoder": None
                if self.vision_encoder is None
                else keras.layers.serialize(self.vision_encoder),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
            }
        )
        return config

    def default_lora_layer_names(self):
        target_names = super().default_lora_layer_names()

        # Add these for `Gemma3VITAttention`.
        if not self.text_only_model:
            target_names += ["query_proj", "value_proj"]
        return target_names

    @classmethod
    def from_config(cls, config):
        config.update(
            {
                "vision_encoder": None
                if config["vision_encoder"] is None
                else keras.layers.deserialize(config["vision_encoder"]),
            }
        )

        return super().from_config(config)
