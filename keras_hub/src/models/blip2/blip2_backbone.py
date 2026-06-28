import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone


@keras_hub_export("keras_hub.models.BLIP2Backbone")
class BLIP2Backbone(Backbone):
    """BLIP-2 core network.

    BLIP-2 is a vision-language model that connects a frozen image encoder
    and a frozen large language model (LLM) through a lightweight trainable
    Querying Transformer (Q-Former). The Q-Former distills visual information
    into a fixed number of query embeddings which are then fed as a soft visual
    prompt to the language model.

    The forward pass follows three stages:
      1. A `BLIP2VisionEncoder` (ViT) maps raw images to patch features.
      2. A `BLIP2QFormer` cross-attends learned query tokens against those
         patch features and produces a compact set of visual embeddings.
      3. A language model (OPT, Flan-T5, or Vicuna) receives the query
         embeddings prepended to its token sequence and generates text.

    When `vision_encoder` is `None` the backbone operates in text-only mode:
    the Q-Former is bypassed and the language model receives only token ids.

    For a higher-level text-generation interface see
    `keras_hub.models.BLIP2CausalLM`.

    Args:
        vision_encoder: A `keras_hub.models.BLIP2VisionEncoder` instance. Pass
            `None` for a text-only backbone.
        qformer: A `keras_hub.models.BLIP2QFormer` instance. Pass `None` when
            `vision_encoder` is `None`.
        language_model: The language model instance (e.g. `BLIP2CustomOPT`,
            `BLIP2FlanT5`, or `BLIP2Vicuna`).
        dtype: string or `keras.mixed_precision.DTypePolicy`. Dtype used for
            model computations and weights. Defaults to `None` (Keras global
            default).
        **kwargs: Additional keyword arguments forwarded to the base
            `Backbone` / `keras.Model`.

    Example:
    ```python
    # Text-only (no vision encoder)
    backbone = keras_hub.models.BLIP2Backbone(
        vision_encoder=None,
        qformer=None,
        language_model=my_opt,
    )
    output = backbone({"token_ids": token_ids, "padding_mask": mask})

    # Full vision-language backbone
    backbone = keras_hub.models.BLIP2Backbone.from_preset("blip2_opt_2.7b")
    output = backbone({
        "images": images,           # (B, H, W, 3)
        "token_ids": token_ids,     # (B, seq_len)
        "padding_mask": mask,       # (B, seq_len)
    })
    ```
    """

    def __init__(
        self,
        vision_encoder,
        qformer,
        language_model,
        dtype=None,
        **kwargs,
    ):
        self.vision_encoder = vision_encoder
        self.qformer = qformer
        self.language_model = language_model

        multimodal = self.vision_encoder is not None

        # Encoder-decoder language models (Flan-T5) follow the keras-hub
        # seq2seq convention and name their encoder inputs
        # `encoder_token_ids` / `encoder_padding_mask`; decoder-only language
        # models (OPT, Vicuna) keep the plain `token_ids` / `padding_mask`.
        is_encoder_decoder = hasattr(
            self.language_model, "encoder_transformer_layers"
        )
        if is_encoder_decoder:
            token_ids_name, padding_mask_name = (
                "encoder_token_ids",
                "encoder_padding_mask",
            )
        else:
            token_ids_name, padding_mask_name = "token_ids", "padding_mask"

        token_ids_input = keras.Input(
            shape=(None,), dtype="int32", name=token_ids_name
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name=padding_mask_name
        )
        inputs = {
            token_ids_name: token_ids_input,
            padding_mask_name: padding_mask_input,
        }

        if multimodal:
            raw_image_size = self.vision_encoder.image_size
            if isinstance(raw_image_size, (tuple, list)):
                image_h, image_w = (
                    int(raw_image_size[0]),
                    int(raw_image_size[1]),
                )
            else:
                image_h = image_w = int(raw_image_size)

            images_input = keras.Input(
                shape=(image_h, image_w, 3),
                dtype="float32",
                name="images",
            )
            inputs["images"] = images_input

            patch_features = self.vision_encoder(images_input)
            if getattr(self.qformer, "instruction_aware", False):
                qformer_token_ids_input = keras.Input(
                    shape=(None,), dtype="int32", name="qformer_token_ids"
                )
                qformer_padding_mask_input = keras.Input(
                    shape=(None,), dtype="int32", name="qformer_padding_mask"
                )
                inputs["qformer_token_ids"] = qformer_token_ids_input
                inputs["qformer_padding_mask"] = qformer_padding_mask_input
                query_embeddings = self.qformer(
                    {
                        "vision_features": patch_features,
                        "qformer_token_ids": qformer_token_ids_input,
                        "qformer_padding_mask": qformer_padding_mask_input,
                    }
                )
            else:
                query_embeddings = self.qformer(patch_features)
        else:
            query_embeddings = None

        # The language model itself always consumes `token_ids` /
        # `padding_mask` for its (encoder) input; the backbone maps the
        # externally exposed names onto these.
        lm_inputs = {
            "token_ids": token_ids_input,
            "padding_mask": padding_mask_input,
        }

        if is_encoder_decoder:
            decoder_token_ids_input = keras.Input(
                shape=(None,), dtype="int32", name="decoder_token_ids"
            )
            decoder_padding_mask_input = keras.Input(
                shape=(None,), dtype="int32", name="decoder_padding_mask"
            )
            inputs["decoder_token_ids"] = decoder_token_ids_input
            inputs["decoder_padding_mask"] = decoder_padding_mask_input
            lm_inputs["decoder_token_ids"] = decoder_token_ids_input
            lm_inputs["decoder_padding_mask"] = decoder_padding_mask_input

        if query_embeddings is not None:
            lm_inputs["qformer_features"] = query_embeddings

        output = self.language_model(lm_inputs)

        super().__init__(
            inputs=inputs,
            outputs=output,
            dtype=dtype,
            **kwargs,
        )

        self.multimodal = multimodal

    @property
    def token_embedding(self):
        """The token embedding layer of the language model."""
        return self.language_model.token_embedding

    @property
    def num_query_tokens(self):
        """Number of Q-Former query tokens (0 in text-only mode)."""
        return self.qformer.num_query_tokens if self.qformer is not None else 0

    @property
    def qformer_hidden_dim(self):
        """Hidden dimensionality of the Q-Former (0 in text-only mode)."""
        return self.qformer.hidden_dim if self.qformer is not None else 0

    def enable_lora(self, rank):
        super().enable_lora(rank)
        lora_prefixes = set()
        for v in self.variables:
            if "lora_kernel" in v.path:
                prefix = v.path.rsplit("/lora_kernel", 1)[0]
                lora_prefixes.add(prefix)

        for v in self.variables:
            if "bias" in v.path and "lora_kernel" not in v.path:
                prefix = v.path.rsplit("/bias", 1)[0]
                if prefix in lora_prefixes:
                    v.trainable = False

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vision_encoder": (
                    keras.layers.serialize(self.vision_encoder)
                    if self.vision_encoder is not None
                    else None
                ),
                "qformer": (
                    keras.layers.serialize(self.qformer)
                    if self.qformer is not None
                    else None
                ),
                "language_model": keras.layers.serialize(self.language_model),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        for key in ("vision_encoder", "qformer"):
            if config.get(key) is not None:
                config[key] = keras.layers.deserialize(config[key])
        config["language_model"] = keras.layers.deserialize(
            config["language_model"]
        )
        return cls(**config)
