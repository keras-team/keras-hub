"""BLIP-2 backbone model."""

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
      3. A `BLIP2CustomOPT` language model receives the query embeddings
         prepended to its token sequence and autoregressively generates text.

    When `vision_encoder` is `None` the backbone operates in text-only mode:
    the Q-Former is bypassed and the language model receives only token ids.

    For a higher-level text-generation interface see
    `keras_hub.models.BLIP2CausalLM`.

    Args:
        vision_encoder: A `keras_hub.models.BLIP2VisionEncoder` instance. Pass
            `None` for a text-only backbone.
        qformer: A `keras_hub.models.BLIP2QFormer` instance. Pass `None` when
            `vision_encoder` is `None`.
        language_model: A `keras_hub.models.BLIP2CustomOPT` instance.
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
        # === Layers ===
        self.vision_encoder = vision_encoder
        self.qformer = qformer
        self.language_model = language_model

        multimodal = self.vision_encoder is not None

        # === Inputs ===
        token_ids_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        inputs = {
            "token_ids": token_ids_input,
            "padding_mask": padding_mask_input,
        }

        # === Vision branch (optional) ===
        if multimodal:
            image_size = self.vision_encoder.image_size
            images_input = keras.Input(
                shape=(image_size, image_size, 3),
                dtype="float32",
                name="pixel_values",
            )
            inputs["images"] = images_input

            # Stage 1 – frozen ViT: image → patch features
            patch_features = self.vision_encoder(images_input)

            # Stage 2 – Q-Former bridge: patch features → query embeddings
            query_embeddings = self.qformer(patch_features)
        else:
            query_embeddings = None

        # === Language model ===
        # Stage 3 – LLM: (query embeddings +) token ids → sequence output
        lm_inputs = {
            "token_ids": token_ids_input,
            "padding_mask": padding_mask_input,
        }
        if query_embeddings is not None:
            lm_inputs["qformer_features"] = query_embeddings

        output = self.language_model(lm_inputs)

        super().__init__(
            inputs=inputs,
            outputs=output,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.multimodal = multimodal

    # === Public properties for downstream task heads ===

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
        # Freeze biases of LoRA-enabled layers so they are fully covered by
        # save_weights/load_weights (base weights). Without this, biases get
        # trained but are NOT saved by save_lora_weights(), causing a mismatch
        # on reload.
        #
        # _lora_enabled_layers holds variable indices, not layer indices, so
        # we derive the affected layer path prefixes from the lora_kernel paths.
        lora_prefixes = set()
        for v in self.variables:
            if "lora_kernel" in v.path:
                # e.g. "encoder/block_1/mha/query/lora_kernel_a"
                # → prefix = "encoder/block_1/mha/query"
                prefix = v.path.rsplit("/lora_kernel", 1)[0]
                lora_prefixes.add(prefix)

        for v in self.variables:
            if "bias" in v.path and "lora_kernel" not in v.path:
                prefix = v.path.rsplit("/bias", 1)[0]
                if prefix in lora_prefixes:
                    v.trainable = False

    # === Serialization ===

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
