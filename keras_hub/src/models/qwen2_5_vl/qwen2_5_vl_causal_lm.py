import keras
from keras import ops

from qwen2_5_vl_backbone import Qwen2_5_VLBackbone
from qwen2_5_vl_presets import PRESETS


@keras.saving.register_keras_serializable(package="keras_hub")
class Qwen2_5_VLCausalLM(keras.Model):
    """
    Qwen2.5-VL Causal Language Model.

    Wraps Qwen2_5_VLBackbone with a weight-tied language model head that
    projects hidden states to vocabulary logits.

    Parameters
    ----------
    backbone : Qwen2_5_VLBackbone
    tie_embeddings : bool
        If True (default), the LM head reuses the token embedding matrix.

    Inputs (dict)
    -------------
    token_ids      : (B, S) int32                required
    pixel_values   : (B, T, H, W, 3) float32     optional
    image_mask     : (B, S) bool                 required if pixel_values given
    attention_mask : (B, 1, S, S) float32        optional

    Returns
    -------
    logits : (B, S, vocab_size) float32
    """

    def __init__(self, backbone, tie_embeddings=True, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.tie_embeddings = tie_embeddings

        if not tie_embeddings:
            self.lm_head = keras.layers.Dense(
                backbone.vocab_size,
                use_bias=False,
                name="lm_head",
            )

    def build(self, input_shape=None):
        self.built = True

    def call(self, inputs, training=False):
        hidden_states = self.backbone(inputs, training=training)

        if self.tie_embeddings:
            embedding_weights = self.backbone.token_embedding.embeddings
            logits = ops.matmul(hidden_states, ops.transpose(embedding_weights))
        else:
            logits = self.lm_head(hidden_states)

        return logits

    def generate(self, inputs, max_length=20, eos_token_id=None):
        """
        Greedy autoregressive generation.

        Parameters
        ----------
        inputs : dict
            token_ids     : (B, S) int32
            pixel_values  : (B, T, H, W, 3) float32  optional
            image_mask    : (B, S) bool               optional
        max_length : int
        eos_token_id : int or None

        Returns
        -------
        token_ids : (B, S + generated_length) int32
        """
        token_ids    = inputs["token_ids"]
        pixel_values = inputs.get("pixel_values", None)
        image_mask   = inputs.get("image_mask",   None)

        for _ in range(max_length):
            model_inputs = {"token_ids": token_ids}
            if pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values
                model_inputs["image_mask"]   = image_mask

            logits     = self(model_inputs, training=False)
            next_token = ops.argmax(logits[:, -1, :], axis=-1)
            next_token = ops.cast(
                ops.expand_dims(next_token, axis=-1), token_ids.dtype
            )
            token_ids  = ops.concatenate([token_ids, next_token], axis=1)

            if eos_token_id is not None:
                if bool(ops.all(next_token == eos_token_id)):
                    break

        return token_ids

    @classmethod
    def from_preset(cls, name, tie_embeddings=True, dtype="float32", **kwargs):
        """
        Instantiate from a named preset.

        Parameters
        ----------
        name : str
            One of the keys in PRESETS.
        tie_embeddings : bool
        dtype : str
            Compute dtype. Use "bfloat16" for P100/A100.
        """
        if name not in PRESETS:
            raise ValueError(
                f"Unknown preset '{name}'. Available: {list(PRESETS.keys())}"
            )
        cfg = PRESETS[name]
        backbone = Qwen2_5_VLBackbone(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            num_kv_heads=cfg["num_kv_heads"],
            intermediate_size=cfg["intermediate_size"],
            vision_hidden_size=cfg["vision_hidden_size"],
            vision_num_heads=cfg["vision_num_heads"],
            vision_intermediate_size=cfg["vision_intermediate_size"],
            vision_num_layers=cfg["vision_num_layers"],
            patch_size=cfg["patch_size"],
            window_size=cfg["window_size"],
            vision_projector_intermediate_multiplier=cfg.get(
                "vision_projector_intermediate_multiplier", 2
            ),
            dtype=dtype,
        )
        return cls(backbone=backbone, tie_embeddings=tie_embeddings, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "backbone": keras.saving.serialize_keras_object(self.backbone),
            "tie_embeddings": self.tie_embeddings,
        })
        return config

    @classmethod
    def from_config(cls, config):
        config["backbone"] = keras.saving.deserialize_keras_object(
            config.pop("backbone")
        )
        return cls(**config)