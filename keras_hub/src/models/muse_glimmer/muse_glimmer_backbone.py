import keras
from keras import ops
from keras.layers import ReversibleEmbedding

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.muse_glimmer.muse_glimmer_decoder import (
    MuseGlimmerTextDecoder,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_layers import (
    MuseGlimmerInterleaveEmbeddings,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_layers import (
    MuseGlimmerRMSNorm,
)


@keras_hub_export("keras_hub.models.MuseGlimmerBackbone")
class MuseGlimmerBackbone(Backbone):
    """MuseGlimmer core network with hyperparameters.

    MuseGlimmer is a dense causal decoder-transformer (Gemma-lineage
    normalization/softcapping, Llama3-lineage BPE tokenizer) optionally
    fused with a windowed-attention perception encoder
    (`MuseGlimmerVisionEncoder`) for image-text-to-text generation.

    The text tower alternates sliding-window and full attention every 4
    layers; full-attention layers additionally receive no rotary position
    encoding (NoPE). Each layer applies a scaleless QK-RMSNorm plus an
    extra `qk_scale_factor` on the query, and gates the attention output by
    a sigmoid computed from the pre-attention normed input. When a vision
    encoder is attached, its output passes through a double-activation
    adapter, a projection, and a final scaleless RMSNorm before being
    scattered into the text embedding sequence — see `modeling_muse_glimmer.py`,
    `MuseGlimmerModel.get_image_features`.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer decoder layers.
        num_query_heads: int. The number of query attention heads.
        num_key_value_heads: int. The number of key/value attention heads.
        hidden_dim: int. The transformer hidden dimension.
        intermediate_dim: int. The SwiGLU MLP intermediate dimension.
        head_dim: int. Per-head dimension.
        sliding_window_size: int. Sliding window size for
            `"sliding_attention"` layers. Defaults to `2048`.
        rope_max_wavelength: float. RoPE base theta for layers that use
            rotary position embeddings. Defaults to `500000.0`.
        rms_norm_eps: float. Epsilon for pre-sublayer norms, QK-norm, the
            normed embedding, and the final sequence norm. Defaults to
            `1e-5`.
        post_norm_eps: float. Epsilon for the post-sublayer sandwich
            norms. Defaults to `1e-8`.
        qk_scale_factor: float. Extra multiplier applied to Q after
            QK-norm, on top of the standard `1/sqrt(head_dim)` attention
            scaling. Defaults to `3.87`.
        output_multiplier: float. Scale applied to logits before the final
            tanh softcap. Defaults to `0.19611613513818404`.
        final_logit_softcapping: float. Softcap value `T` in
            `T * tanh(logits / T)`. Defaults to `20.0`.
        layer_types: list of str or `None`. Per-layer `"sliding_attention"`
            or `"full_attention"`. Defaults to every 4th layer (counted
            from the end) being full attention.
        vision_encoder: A `MuseGlimmerVisionEncoder` instance, or `None`
            for a text-only model.
        projector_hidden_dim: int or `None`. Intermediate/output dimension
            of the multimodal adapter. Required if `vision_encoder` is set.
        projector_hidden_act: str. Activation for the adapter and the
            top-level projection. Defaults to `"gelu"`.
        dropout: float. Dropout probability. Defaults to `0`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to
            use for model computations and weights.
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_query_heads,
        num_key_value_heads,
        hidden_dim,
        intermediate_dim,
        head_dim,
        sliding_window_size=2048,
        rope_max_wavelength=500000.0,
        rms_norm_eps=1e-5,
        post_norm_eps=1e-8,
        qk_scale_factor=3.87,
        output_multiplier=0.19611613513818404,
        final_logit_softcapping=20.0,
        layer_types=None,
        vision_encoder=None,
        projector_hidden_dim=None,
        projector_hidden_act="gelu",
        dropout=0,
        dtype=None,
        **kwargs,
    ):
        if layer_types is None:
            layer_types = [
                "full_attention"
                if (num_layers - 1 - i) % 4 == 0
                else "sliding_attention"
                for i in range(num_layers)
            ]

        # === Layers ===
        # The normed-embedding scaleless RMSNorm is folded into the
        # embedding lookup itself (`MuseGlimmerTextNormedEmbedding` in HF),
        # applied right after `token_embedding` in the functional graph
        # below.
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=False,
            dtype=dtype,
            name="token_embedding",
        )
        self.embed_norm = MuseGlimmerRMSNorm(
            eps=rms_norm_eps, with_scale=False, dtype=dtype, name="embed_norm"
        )

        self.vision_encoder = vision_encoder
        text_only_model = vision_encoder is None
        if not text_only_model:
            self.interleave_embeddings = MuseGlimmerInterleaveEmbeddings(
                hidden_dim=hidden_dim, dtype=dtype, name="interleave_embeddings"
            )
            # Multimodal fusion: double-activation adapter (no bias), a
            # separate projection Linear (no bias), then a scaleless
            # RMSNorm — matches `MuseGlimmerVisionAdapter` +
            # `MuseGlimmerModel.get_image_features`'s extra projection/norm.
            self.vision_adapter_fc1 = keras.layers.Dense(
                projector_hidden_dim,
                use_bias=False,
                dtype=dtype,
                name="vision_adapter_fc1",
            )
            self.vision_adapter_fc2 = keras.layers.Dense(
                projector_hidden_dim,
                use_bias=False,
                dtype=dtype,
                name="vision_adapter_fc2",
            )
            self.vision_projection = keras.layers.Dense(
                hidden_dim,
                use_bias=False,
                dtype=dtype,
                name="vision_projection",
            )
            self.perception_emb_norm = MuseGlimmerRMSNorm(
                eps=rms_norm_eps,
                with_scale=False,
                dtype=dtype,
                name="perception_emb_norm",
            )
            self.projector_activation = keras.activations.get(
                projector_hidden_act
            )

        self.transformer_layers = []
        for i in range(num_layers):
            is_full_attention = layer_types[i] == "full_attention"
            layer = MuseGlimmerTextDecoder(
                intermediate_dim=intermediate_dim,
                num_query_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                rms_norm_eps=rms_norm_eps,
                post_norm_eps=post_norm_eps,
                qk_scale_factor=qk_scale_factor,
                # NoPE on full-attention layers; RoPE otherwise.
                use_rope=not is_full_attention,
                rope_max_wavelength=rope_max_wavelength,
                sliding_window_size=(
                    None if is_full_attention else sliding_window_size
                ),
                dropout=dropout,
                dtype=dtype,
                name=f"transformer_layer_{i}",
            )
            self.transformer_layers.append(layer)

        self.layer_norm = MuseGlimmerRMSNorm(
            eps=rms_norm_eps,
            dtype=dtype,
            name="sequence_output_layernorm",
        )

        # === Functional Model ===
        if not text_only_model:
            pixel_values_input = keras.Input(
                shape=(None, None), name="pixel_values"
            )
            image_grid_thw_input = keras.Input(
                shape=(None, 3), dtype="int32", name="image_grid_thw"
            )
            vision_indices_input = keras.Input(
                shape=(None,), dtype="int32", name="vision_indices"
            )

        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        x = self.token_embedding(token_id_input)
        x = self.embed_norm(x)

        if not text_only_model:
            img_embeddings = self.vision_encoder(
                pixel_values_input, image_grid_thw_input
            )
            img_embeddings = self.projector_activation(
                self.vision_adapter_fc1(img_embeddings)
            )
            img_embeddings = self.projector_activation(
                self.vision_adapter_fc2(img_embeddings)
            )
            img_embeddings = self.vision_projection(img_embeddings)
            img_embeddings = self.perception_emb_norm(img_embeddings)
            x = self.interleave_embeddings(
                image_embeddings=img_embeddings,
                text_embeddings=x,
                vision_indices=vision_indices_input,
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
                    "image_grid_thw": image_grid_thw_input,
                    "vision_indices": vision_indices_input,
                }
            )

        super().__init__(
            inputs=inputs, outputs=sequence_output, dtype=dtype, **kwargs
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.head_dim = head_dim
        self.sliding_window_size = sliding_window_size
        self.rope_max_wavelength = rope_max_wavelength
        self.rms_norm_eps = rms_norm_eps
        self.post_norm_eps = post_norm_eps
        self.qk_scale_factor = qk_scale_factor
        self.output_multiplier = output_multiplier
        self.final_logit_softcapping = final_logit_softcapping
        self.layer_types = layer_types
        self.projector_hidden_dim = projector_hidden_dim
        self.projector_hidden_act = projector_hidden_act
        self.dropout = dropout
        self.text_only_model = text_only_model

    def __call__(self, inputs, *args, **kwargs):
        """Inject empty vision inputs for text-only calls on a VLM backbone."""
        if isinstance(inputs, dict) and not self.text_only_model:
            inputs = {
                key: ops.convert_to_tensor(value)
                for key, value in inputs.items()
            }
            batch_size = ops.shape(inputs["token_ids"])[0]
            if "pixel_values" not in inputs:
                patch_dim = (
                    self.vision_encoder.patch_temporal
                    * 3
                    * self.vision_encoder.patch_size**2
                )
                inputs["pixel_values"] = ops.zeros(
                    (batch_size, 0, patch_dim), dtype="float32"
                )
            if "image_grid_thw" not in inputs:
                inputs["image_grid_thw"] = ops.zeros(
                    (batch_size, 0, 3), dtype="int32"
                )
            if "vision_indices" not in inputs:
                inputs["vision_indices"] = ops.zeros(
                    (batch_size, 0), dtype="int32"
                )
        return super().__call__(inputs, *args, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "head_dim": self.head_dim,
                "sliding_window_size": self.sliding_window_size,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rms_norm_eps": self.rms_norm_eps,
                "post_norm_eps": self.post_norm_eps,
                "qk_scale_factor": self.qk_scale_factor,
                "output_multiplier": self.output_multiplier,
                "final_logit_softcapping": self.final_logit_softcapping,
                "layer_types": self.layer_types,
                "projector_hidden_dim": self.projector_hidden_dim,
                "projector_hidden_act": self.projector_hidden_act,
                "dropout": self.dropout,
                "vision_encoder": None
                if self.vision_encoder is None
                else keras.layers.serialize(self.vision_encoder),
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
            }
        )
        return super().from_config(config)
