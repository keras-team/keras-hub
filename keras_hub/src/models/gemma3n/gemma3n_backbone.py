import inspect

import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.gemma3n.gemma3n_audio_encoder import (
    Gemma3nAudioEncoder,
)
from keras_hub.src.models.gemma3n.gemma3n_text_model import Gemma3nTextModel
from keras_hub.src.models.gemma3n.rms_normalization import Gemma3nRMSNorm
from keras_hub.src.models.mobilenetv5.mobilenetv5_backbone import (
    MobileNetV5Backbone,
)


class Gemma3nMultimodalEmbedder(keras.layers.Layer):
    """A layer for handling multimodal embeddings.

    This layer manages embeddings for different modalities (here, vision, text,
    and audio). It can take either token IDs or pre-computed embedding vectors
    as input. The embeddings are normalized and projected to match the text
    model's hidden size.

    Args:
        multimodal_hidden_size: int. The hidden size of the multimodal
            embeddings.
        text_hidden_size: int. The hidden size of the text model.
        rms_norm_eps: float. The epsilon value for the Gemma 3n RMS
            normalization layers.
        vocab_offset: int. The vocabulary offset for the specific modality.
        vocab_size: int. The vocabulary size for the specific modality.
    """

    def __init__(
        self,
        multimodal_hidden_size,
        text_hidden_size,
        rms_norm_eps,
        vocab_offset,
        vocab_size,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.multimodal_hidden_size = multimodal_hidden_size
        self.text_hidden_size = text_hidden_size
        self.rms_norm_eps = rms_norm_eps
        self.vocab_offset = vocab_offset
        self.vocab_size = vocab_size
        self.embedding = keras.layers.Embedding(
            vocab_size,
            multimodal_hidden_size,
            name="embedding",
            dtype=self.dtype_policy,
        )
        self.hard_embedding_norm = Gemma3nRMSNorm(
            multimodal_hidden_size,
            eps=rms_norm_eps,
            name="hard_embedding_norm",
            dtype=self.dtype_policy,
        )
        self.soft_embedding_norm = Gemma3nRMSNorm(
            multimodal_hidden_size,
            eps=rms_norm_eps,
            name="soft_embedding_norm",
            dtype=self.dtype_policy,
        )
        self.embedding_projection = keras.layers.Dense(
            text_hidden_size,
            use_bias=False,
            name="embedding_projection",
            dtype=self.dtype_policy,
        )
        self.embedding_post_projection_norm = Gemma3nRMSNorm(
            text_hidden_size,
            eps=rms_norm_eps,
            with_scale=False,
            name="embedding_post_projection_norm",
            dtype=self.dtype_policy,
        )

    def build(self, input_shape):
        embeds_shape = (None, None, self.multimodal_hidden_size)
        self.hard_embedding_norm.build(embeds_shape)
        self.soft_embedding_norm.build(embeds_shape)
        self.embedding_projection.build(embeds_shape)
        proj_shape = (None, None, self.text_hidden_size)
        self.embedding_post_projection_norm.build(proj_shape)
        self.embedding.build((None, None))
        super().build(input_shape)

    def call(self, inputs):
        input_ids, inputs_embeds = None, None
        if isinstance(inputs, list):
            input_ids, inputs_embeds = inputs
        elif "int" in str(inputs.dtype):
            input_ids = inputs
        else:
            inputs_embeds = inputs
        if (input_ids is None) and (inputs_embeds is None):
            raise ValueError(
                "You must specify either input_ids or inputs_embeds"
            )
        if (input_ids is not None) and (inputs_embeds is not None):
            raise ValueError(
                "You can only specify one of input_ids or inputs_embeds"
            )
        if inputs_embeds is not None:
            emb_norm = self.soft_embedding_norm(inputs_embeds)
        else:
            index_to_lookup = input_ids - self.vocab_offset
            hard_emb = self.embedding(index_to_lookup)
            emb_norm = self.hard_embedding_norm(hard_emb)

        emb_norm_proj = self.embedding_projection(emb_norm)
        return self.embedding_post_projection_norm(emb_norm_proj)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "multimodal_hidden_size": self.multimodal_hidden_size,
                "text_hidden_size": self.text_hidden_size,
                "rms_norm_eps": self.rms_norm_eps,
                "vocab_offset": self.vocab_offset,
                "vocab_size": self.vocab_size,
            }
        )
        return config


class Gemma3nMultimodalEmbeddingProcessor(keras.layers.Layer):
    """Processes and interleaves text, vision, and audio embeddings.

    This layer takes raw token IDs and multimodal inputs (pixel values, audio
    features) and produces a final sequence of embeddings ready for the
    decoder. It handles the embedding lookup for text and special tokens,
    and replaces the special tokens with the processed features from the
    vision and audio encoders.

    Args:
        language_model: `keras_hub.models.gemma3n.Gemma3nTextModel`. The
            underlying text model containing embedding layers.
        vision_encoder: `keras.Model`. The vision encoder model.
        embed_vision: `keras_hub.models.gemma3n.Gemma3nMultimodalEmbedder`. The
            embedder for vision.
        audio_encoder: `keras_hub.models.gemma3n.Gemma3nAudioEncoder`. The audio
            encoder model.
        embed_audio: `keras_hub.models.gemma3n.Gemma3nMultimodalEmbedder`. The
            embedder for audio.
        vision_soft_tokens_per_image: int. Number of tokens to represent an
            image.
        audio_soft_tokens_per_image: int. Number of tokens to represent an
            audio clip.
        image_token_id: int. The special token ID for images.
        audio_token_id: int. The special token ID for audio.
        vocab_size_per_layer_input: int. The vocabulary size for per-layer
            inputs.
    """

    def __init__(
        self,
        language_model,
        vision_encoder,
        embed_vision,
        audio_encoder,
        embed_audio,
        vision_soft_tokens_per_image,
        audio_soft_tokens_per_image,
        image_token_id,
        audio_token_id,
        vocab_size_per_layer_input,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.language_model = language_model
        self.vision_encoder = vision_encoder
        self.embed_vision = embed_vision
        self.audio_encoder = audio_encoder
        self.embed_audio = embed_audio
        self.vision_soft_tokens_per_image = vision_soft_tokens_per_image
        self.audio_soft_tokens_per_image = audio_soft_tokens_per_image
        self.image_token_id = image_token_id
        self.audio_token_id = audio_token_id
        self.vocab_size_per_layer_input = vocab_size_per_layer_input
        self.text_hidden_size = language_model.embed_tokens.embedding_dim

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_spec(self, inputs):
        input_ids_spec = inputs["token_ids"]
        batch_size = input_ids_spec.shape[0]
        seq_len = input_ids_spec.shape[1]
        inputs_embeds_spec = keras.KerasTensor(
            shape=(batch_size, seq_len, self.text_hidden_size),
            dtype=input_ids_spec.dtype
            if hasattr(input_ids_spec.dtype, "name")
            else "float32",
        )
        num_layers = self.language_model.num_hidden_layers
        per_layer_hidden_size = self.language_model.hidden_size_per_layer_input
        per_layer_inputs_spec = keras.KerasTensor(
            shape=(batch_size, seq_len, num_layers, per_layer_hidden_size),
            dtype=input_ids_spec.dtype
            if hasattr(input_ids_spec.dtype, "name")
            else "float32",
        )
        return inputs_embeds_spec, per_layer_inputs_spec

    def call(self, inputs):
        input_ids = inputs["token_ids"]
        pixel_values = inputs.get("pixel_values")
        input_features = inputs.get("input_features")
        input_features_mask = inputs.get("input_features_mask")
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        per_layer_inputs_mask = keras.ops.logical_and(
            input_ids >= 0, input_ids < self.vocab_size_per_layer_input
        )
        per_layer_inputs_tokens = keras.ops.where(
            per_layer_inputs_mask, input_ids, keras.ops.zeros_like(input_ids)
        )
        per_layer_inputs = self.language_model.get_per_layer_inputs(
            per_layer_inputs_tokens
        )
        if self.vision_encoder and self.embed_vision:
            if self.embed_audio:
                vision_upper_bound = self.embed_audio.vocab_offset
            else:
                vision_upper_bound = (
                    self.embed_vision.vocab_offset
                    + self.embed_vision.vocab_size
                )
            vision_mask = keras.ops.logical_and(
                input_ids >= self.embed_vision.vocab_offset,
                input_ids < vision_upper_bound,
            )
            dummy_vision_token_id = (
                self.embed_vision.vocab_offset
                + self.embed_vision.embedding.input_dim
                - 1
            )
            vision_input_ids = keras.ops.where(
                vision_mask, input_ids, dummy_vision_token_id
            )
            vision_embeds_from_vocab = self.embed_vision(vision_input_ids)
            expanded_vision_mask = keras.ops.expand_dims(vision_mask, axis=-1)
            inputs_embeds = keras.ops.where(
                expanded_vision_mask,
                vision_embeds_from_vocab,
                inputs_embeds,
            )
        if self.audio_encoder and self.embed_audio:
            audio_mask = input_ids >= self.embed_audio.vocab_offset
            dummy_audio_token_id = (
                self.embed_audio.vocab_offset
                + self.embed_audio.embedding.input_dim
                - 1
            )
            audio_input_ids = keras.ops.where(
                audio_mask, input_ids, dummy_audio_token_id
            )
            audio_embeds_from_vocab = self.embed_audio(audio_input_ids)
            expanded_audio_mask = keras.ops.expand_dims(audio_mask, axis=-1)
            inputs_embeds = keras.ops.where(
                expanded_audio_mask, audio_embeds_from_vocab, inputs_embeds
            )

        if pixel_values is not None and self.vision_encoder:
            reshape_target = (-1,) + tuple(self.vision_encoder.image_shape)
            pixel_values = keras.ops.reshape(pixel_values, reshape_target)
            vision_features = self.vision_encoder(pixel_values)
            if self.vision_encoder.data_format == "channels_first":
                vision_features = keras.ops.transpose(
                    vision_features, (0, 2, 3, 1)
                )
            shape = keras.ops.shape(vision_features)
            vision_features = keras.ops.reshape(
                vision_features, (shape[0], shape[1] * shape[2], shape[3])
            )
            vision_features *= keras.ops.sqrt(
                keras.ops.cast(
                    self.vision_encoder.num_features, dtype=inputs_embeds.dtype
                )
            )
            vision_embeds = self.embed_vision(vision_features)
            image_token_mask = keras.ops.equal(input_ids, self.image_token_id)

            def scatter_vision_features():
                batch_size, seq_len, hidden_size = keras.ops.shape(
                    inputs_embeds
                )
                flat_vision_embeds = keras.ops.reshape(
                    vision_embeds, [-1, hidden_size]
                )
                flat_full_mask = keras.ops.reshape(image_token_mask, [-1])
                gather_indices = (
                    keras.ops.cumsum(keras.ops.cast(flat_full_mask, "int32"))
                    - 1
                )
                gather_indices = keras.ops.where(
                    flat_full_mask, gather_indices, 0
                )
                replacement_values = keras.ops.take(
                    flat_vision_embeds, gather_indices, axis=0
                )
                replacement_tensor = keras.ops.reshape(
                    replacement_values, (batch_size, seq_len, hidden_size)
                )
                expanded_full_mask = keras.ops.expand_dims(
                    image_token_mask, axis=-1
                )
                return keras.ops.where(
                    expanded_full_mask, replacement_tensor, inputs_embeds
                )

            inputs_embeds = keras.ops.cond(
                keras.ops.any(image_token_mask),
                scatter_vision_features,
                lambda: inputs_embeds,
            )

        if (
            input_features is not None
            and input_features_mask is not None
            and self.audio_encoder
        ):
            original_shape = keras.ops.shape(input_features)
            b, n, t, f = (
                original_shape[0],
                original_shape[1],
                original_shape[2],
                original_shape[3],
            )
            input_features = keras.ops.reshape(input_features, (b * n, t, f))
            input_features_mask = keras.ops.reshape(
                input_features_mask, (b * n, t)
            )
            audio_features, _ = self.audio_encoder(
                (input_features, input_features_mask)
            )
            audio_embeds = self.embed_audio(audio_features)
            audio_embeds_shape = keras.ops.shape(audio_embeds)
            t_out, h = audio_embeds_shape[1], audio_embeds_shape[2]
            audio_embeds = keras.ops.reshape(audio_embeds, (b, n, t_out, h))
            shape = keras.ops.shape(audio_embeds)
            audio_batch_size, audio_num_clips, audio_seq_len, hidden_size = (
                shape[0],
                shape[1],
                shape[2],
                shape[3],
            )
            target_len = self.audio_soft_tokens_per_image
            last_audio_token_id = (
                self.embed_audio.vocab_offset
                + self.embed_audio.embedding.input_dim
                - 1
            )
            padding_toks = keras.ops.convert_to_tensor(
                [[last_audio_token_id]], dtype="int64"
            )
            padding_embs = self.embed_audio(padding_toks)
            padding_token = keras.ops.squeeze(padding_embs, axis=[0])
            flat_audio_embeds = keras.ops.reshape(
                audio_embeds, [-1, hidden_size]
            )
            vocab = keras.ops.concatenate(
                [flat_audio_embeds, padding_token], axis=0
            )
            pad_token_index = keras.ops.shape(flat_audio_embeds)[0]
            indices = keras.ops.arange(target_len)
            is_real_token = indices < audio_seq_len
            batch_offsets = (
                keras.ops.arange(audio_batch_size * audio_num_clips)
                * audio_seq_len
            )
            real_indices = keras.ops.expand_dims(
                indices, 0
            ) + keras.ops.expand_dims(batch_offsets, 1)
            final_indices = keras.ops.where(
                keras.ops.expand_dims(is_real_token, 0),
                real_indices,
                pad_token_index,
            )
            audio_embeds = keras.ops.take(vocab, final_indices, axis=0)
            audio_embeds = keras.ops.reshape(
                audio_embeds,
                (audio_batch_size, audio_num_clips * target_len, hidden_size),
            )
            audio_token_mask = keras.ops.equal(input_ids, self.audio_token_id)

            def scatter_audio_features():
                batch_size, seq_len, hidden_size = keras.ops.shape(
                    inputs_embeds
                )
                flat_audio_embeds = keras.ops.reshape(
                    audio_embeds, [-1, hidden_size]
                )
                flat_full_mask = keras.ops.reshape(audio_token_mask, [-1])
                gather_indices = (
                    keras.ops.cumsum(keras.ops.cast(flat_full_mask, "int32"))
                    - 1
                )
                gather_indices = keras.ops.where(
                    flat_full_mask, gather_indices, 0
                )
                replacement_values = keras.ops.take(
                    flat_audio_embeds, gather_indices, axis=0
                )
                replacement_tensor = keras.ops.reshape(
                    replacement_values, (batch_size, seq_len, hidden_size)
                )
                expanded_full_mask = keras.ops.expand_dims(
                    audio_token_mask, axis=-1
                )
                return keras.ops.where(
                    expanded_full_mask, replacement_tensor, inputs_embeds
                )

            inputs_embeds = keras.ops.cond(
                keras.ops.any(audio_token_mask),
                scatter_audio_features,
                lambda: inputs_embeds,
            )
        projected_per_layer_inputs = (
            self.language_model.project_per_layer_inputs(
                inputs_embeds, per_layer_inputs
            )
        )
        return inputs_embeds, projected_per_layer_inputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "language_model": keras.layers.serialize(self.language_model),
                "vision_encoder": keras.layers.serialize(self.vision_encoder),
                "embed_vision": keras.layers.serialize(self.embed_vision),
                "audio_encoder": keras.layers.serialize(self.audio_encoder),
                "embed_audio": keras.layers.serialize(self.embed_audio),
                "vision_soft_tokens_per_image": self.vision_soft_tokens_per_image,  # noqa: E501
                "audio_soft_tokens_per_image": self.audio_soft_tokens_per_image,
                "image_token_id": self.image_token_id,
                "audio_token_id": self.audio_token_id,
                "vocab_size_per_layer_input": self.vocab_size_per_layer_input,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        language_model = keras.layers.deserialize(config.pop("language_model"))
        vision_encoder = keras.layers.deserialize(config.pop("vision_encoder"))
        embed_vision = keras.layers.deserialize(config.pop("embed_vision"))
        audio_encoder = keras.layers.deserialize(config.pop("audio_encoder"))
        embed_audio = keras.layers.deserialize(config.pop("embed_audio"))
        return cls(
            language_model=language_model,
            vision_encoder=vision_encoder,
            embed_vision=embed_vision,
            audio_encoder=audio_encoder,
            embed_audio=embed_audio,
            **config,
        )


@keras_hub_export("keras_hub.models.Gemma3nBackbone")
class Gemma3nBackbone(Backbone):
    """The Gemma3n model backbone.

    This model is a multimodal transformer that can process text, image, and
    audio inputs. It consists of a text decoder and optional vision and audio
    encoders.

    Args:
        text_vocab_size: int. The size of the text vocabulary.
        text_hidden_size: int. The hidden size of the text model.
        num_hidden_layers: int. The number of hidden layers in the text model.
        pad_token_id: int. The ID of the padding token.
        num_attention_heads: int. The number of attention heads in the text
            model.
        num_key_value_heads: int. The number of key-value heads for GQA.
        head_dim: int. The dimension of each attention head.
        intermediate_size: list[int]. A list of intermediate sizes for the MLP
            layers.
        hidden_activation: str. The activation function for the MLP layers.
        layer_types: list[str]. A list of layer types ('full_attention' or
            'sliding_attention').
        sliding_window: int. The sliding window size for sliding window
            attention.
        rope_theta: float. The theta value for RoPE.
        max_position_embeddings: int. The maximum sequence length.
        vocab_size_per_layer_input: int. The vocab size for per-layer inputs.
        hidden_size_per_layer_input: int. The hidden size for per-layer inputs.
        altup_num_inputs: int. The number of inputs for the AltUp mechanism.
        laurel_rank: int. The rank for the Laurel block.
        attention_bias: bool. Whether to use a bias in the attention
            projections.
        attention_dropout: float. The dropout rate for attention weights.
        rope_scaling: float. The scaling factor for RoPE.
        rope_local_base_freq: float. The base frequency for local RoPE.
        activation_sparsity_pattern: list[float]. The sparsity pattern for MLP
            activations.
        altup_coef_clip: float. The coefficient clipping value for AltUp.
        altup_active_idx: int. The active index for AltUp.
        altup_correct_scale: bool. Whether to correct the scale in AltUp.
        num_kv_shared_layers: int. The number of shared KV layers.
        vision_encoder_config: dict. The config for the vision encoder.
        vision_hidden_size: int. The hidden size of the vision embeddings.
        vision_vocab_size: int. The vocabulary size for vision tokens.
        vision_vocab_offset: int. The vocabulary offset for vision tokens.
        vision_soft_tokens_per_image: int. The number of tokens per image.
        image_token_id: int. The special token ID for images.
        audio_encoder_config: dict. The config for the audio encoder.
        audio_hidden_size: int. The hidden size of the audio embeddings.
        audio_vocab_size: int. The vocabulary size for audio tokens.
        audio_vocab_offset: int. The vocabulary offset for audio tokens.
        audio_soft_tokens_per_image: int. The number of tokens per audio clip.
        audio_token_id: int. The special token ID for audio.
        rms_norm_eps: float. The epsilon value for RMS normalization.
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the model's computations and weights. Defaults to `None`.

    Example:
    ```python
    import numpy as np
    from keras_hub.src.models.gemma3n.gemma3n_audio_encoder import (
        Gemma3nAudioEncoder,
    )
    from keras_hub.src.models.gemma3n.gemma3n_backbone import Gemma3nBackbone
    from keras_hub.src.models.mobilenetv5.mobilenetv5_backbone import (
        MobileNetV5Backbone,
    )
    from keras_hub.src.models.mobilenetv5.mobilenetv5_builder import (
        convert_arch_def_to_stackwise,
    )

    # Vision encoder config.
    vision_arch_def = [["er_r1_k3_s1_e1_c16"]]
    stackwise_params = convert_arch_def_to_stackwise(vision_arch_def)
    vision_encoder = MobileNetV5Backbone(
        **stackwise_params,
        num_features=4,
        image_shape=(224, 224, 3),
        use_msfa=False,
    )

    # Audio encoder config.
    audio_encoder = Gemma3nAudioEncoder(
        hidden_size=8,
        input_feat_size=32,
        sscp_conv_channel_size=[4, 8],
        sscp_conv_kernel_size=[(3, 3), (3, 3)],
        sscp_conv_stride_size=[(2, 2), (2, 2)],
        sscp_conv_group_norm_eps=1e-5,
        conf_num_hidden_layers=1,
        rms_norm_eps=1e-6,
        gradient_clipping=1.0,
        conf_residual_weight=0.5,
        conf_num_attention_heads=1,
        conf_attention_chunk_size=4,
        conf_attention_context_right=5,
        conf_attention_context_left=5,
        conf_attention_logit_cap=50.0,
        conf_conv_kernel_size=5,
        conf_reduction_factor=1,
    )

    # Backbone config.
    backbone = Gemma3nBackbone(
        text_vocab_size=50,
        text_hidden_size=8,
        num_hidden_layers=1,
        pad_token_id=0,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=[16],
        hidden_activation="gelu_approximate",
        layer_types=["full_attention"],
        sliding_window=4,
        rope_theta=10000.0,
        max_position_embeddings=16,
        vocab_size_per_layer_input=50,
        hidden_size_per_layer_input=2,
        altup_num_inputs=2,
        laurel_rank=1,
        vision_encoder_config=vision_encoder.get_config(),
        vision_hidden_size=16,
        audio_encoder_config=audio_encoder.get_config(),
        audio_hidden_size=8,
    )

    # Create dummy inputs.
    input_data = {
        "token_ids": np.random.randint(0, 50, size=(1, 16), dtype="int32"),
        "attention_mask": np.ones((1, 1, 16, 16), dtype=bool),
        "pixel_values": np.random.rand(1, 1, 224, 224, 3).astype("float32"),
        "input_features": np.random.rand(1, 16, 32).astype("float32"),
        "input_features_mask": np.zeros((1, 16), dtype=bool),
    }

    # Forward pass.
    outputs = backbone(input_data)
    ```
    """

    def __init__(
        self,
        text_vocab_size,
        text_hidden_size,
        num_hidden_layers,
        pad_token_id,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        intermediate_size,
        hidden_activation,
        layer_types,
        sliding_window,
        rope_theta,
        max_position_embeddings,
        vocab_size_per_layer_input,
        hidden_size_per_layer_input,
        altup_num_inputs,
        laurel_rank,
        attention_bias=False,
        attention_dropout=0.0,
        rope_scaling=None,
        rope_local_base_freq=10000.0,
        activation_sparsity_pattern=None,
        altup_coef_clip=None,
        altup_active_idx=0,
        altup_correct_scale=True,
        num_kv_shared_layers=0,
        final_logit_soft_cap=None,
        vision_encoder_config=None,
        vision_hidden_size=2048,
        vision_vocab_size=128,
        vision_vocab_offset=100,
        vision_soft_tokens_per_image=256,
        image_token_id=98,
        audio_encoder_config=None,
        audio_hidden_size=32,
        audio_vocab_size=128,
        audio_vocab_offset=228,
        audio_soft_tokens_per_image=188,
        audio_token_id=99,
        rms_norm_eps=1e-6,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.vision_encoder = None
        if vision_encoder_config:
            local_vision_encoder_config = vision_encoder_config.copy()
            local_vision_encoder_config["dtype"] = dtype
            self.vision_encoder = MobileNetV5Backbone.from_config(
                local_vision_encoder_config
            )
            if not self.vision_encoder.built:
                input_shape = (None,) + tuple(self.vision_encoder.image_shape)
                self.vision_encoder.build(input_shape)
        self.audio_encoder = None
        if audio_encoder_config:
            audio_encoder_sig = inspect.signature(Gemma3nAudioEncoder.__init__)
            audio_encoder_args = {
                p.name for p in audio_encoder_sig.parameters.values()
            }
            keras_layer_sig = inspect.signature(keras.layers.Layer.__init__)
            keras_layer_args = {
                p.name for p in keras_layer_sig.parameters.values()
            }
            valid_args = audio_encoder_args.union(keras_layer_args)
            filtered_kwargs = {
                key: value
                for key, value in audio_encoder_config.items()
                if key in valid_args
            }
            filtered_kwargs.pop("dtype", None)
            self.audio_encoder = Gemma3nAudioEncoder(
                dtype=dtype, **filtered_kwargs
            )
            if not self.audio_encoder.built:
                mel_shape = (
                    None,
                    None,
                    self.audio_encoder.input_feat_size,
                )
                mask_shape = (None, None)
                self.audio_encoder.build((mel_shape, mask_shape))
        self.language_model = Gemma3nTextModel(
            pad_token_id=pad_token_id,
            vocab_size=text_vocab_size,
            hidden_size=text_hidden_size,
            num_hidden_layers=num_hidden_layers,
            rms_norm_eps=rms_norm_eps,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            layer_types=layer_types,
            sliding_window=sliding_window,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rope_local_base_freq=rope_local_base_freq,
            max_position_embeddings=max_position_embeddings,
            intermediate_size=intermediate_size,
            hidden_activation=hidden_activation,
            activation_sparsity_pattern=activation_sparsity_pattern,
            altup_num_inputs=altup_num_inputs,
            altup_coef_clip=altup_coef_clip,
            altup_active_idx=altup_active_idx,
            altup_correct_scale=altup_correct_scale,
            laurel_rank=laurel_rank,
            hidden_size_per_layer_input=hidden_size_per_layer_input,
            vocab_size_per_layer_input=vocab_size_per_layer_input,
            num_kv_shared_layers=num_kv_shared_layers,
            final_logit_soft_cap=final_logit_soft_cap,
            dtype=dtype,
            name="text_model",
        )
        self.embed_vision = None
        if self.vision_encoder:
            self.embed_vision = Gemma3nMultimodalEmbedder(
                multimodal_hidden_size=vision_hidden_size,
                text_hidden_size=text_hidden_size,
                rms_norm_eps=rms_norm_eps,
                vocab_offset=vision_vocab_offset,
                vocab_size=vision_vocab_size,
                dtype=dtype,
                name="vision_embedder",
            )
            if not self.embed_vision.built:
                self.embed_vision.build((None, None))
        self.embed_audio = None
        if self.audio_encoder:
            self.embed_audio = Gemma3nMultimodalEmbedder(
                multimodal_hidden_size=audio_hidden_size,
                text_hidden_size=text_hidden_size,
                rms_norm_eps=rms_norm_eps,
                vocab_offset=audio_vocab_offset,
                vocab_size=audio_vocab_size,
                dtype=dtype,
                name="audio_embedder",
            )
            if not self.embed_audio.built:
                self.embed_audio.build((None, None))
        self.embedding_processor = Gemma3nMultimodalEmbeddingProcessor(
            language_model=self.language_model,
            vision_encoder=self.vision_encoder,
            embed_vision=self.embed_vision,
            audio_encoder=self.audio_encoder,
            embed_audio=self.embed_audio,
            vision_soft_tokens_per_image=vision_soft_tokens_per_image,
            audio_soft_tokens_per_image=audio_soft_tokens_per_image,
            image_token_id=image_token_id,
            audio_token_id=audio_token_id,
            vocab_size_per_layer_input=vocab_size_per_layer_input,
            dtype=dtype,
            name="multimodal_embedding_processor",
        )

        # === Functional Model ===
        # === Model Inputs ===
        token_ids_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="bool", name="padding_mask"
        )
        processor_inputs = {
            "token_ids": token_ids_input,
        }
        model_inputs_list = [token_ids_input, padding_mask_input]
        model_inputs_dict = {
            "token_ids": token_ids_input,
            "padding_mask": padding_mask_input,
        }

        # === Modality Feature Extraction and Interleaving ===
        if self.vision_encoder:
            input_shape = (None,) + tuple(self.vision_encoder.image_shape)
            images_input = keras.Input(
                shape=input_shape,
                dtype="float32",
                name="images",
            )
            processor_inputs["pixel_values"] = images_input
            model_inputs_list.append(images_input)
            model_inputs_dict["images"] = images_input
        if self.audio_encoder:
            input_features_input = keras.Input(
                shape=(None, None, self.audio_encoder.input_feat_size),
                dtype="float32",
                name="input_features",
            )
            input_features_mask_input = keras.Input(
                shape=(None, None), dtype="bool", name="input_features_mask"
            )
            processor_inputs["input_features"] = input_features_input
            processor_inputs["input_features_mask"] = input_features_mask_input
            model_inputs_list.append(input_features_input)
            model_inputs_list.append(input_features_mask_input)
            model_inputs_dict["input_features"] = input_features_input
            model_inputs_dict["input_features_mask"] = input_features_mask_input
        final_embeds, per_layer_inputs = self.embedding_processor(
            processor_inputs
        )

        # === Decoder layers ===
        # The Gemma3nTextModel encapsulates the decoder loop and final norm.
        # It requires `input_ids` for its internal per-layer logic.
        sequence_output = self.language_model(
            token_ids_input,
            padding_mask_input,
            final_embeds,
            per_layer_inputs,
        )
        super().__init__(
            inputs=model_inputs_list,
            outputs=sequence_output,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self._model_inputs_dict = model_inputs_dict
        self.text_vocab_size = text_vocab_size
        self.text_hidden_size = text_hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.pad_token_id = pad_token_id
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.hidden_activation = hidden_activation
        self.layer_types = layer_types
        self.sliding_window = sliding_window
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size_per_layer_input = vocab_size_per_layer_input
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.altup_num_inputs = altup_num_inputs
        self.laurel_rank = laurel_rank
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling
        self.rope_local_base_freq = rope_local_base_freq
        self.activation_sparsity_pattern = activation_sparsity_pattern
        self.altup_coef_clip = altup_coef_clip
        self.altup_active_idx = altup_active_idx
        self.altup_correct_scale = altup_correct_scale
        self.num_kv_shared_layers = num_kv_shared_layers
        self.final_logit_soft_cap = final_logit_soft_cap
        self.vision_encoder_config = vision_encoder_config
        self.vision_hidden_size = vision_hidden_size
        self.vision_vocab_size = vision_vocab_size
        self.vision_vocab_offset = vision_vocab_offset
        self.vision_soft_tokens_per_image = vision_soft_tokens_per_image
        self.image_token_id = image_token_id
        self.audio_encoder_config = audio_encoder_config
        self.audio_hidden_size = audio_hidden_size
        self.audio_vocab_size = audio_vocab_size
        self.audio_vocab_offset = audio_vocab_offset
        self.audio_soft_tokens_per_image = audio_soft_tokens_per_image
        self.audio_token_id = audio_token_id
        self.rms_norm_eps = rms_norm_eps

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "text_vocab_size": self.text_vocab_size,
                "text_hidden_size": self.text_hidden_size,
                "num_hidden_layers": self.num_hidden_layers,
                "pad_token_id": self.pad_token_id,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
                "intermediate_size": self.intermediate_size,
                "hidden_activation": self.hidden_activation,
                "layer_types": self.layer_types,
                "sliding_window": self.sliding_window,
                "rope_theta": self.rope_theta,
                "max_position_embeddings": self.max_position_embeddings,
                "vocab_size_per_layer_input": self.vocab_size_per_layer_input,
                "hidden_size_per_layer_input": self.hidden_size_per_layer_input,
                "altup_num_inputs": self.altup_num_inputs,
                "laurel_rank": self.laurel_rank,
                "attention_bias": self.attention_bias,
                "attention_dropout": self.attention_dropout,
                "rope_scaling": self.rope_scaling,
                "rope_local_base_freq": self.rope_local_base_freq,
                "activation_sparsity_pattern": self.activation_sparsity_pattern,
                "altup_coef_clip": self.altup_coef_clip,
                "altup_active_idx": self.altup_active_idx,
                "altup_correct_scale": self.altup_correct_scale,
                "num_kv_shared_layers": self.num_kv_shared_layers,
                "final_logit_soft_cap": self.final_logit_soft_cap,
                "vision_encoder_config": self.vision_encoder_config,
                "vision_hidden_size": self.vision_hidden_size,
                "vision_vocab_size": self.vision_vocab_size,
                "vision_vocab_offset": self.vision_vocab_offset,
                "vision_soft_tokens_per_image": self.vision_soft_tokens_per_image,  # noqa: E501
                "image_token_id": self.image_token_id,
                "audio_encoder_config": self.audio_encoder_config,
                "audio_hidden_size": self.audio_hidden_size,
                "audio_vocab_size": self.audio_vocab_size,
                "audio_vocab_offset": self.audio_vocab_offset,
                "audio_soft_tokens_per_image": self.audio_soft_tokens_per_image,
                "audio_token_id": self.audio_token_id,
                "rms_norm_eps": self.rms_norm_eps,
            }
        )
        return config
