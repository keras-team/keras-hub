import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.layers.modeling.rms_normalization import RMSNormalization
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.gemma3n.gemma3n_audio_encoder import (
    Gemma3nAudioEncoder,
)
from keras_hub.src.models.gemma3n.gemma3n_decoder_block import (
    Gemma3nTransformerDecoder,
)
from keras_hub.src.models.gemma3n.gemma3n_interleave_embedding import (
    Gemma3nInterleaveEmbeddings,
)
from keras_hub.src.models.mobilenetv5.mobilenetv5_backbone import (
    MobileNetV5Backbone,
)


class Gemma3nMultimodalEmbedder(keras.layers.Layer):
    """
    Embeds token ids (hard) or feature vectors (soft) for multimodal
    content into the language model's vector space.

    This layer implements two distinct processing paths:
    1.  If `input_ids` are provided, it performs an embedding lookup for
        special multimodal tokens (e.g., `<image_1>`). It normalizes these
        embeddings and projects them.
    2.  If `inputs_embeds` are provided, it assumes these are "soft tokens"
        (feature vectors from an encoder like a ViT), normalizes them,
        and projects them.

    You must provide exactly one of `input_ids` or `inputs_embeds`.

    Args:
        multimodal_hidden_size (int): The hidden size of the modality's native
            embedding space (e.g., the ViT's output dimension).
        text_hidden_size (int): The hidden size of the main language model.
        vocab_offset (int): The starting integer ID for this modality's
            special tokens in the main vocabulary.
        vocab_size (int): The number of special tokens reserved for this
            modality (e.g., 256 image tokens).
        rms_norm_eps (float): The epsilon value for RMSNormalization layers.
    """

    def __init__(
        self,
        multimodal_hidden_size,
        text_hidden_size,
        vocab_offset,
        vocab_size,
        rms_norm_eps=1e-6,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.multimodal_hidden_size = multimodal_hidden_size
        self.text_hidden_size = text_hidden_size
        self.vocab_offset = vocab_offset
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps

        # For hard tokens (input_ids)
        self.embedding = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.multimodal_hidden_size,
            name="multimodal_embedding",
        )
        self.hard_embedding_norm = RMSNormalization(
            epsilon=self.rms_norm_eps, name="hard_embedding_norm"
        )

        # For soft tokens (inputs_embeds)
        self.soft_embedding_norm = RMSNormalization(
            epsilon=self.rms_norm_eps, name="soft_embedding_norm"
        )

        # Common projection and final normalization
        self.embedding_projection = keras.layers.Dense(
            units=self.text_hidden_size,
            use_bias=False,
            name="embedding_projection",
        )
        self.embedding_post_projection_norm = RMSNormalization(
            epsilon=self.rms_norm_eps,
            # In PyTorch, this was `with_scale=False`, which in Keras's
            # RMSNormalization corresponds to `center=False, scale=False`.
            # However, typically RMSNorm only has a scale (`gamma`) parameter.
            # Assuming it means no scaling parameter.
            scale=False,
            name="embedding_post_projection_norm",
        )

    def call(self, input_ids=None, inputs_embeds=None):
        # Enforce that exactly one input is provided.
        if (input_ids is None and inputs_embeds is None) or (
            input_ids is not None and inputs_embeds is not None
        ):
            raise ValueError(
                "You must specify exactly one of 'input_ids' or 'inputs_embeds'."
            )

        if inputs_embeds is not None:
            # Path 1: Soft token embeddings from an encoder
            emb_norm = self.soft_embedding_norm(inputs_embeds)
        else:
            # Path 2: Hard token ids from the vocabulary
            # Map global token IDs to the local 0-indexed embedding table
            local_ids = input_ids - self.vocab_offset
            hard_emb = self.embedding(local_ids)
            emb_norm = self.hard_embedding_norm(hard_emb)

        # Common projection path for both soft and hard embeddings
        emb_norm_proj = self.embedding_projection(emb_norm)
        final_embeds = self.embedding_post_projection_norm(emb_norm_proj)

        return final_embeds

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "multimodal_hidden_size": self.multimodal_hidden_size,
                "text_hidden_size": self.text_hidden_size,
                "vocab_offset": self.vocab_offset,
                "vocab_size": self.vocab_size,
                "rms_norm_eps": self.rms_norm_eps,
            }
        )
        return config


@keras_hub_export("keras_hub.models.Gemma3nBackbone")
class Gemma3nBackbone(Backbone):
    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_query_heads,
        num_key_value_heads,
        hidden_dim,
        intermediate_dim,
        head_dim,
        # Per-layer input parameters
        vocab_size_per_layer_input,
        # Architectural details
        layer_norm_epsilon=1e-6,
        dropout=0,
        query_head_dim_normalize=True,
        use_query_key_norm=True,
        use_post_ffw_norm=False,
        use_post_attention_norm=False,
        attention_logit_soft_cap=None,
        final_logit_soft_cap=None,
        use_sliding_window_attention=False,
        sliding_window_size=4096,
        local_rope_scaling_factor=1.0,
        global_rope_scaling_factor=1.0,
        dtype=None,
        # Vision Encoder (MobileNetV5) Args
        vision_block_args=None,
        vision_stem_size=64,
        vision_num_features=2048,
        vision_msfa_indices=(-3, -2, -1),
        vision_msfa_output_resolution=16,
        vision_act_layer="gelu",
        vision_layer_scale_init_value=1e-5,
        # Audio Encoder (Conformer) Args
        num_conformer_layers=12,
        audio_hidden_size=1536,
        audio_num_attention_heads=12,
        audio_attention_chunk_size=128,
        audio_attention_context_left=128,
        audio_attention_context_right=128,
        audio_attention_logit_cap=10.0,
        audio_gradient_clipping=1.0,
        audio_residual_weight=0.5,
        audio_conv_kernel_size=5,
        # Multimodal Embedder Args
        image_size=(256, 256),
        num_audio_tokens=188,
        vision_vocab_offset=None,
        vision_vocab_size=None,
        audio_vocab_offset=None,
        audio_vocab_size=None,
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
        self.per_layer_embeddings = keras.layers.Embedding(
            input_dim=vocab_size_per_layer_input,
            output_dim=hidden_dim,
            name="per_layer_embeddings",
            dtype=dtype,
        )

        self.vision_encoder = MobileNetV5Backbone(
            block_args=vision_block_args,
            stem_size=vision_stem_size,
            num_features=vision_num_features,
            msfa_indices=vision_msfa_indices,
            msfa_output_resolution=vision_msfa_output_resolution,
            act_layer=vision_act_layer,
            layer_scale_init_value=vision_layer_scale_init_value,
            layer_norm_epsilon=layer_norm_epsilon,
            image_shape=(image_size[0], image_size[1], 3),
            name="vision_encoder",
            dtype=dtype,
        )

        self.audio_encoder = Gemma3nAudioEncoder(
            num_conformer_layers=num_conformer_layers,
            hidden_size=audio_hidden_size,
            num_attention_heads=audio_num_attention_heads,
            attention_chunk_size=audio_attention_chunk_size,
            attention_context_left=audio_attention_context_left,
            attention_context_right=audio_attention_context_right,
            attention_logit_cap=audio_attention_logit_cap,
            gradient_clipping=audio_gradient_clipping,
            residual_weight=audio_residual_weight,
            conv_kernel_size=audio_conv_kernel_size,
            rms_norm_eps=layer_norm_epsilon,
            name="audio_encoder",
            dtype=dtype,
        )

        self.vision_embedder = None
        if vision_vocab_offset is not None:
            self.vision_embedder = Gemma3nMultimodalEmbedder(
                multimodal_hidden_size=self.vision_encoder.output_dim,
                text_hidden_size=hidden_dim,
                vocab_offset=vision_vocab_offset,
                vocab_size=vision_vocab_size,
                rms_norm_eps=layer_norm_epsilon,
                dtype=dtype,
                name="vision_embedder",
            )

        self.audio_embedder = None
        if audio_vocab_offset is not None:
            self.audio_embedder = Gemma3nMultimodalEmbedder(
                multimodal_hidden_size=self.audio_encoder.output_dim,
                text_hidden_size=hidden_dim,
                vocab_offset=audio_vocab_offset,
                vocab_size=audio_vocab_size,
                rms_norm_eps=layer_norm_epsilon,
                dtype=dtype,
                name="audio_embedder",
            )

        self.interleave_embeddings = Gemma3nInterleaveEmbeddings(
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
            layer = Gemma3nTransformerDecoder(
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

        # == Model inputs ==
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="bool", name="padding_mask"
        )
        per_layer_token_ids_input = keras.Input(
            shape=(None,), dtype="int32", name="per_layer_token_ids"
        )

        # Vision inputs (optional)
        image_input = keras.Input(
            shape=(None, image_size, image_size, 3), name="images"
        )
        vision_indices_input = keras.Input(
            shape=(None,), dtype="int32", name="vision_indices"
        )
        vision_mask_input = keras.Input(
            shape=(None,), dtype="bool", name="vision_mask"
        )

        ### NEW ### Audio inputs (optional)
        audio_input = keras.Input(
            shape=(None, None), name="audio_features"
        )  # (batch, seq, features)
        audio_indices_input = keras.Input(
            shape=(None,), dtype="int32", name="audio_indices"
        )
        audio_mask_input = keras.Input(
            shape=(None,), dtype="bool", name="audio_mask"
        )
        ### END NEW ###

        # == Text embeddings ==
        text_embeddings = self.token_embedding(token_id_input)
        text_embeddings *= ops.cast(ops.sqrt(hidden_dim), text_embeddings.dtype)

        per_layer_inputs = self.per_layer_embeddings(per_layer_token_ids_input)

        x = text_embeddings

        if self.vision_embedder:
            # Create a mask for vision tokens based on their vocab range
            vision_mask = ops.logical_and(
                token_id_input >= self.vision_embedder.vocab_offset,
                token_id_input
                < (
                    self.vision_embedder.vocab_offset
                    + self.vision_embedder.vocab_size
                ),
            )
            # Get embeddings for these specific token IDs
            vision_embeds = self.vision_embedder(input_ids=token_id_input)
            # Splice the vision embeddings into the sequence where the mask is True
            x = ops.where(ops.expand_dims(vision_mask, -1), vision_embeds, x)

        if self.audio_embedder:
            # Create a mask for audio tokens
            audio_mask = token_id_input >= self.audio_embedder.vocab_offset
            # Get embeddings for these token IDs
            audio_embeds = self.audio_embedder(input_ids=token_id_input)
            # Splice the audio embeddings into the sequence
            x = ops.where(ops.expand_dims(audio_mask, -1), audio_embeds, x)

        # == Image Embeddings ==
        final_img_embeddings = None
        if self.vision_encoder and self.vision_embedder:
            # 1. Get raw features from the vision encoder
            raw_img_embeddings = self.vision_encoder(image_input)
            # 2. Project features into text space using the embedder's soft token path
            final_img_embeddings = self.vision_embedder(
                inputs_embeds=raw_img_embeddings
            )

        # == Audio Embeddings ==
        ### NEW ###
        final_audio_embeddings = None
        if self.audio_encoder and self.audio_embedder:
            # 1. Get raw features from the audio encoder
            raw_audio_embeddings = self.audio_encoder(audio_input)

            # 2. Get the padding token embedding (using the last token in vocab)
            padding_token_id = ops.array([[vocabulary_size - 1]], dtype="int32")
            padding_embedding = self.token_embedding(padding_token_id)

            # 3. Pad audio embeddings to the fixed sequence length
            batch_size = ops.shape(raw_audio_embeddings)[0]
            current_audio_len = ops.shape(raw_audio_embeddings)[1]
            num_padding_tokens = ops.maximum(
                0, num_audio_tokens - current_audio_len
            )

            padding_features = ops.tile(
                padding_embedding, [batch_size, num_padding_tokens, 1]
            )
            padded_audio_embeddings = ops.concatenate(
                [raw_audio_embeddings, padding_features], axis=1
            )

            # 4. Project features into text space using the embedder's soft token path
            final_audio_embeddings = self.audio_embedder(
                inputs_embeds=padded_audio_embeddings
            )

        ### END NEW ###

        # == Interleaving text, images, and audio ==
        x = self.interleave_embeddings(
            text_embeddings,
            vision_indices=vision_indices_input,
            image_embeddings=final_img_embeddings,
            audio_indices=audio_indices_input,
            audio_embeddings=final_audio_embeddings,
        )

        # == Decoder layers ==
        for transformer_layer in self.transformer_layers:
            # The vision/audio masks might be combined into a single attention mask
            # in the preprocessor, or handled inside the decoder layer.
            x = transformer_layer(
                x,
                per_layer_inputs=per_layer_inputs,
                padding_mask=padding_mask_input,
                # vision_mask=vision_mask_input, # This would need to be updated
            )
        sequence_output = self.layer_norm(x)

        # Define all possible inputs for the model
        inputs = {
            "token_ids": token_id_input,
            "padding_mask": padding_mask_input,
            "per_layer_token_ids": per_layer_token_ids_input,
        }
        if self.vision_encoder:
            inputs.update(
                {
                    "images": image_input,
                    "vision_indices": vision_indices_input,
                }
            )
        if self.audio_encoder:
            inputs.update(
                {
                    "audio_features": audio_input,
                    "audio_indices": audio_indices_input,
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

        self.vocab_size_per_layer_input = vocab_size_per_layer_input
        self.num_vision_tokens_per_image = (
            self.vision_encoder.num_vision_tokens_per_image
        )
        self.vision_vocab_offset = vision_vocab_offset
        self.vision_vocab_size = vision_vocab_size
        self.audio_vocab_offset = audio_vocab_offset
        self.audio_vocab_size = audio_vocab_size
        self.num_audio_tokens = num_audio_tokens

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "vocab_size_per_layer_input": self.vocab_size_per_layer_input,
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
                "audio_encoder": None
                if self.audio_encoder is None
                else keras.layers.serialize(self.audio_encoder),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        # Deserialize nested layers before calling the constructor
        config["vision_encoder"] = keras.layers.deserialize(
            config.pop("vision_encoder", None)
        )
        config["audio_encoder"] = keras.layers.deserialize(
            config.pop("audio_encoder", None)
        )
        return cls(**config)
