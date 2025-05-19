import keras
from keras import ops
import keras_hub
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.models.backbone import Backbone


class SmolVLMBackbone(Backbone):
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
        vision_encoder=None,
        layer_norm_epsilon=1e-6,
        dropout=0,
        dtype=None,
        **kwargs,
    ):
        # https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct
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

        img_embeddings = self.vision_encoder(image_input)

        # == Interleaving text and images ==
        # Place image embeddings in the right position in
        # `text_embeddings`.
        x = self.interleave_embeddings(
            image_embeddings=img_embeddings,
            text_embeddings=text_embeddings,
            vision_indices=vision_indices_input,
        )

        # == Decoder layers ==
        self.text_model = keras_hub.models.LlamaBackbone()

        inputs = {
            "token_ids": token_id_input,
            "padding_mask": padding_mask_input,
        }

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

        self.num_vision_tokens_per_image = (
            self.vision_encoder.num_vision_tokens_per_image
        )
