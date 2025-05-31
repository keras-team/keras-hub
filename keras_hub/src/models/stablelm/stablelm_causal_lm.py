import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.stablelm.stablelm_backbone import StableLMBackbone
from keras_hub.src.models.stablelm.stablelm_causal_lm_preprocessor import (
    StableLMCausalLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.StableLMCausalLM")
class StableLMCausalLM(CausalLM):
    """An end-to-end StableLM model for causal language modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens. This task setup can be used to train the model unsupervised on
    plain text input, or to autoregressively generate plain text similar to
    the data used for training. This task can be used for pre-training or
    fine-tuning a StableLM model, simply by calling `fit()`.

    Args:
        backbone: A `keras_hub.models.StableLMBackbone` instance.
        preprocessor: A `keras_hub.models.StableLMCausalLMPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.
    """


    backbone_cls = StableLMBackbone
    preprocessor_cls = StableLMCausalLMPreprocessor

    def __init__(self, backbone, preprocessor=None, **kwargs):
        """Initialize the StableLMCausalLM model."""
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor
        self.lm_head = keras.layers.Dense(
            self.backbone.vocabulary_size,
            use_bias=False,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
            name="lm_head",
        )

        # === Functional Model ===
        inputs = backbone.input
        hidden_states = backbone(inputs)
        outputs = backbone.token_embedding(hidden_states, reverse=True)
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

    def call_with_cache(self, token_ids, cache, cache_update_index):
        """Forward pass with caching for autoregressive inference.

        This method enables efficient generation by caching previous key/value
        tensors in the attention layers, avoiding recomputation of seen tokens.

        Args:
            token_ids: A dense int Tensor with shape `(batch_size, max_length)`.
            cache: A dense float Tensor representing the cache of key and value.
            cache_update_index: int or int Tensor, the index of current inputs
                in the sequence.

        Returns:
            A tuple (logits, hidden_states, cache) where:
            - `logits`: Language model logits for the input token_ids.
            - `hidden_states`: Final hidden representation of the input tokens.
            - `cache`: Updated decoding cache.
        """
        x = self.backbone.token_embedding(token_ids)
        updated_cache = []
        for i, layer in enumerate(self.backbone.transformer_layers):
            current_cache = cache[:, i, ...]
            x, next_cache = layer(
                x,
                self_attention_cache=current_cache,
                self_attention_cache_update_index=cache_update_index,
            )
            updated_cache.append(next_cache)
        cache = ops.stack(updated_cache, axis=1)
        hidden_states = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(hidden_states, reverse=True)
        return logits, hidden_states, cache

    def _build_cache(self, token_ids):
        batch_size = ops.shape(token_ids)[0]
        max_length = ops.shape(token_ids)[1]
        num_layers = self.backbone.num_layers
        num_key_value_heads = self.backbone.num_key_value_heads
        head_dim = self.backbone.hidden_dim // self.backbone.num_query_heads
        shape = [
            batch_size,
            num_layers,
            2,
            max_length,
            num_key_value_heads,
            head_dim,
        ]
        cache = ops.zeros(shape, dtype=self.compute_dtype)
        _, hidden_states, cache = self.call_with_cache(token_ids, cache, 0)
        return hidden_states, cache

    def generate_step(self, inputs, stop_token_ids=None):
        """A compilable generation function for a single batch of inputs.

        Args:
            inputs: A dictionary with keys `"token_ids"` and `"padding_mask"`.
            stop_token_ids: Tuple of token IDs to stop generation on. If all
                sequences produce a stop token, generation halts.

        Returns:
            A dictionary with updated `"token_ids"` and `"padding_mask"`.
        """
        token_ids, padding_mask = inputs["token_ids"], inputs["padding_mask"]
        hidden_states, cache = self._build_cache(token_ids)
        row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
        index = ops.min(row_lengths)

        def next(prompt, cache, index):
            cache_update_index = index - 1
            batch_size = ops.shape(prompt)[0]
            prompt = ops.slice(prompt, [0, cache_update_index], [batch_size, 1])
            logits, hidden_states, cache = self.call_with_cache(
                prompt, cache, cache_update_index
            )
            return (
                ops.squeeze(logits, axis=1),
                ops.squeeze(hidden_states, axis=1),
                cache,
            )

        token_ids = self.sampler(
            next=next,
            prompt=token_ids,
            cache=cache,
            index=index,
            mask=padding_mask,
            stop_token_ids=stop_token_ids,
            hidden_states=hidden_states,
            model=self,
        )

        if stop_token_ids is not None:
            end_locations = any_equal(
                token_ids, stop_token_ids, ops.logical_not(padding_mask)
            )
            end_locations = ops.cast(end_locations, "int32")
            cumsum = ops.cast(ops.cumsum(end_locations, axis=-1), "int32")
            overflow = cumsum - end_locations
            padding_mask = ops.logical_not(ops.cast(overflow, "bool"))
        else:
            padding_mask = ops.ones_like(token_ids, dtype="bool")

        return {"token_ids": token_ids, "padding_mask": padding_mask}
    
    def score(
        self,
        token_ids,
        padding_mask=None,
        scoring_mode="logits",
        layer_intercept_fn=None,
        target_ids=None,
    ):
        """Score a generation represented by the provided token ids.

        This method computes scores for a sequence of token IDs, returning
        either logits or per-token loss, depending on the `scoring_mode`.
        It’s useful for evaluating model performance or conducting
        interpretability research.

        Args:
            token_ids: A <int>[batch_size, num_tokens] tensor containing
                tokens to score.
            padding_mask: A <bool>[batch_size, num_tokens] tensor indicating
                valid tokens.
            scoring_mode: Either "logits" or "loss", specifying the type of
                scores to return.
            layer_intercept_fn: Optional function to modify activations at
                each layer.
            target_ids: A <int>[batch_size, num_tokens] tensor of true token
                IDs, required for "loss" mode.

        Returns:
            - In "logits" mode: <float>[batch_size, num_tokens, vocab_size]
                tensor of logits.
            - In "loss" mode: <float>[batch_size, num_tokens] tensor of
                per-token loss.
        """
        if scoring_mode not in ("logits", "loss"):
            raise ValueError(
                "Unsupported scoring_mode. Must be one of 'logits' or 'loss'."
            )

        if scoring_mode == "loss" and target_ids is None:
            raise ValueError(
                "Cannot compute loss without targets. Please provide target "
                "token ids via the target_ids parameter."
            )

        batch_shape = ops.shape(token_ids)[:2]
        if padding_mask is None:
            padding_mask = ops.ones(shape=batch_shape, dtype="bool")

        if layer_intercept_fn is None:
            def default_layer_intercept_fn(x, unused_i):
                return x
            layer_intercept_fn = default_layer_intercept_fn

        # Forward pass through the model
        x = self.backbone.token_embedding(token_ids)
        x = layer_intercept_fn(x, -1)  # Apply to embeddings (index -1)

        for i, transformer_layer in enumerate(self.backbone.transformer_layers):
            x = transformer_layer(x, decoder_padding_mask=padding_mask)
            x = layer_intercept_fn(x, i)  # Apply to each transformer layer

        x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)

        if scoring_mode == "logits":
            return logits

        # Compute per-token loss if scoring_mode is "loss"
        per_token_loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        per_token_loss = per_token_loss_fn(target_ids, logits)
        return per_token_loss