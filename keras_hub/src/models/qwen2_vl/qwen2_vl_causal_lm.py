import keras
import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.qwen2_vl.qwen2_vl_backbone import Qwen2VLBackbone
from keras_hub.src.models.qwen2_vl.qwen2_vl_causal_lm_preprocessor import (
    Qwen2VLCausalLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.models.Qwen2VLCausalLM")
class Qwen2VLCausalLM(CausalLM):
    """An end-to-end Qwen2-VL model for causal language modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens. This task setup can be used to train the model unsupervised on
    plain text input, or to autoregressively generate plain text similar to
    the data used for training. This task can be used for pre-training or
    fine-tuning a Qwen2-VL model, simply by calling ``fit()``.

    This model has a ``generate()`` method, which generates text based on
    a prompt. The generation strategy used is controlled by an additional
    ``sampler`` argument on ``compile()``. You can recompile the model with
    different ``keras_hub.samplers`` objects to control the generation. By
    default, ``"greedy"`` sampling will be used.

    This model supports multimodal (image/video + text) inputs when the
    backbone has a ``vision_encoder`` attached and the preprocessor has an
    ``image_converter``.

    Args:
        backbone: A ``keras_hub.models.Qwen2VLBackbone`` instance.
        preprocessor: A ``keras_hub.models.Qwen2VLCausalLMPreprocessor``
            instance or ``None``. If ``None``, this model will not apply
            preprocessing, and inputs should be preprocessed before calling
            the model.
    """

    backbone_cls = Qwen2VLBackbone
    preprocessor_cls = Qwen2VLCausalLMPreprocessor

    def __init__(self, backbone, preprocessor=None, **kwargs):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

        # === Functional Model ===
        inputs = backbone.input
        hidden_states = backbone(inputs)
        outputs = backbone.token_embedding(hidden_states, reverse=True)
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

    def __call__(self, inputs, *args, **kwargs):
        # The functional model requires all vision inputs; inject empty
        # defaults for text-only calls (mirrors the backbone's __call__).
        inputs = self.backbone._inject_empty_vision_inputs(inputs)
        return super().__call__(inputs, *args, **kwargs)

    def _normalize_generate_inputs(
        self,
        inputs,
    ):
        """Handle unbatched image/video inputs for ``generate()``."""
        if tf and isinstance(inputs, tf.data.Dataset):
            return inputs.as_numpy_iterator(), False

        if self.preprocessor is None:
            return [inputs], False

        def normalize(x):
            if isinstance(x, str):
                return [x], True
            if tf and isinstance(x, tf.Tensor) and x.shape.rank == 0:
                return x[tf.newaxis], True
            return x, False

        if isinstance(inputs, dict):
            inputs["prompts"], input_is_scalar = normalize(inputs["prompts"])

            # If prompt is scalar, images can be either a 3D NumPy
            # array/Tensor, or, list of 3D NumPy arrays. Let's uprank.
            if input_is_scalar and "images" in inputs:
                x = inputs["images"]
                if isinstance(x, np.ndarray) and len(x.shape) == 3:
                    inputs["images"] = [x]
                elif tf and isinstance(x, tf.Tensor) and x.shape.rank == 3:
                    inputs["images"] = x[tf.newaxis]
                elif isinstance(x, list):
                    inputs["images"] = [x]

            if input_is_scalar and "videos" in inputs:
                x = inputs["videos"]
                if isinstance(x, np.ndarray) and len(x.shape) == 4:
                    inputs["videos"] = [x]
                elif tf and isinstance(x, tf.Tensor) and x.shape.rank == 4:
                    inputs["videos"] = x[tf.newaxis]
                elif isinstance(x, list):
                    inputs["videos"] = [x]

            if "responses" in inputs:
                inputs["responses"], _ = normalize(inputs["responses"])
        else:
            inputs, input_is_scalar = normalize(inputs)

        return [inputs], input_is_scalar

    def call_with_cache(
        self,
        token_ids,
        cache,
        cache_update_index,
        padding_mask=None,
        img_embeddings=None,
        vision_indices=None,
        position_ids=None,
    ):
        """Forward pass with cache for autoregressive decoding.

        Args:
            token_ids: Dense int tensor ``(batch_size, seq_len)``.
            cache: KV cache ``(batch, num_layers, 2, seq_len, kv_heads,
                head_dim)``.
            cache_update_index: Int or int tensor, current step index.
            padding_mask: Optional padding mask.
            img_embeddings: Optional vision embeddings
                ``(total_vision_tokens, hidden_dim)``. Only used on the
                first (prefill) call to interleave into text embeddings.
            vision_indices: Optional int tensor ``(total_vision_tokens,)``
                with flat scatter indices.
            position_ids: Optional int tensor ``(batch, 3, seq_len)`` of
                M-RoPE positions for the *current* tokens (sliced by the
                caller during decode).

        Returns:
            ``(logits, hidden_states, cache)`` tuple.
        """
        x = self.backbone.token_embedding(token_ids)

        # Interleave vision embeddings on the prefill step.
        if img_embeddings is not None and vision_indices is not None:
            if hasattr(self.backbone, "interleave_embeddings"):
                x = self.backbone.interleave_embeddings(
                    image_embeddings=img_embeddings,
                    text_embeddings=x,
                    vision_indices=vision_indices,
                )

        # Each transformer layer computes its own attention.
        next_cache = []
        for i in range(self.backbone.num_layers):
            current_cache = cache[:, i, ...]
            x, next_cache_i = self.backbone.transformer_layers[i](
                x,
                self_attention_cache=current_cache,
                self_attention_cache_update_index=cache_update_index,
                decoder_padding_mask=padding_mask,
                position_ids=position_ids,
            )
            next_cache.append(next_cache_i)
        cache = ops.stack(next_cache, axis=1)

        hidden_states = x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)
        return logits, hidden_states, cache

    def _build_cache(
        self,
        token_ids,
        padding_mask,
        img_embeddings=None,
        vision_indices=None,
        position_ids=None,
    ):
        """Build an empty cache for use with ``call_with_cache()``."""
        batch_size = ops.shape(token_ids)[0]
        max_length = ops.shape(token_ids)[1]
        num_layers = self.backbone.num_layers
        num_heads = self.backbone.num_key_value_heads
        head_dim = self.backbone.hidden_dim // self.backbone.num_query_heads
        shape = [
            batch_size,
            num_layers,
            2,
            max_length,
            num_heads,
            head_dim,
        ]
        cache = ops.zeros(shape, dtype=self.compute_dtype)
        # Seed the cache with a full forward pass, including vision
        # embeddings on the first call.
        _, hidden_states, cache = self.call_with_cache(
            token_ids,
            cache,
            0,
            padding_mask=padding_mask,
            img_embeddings=img_embeddings,
            vision_indices=vision_indices,
            position_ids=position_ids,
        )
        return hidden_states, cache

    def generate_step(self, inputs, stop_token_ids=None):
        """A compilable generation function for a single batch.

        For multimodal inputs, the preprocessor populates extra keys:
        ``pixel_values``, ``image_grid_thw``, ``vision_indices`` and
        ``position_ids``. Vision inputs are consumed on the first forward
        pass (cache prefill) and ``position_ids`` are sliced per decode
        step — this reproduces HF's ``mrope_position_deltas`` bookkeeping.
        """
        token_ids = inputs["token_ids"]
        padding_mask = inputs["padding_mask"]

        # Check for multimodal inputs.
        pixel_values = inputs.get("pixel_values", None)
        image_grid_thw = inputs.get("image_grid_thw", None)
        vision_indices = inputs.get("vision_indices", None)
        position_ids = inputs.get("position_ids", None)

        # Run vision encoder if present and we have pixel data.
        img_embeddings = None
        if (
            self.backbone.vision_encoder is not None
            and pixel_values is not None
        ):
            img_embeddings = self.backbone.vision_encoder(
                pixel_values, image_grid_thw
            )

        hidden_states, cache = self._build_cache(
            token_ids,
            padding_mask,
            img_embeddings=img_embeddings,
            vision_indices=vision_indices,
            position_ids=position_ids,
        )
        row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
        index = ops.min(row_lengths)

        def next(prompt, cache, index):
            cache_update_index = index - 1
            batch_size = ops.shape(prompt)[0]
            prompt = ops.slice(prompt, [0, cache_update_index], [batch_size, 1])
            # Slice the M-RoPE positions for the token at
            # `cache_update_index`; for text-only inputs (position_ids
            # absent) the decoder falls back to standard RoPE.
            step_position_ids = None
            if position_ids is not None:
                step_position_ids = ops.slice(
                    position_ids,
                    [0, 0, cache_update_index],
                    [batch_size, 3, 1],
                )
            logits, hidden_states, cache = self.call_with_cache(
                prompt,
                cache,
                cache_update_index,
                padding_mask=None,
                position_ids=step_position_ids,
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

        # Compute an output padding mask that truncates after the first
        # end token.
        if stop_token_ids is not None:
            end_locations = any_equal(
                token_ids,
                stop_token_ids,
                ops.logical_not(padding_mask),
            )
            end_locations = ops.cast(end_locations, "int32")
            cumsum = ops.cast(ops.cumsum(end_locations, axis=-1), "int32")
            overflow = cumsum - end_locations
            padding_mask = ops.logical_not(ops.cast(overflow, "bool"))
        else:
            padding_mask = ops.ones_like(token_ids, dtype="bool")
        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }

    def score(
        self,
        token_ids,
        padding_mask=None,
        scoring_mode="logits",
        layer_intercept_fn=None,
        target_ids=None,
    ):
        """Score a generation represented by the provided token ids.

        Accepts either a plain ``token_ids`` tensor (text-only) or a dict
        with ``token_ids``, ``padding_mask``, and optional multimodal keys
        (``pixel_values``, ``image_grid_thw``, ``vision_indices``,
        ``position_ids``).
        """
        if scoring_mode not in ("logits", "loss"):
            raise ValueError(
                "Unsupported scoring_mode. Must be 'logits' or 'loss'."
            )
        if scoring_mode == "loss" and target_ids is None:
            raise ValueError(
                "Cannot compute loss without targets. Please provide "
                "target token ids via the target_ids parameter."
            )

        # Unpack multimodal dict inputs if provided.
        pixel_values = None
        image_grid_thw = None
        vision_indices = None
        position_ids = None
        if isinstance(token_ids, dict):
            padding_mask = token_ids.get("padding_mask", padding_mask)
            pixel_values = token_ids.get("pixel_values", None)
            image_grid_thw = token_ids.get("image_grid_thw", None)
            vision_indices = token_ids.get("vision_indices", None)
            position_ids = token_ids.get("position_ids", None)
            token_ids = token_ids["token_ids"]

        batch_shape = ops.shape(token_ids)[:2]
        assert len(batch_shape) == 2

        if padding_mask is None:
            padding_mask = ops.ones(shape=batch_shape)

        if layer_intercept_fn is None:

            def default_layer_intercept_fn(x, unused_i):
                return x

            layer_intercept_fn = default_layer_intercept_fn

        token_embeddings = self.backbone.token_embedding(token_ids)

        # Interleave vision embeddings if multimodal inputs are present.
        if (
            self.backbone.vision_encoder is not None
            and pixel_values is not None
        ):
            img_embeddings = self.backbone.vision_encoder(
                pixel_values, image_grid_thw
            )
            token_embeddings = self.backbone.interleave_embeddings(
                image_embeddings=img_embeddings,
                text_embeddings=token_embeddings,
                vision_indices=vision_indices,
            )

        x = layer_intercept_fn(token_embeddings, -1)

        for i, transformer_layer in enumerate(self.backbone.transformer_layers):
            x = transformer_layer(
                x,
                decoder_padding_mask=padding_mask,
                position_ids=position_ids,
            )
            x = layer_intercept_fn(x, i)

        x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)

        if scoring_mode == "logits":
            return logits

        per_token_loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        per_token_loss = per_token_loss_fn(target_ids, logits)
        return per_token_loss

    def generate(
        self,
        inputs,
        max_length=None,
        stop_token_ids="auto",
        strip_prompt=False,
    ):
        # If `auto`, add `<|endoftext|>` as a stop token too — base-model
        # generations terminate with it rather than `<|im_end|>`.
        if self.preprocessor is None and stop_token_ids == "auto":
            raise ValueError(
                "A `preprocessor` must be attached to the model if "
                '`stop_token_ids="auto"`. Currently `preprocessor=None`. To '
                "call `generate()` with preprocessing detached, either pass "
                "`stop_token_ids=None` to always generate until `max_length` "
                "or pass a tuple of token ids that should terminate generation "
                "as `stop_token_ids`."
            )
        elif stop_token_ids == "auto":
            stop_token_ids = [
                self.preprocessor.tokenizer.end_token_id,
                self.preprocessor.tokenizer.token_to_id("<|endoftext|>"),
            ]

        return super().generate(
            inputs,
            max_length=max_length,
            stop_token_ids=stop_token_ids,
            strip_prompt=strip_prompt,
        )
