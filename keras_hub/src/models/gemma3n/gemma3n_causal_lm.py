import keras
import numpy as np

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.gemma3n.gemma3n_backbone import Gemma3nBackbone
from keras_hub.src.models.gemma3n.gemma3n_causal_lm_preprocessor import (
    Gemma3nCausalLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.models.Gemma3nCausalLM")
class Gemma3nCausalLM(CausalLM):
    """An end-to-end multimodal Gemma3n model for causal language modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens. This task setup can be used to train the model unsupervised on
    images, audio, and plain text inputs, or to autoregressively generate plain
    text similar to the data used for training. Note that the model is
    image-audio-text in, text out.

    This model has a `generate()` method, which generates text based on a
    prompt. The generation strategy used is controlled by an additional
    `sampler` argument on `compile()`. You can recompile the model with
    different `keras_hub.samplers` objects to control the generation. By
    default, `"greedy"` sampling will be used.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to string inputs during
    `fit()`, `predict()`, `evaluate()` and `generate()`. This is done by default
    when creating the model with `from_preset()`.

    Args:
        preprocessor: A `keras_hub.models.Gemma3nCausalLMPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.
        backbone: A `keras_hub.models.Gemma3nBackbone` instance.

    Examples:
    ```python
    import numpy as np
    from keras_hub.models import Gemma3nCausalLM

    # === Text-only usage ===
    # Load a text-only Gemma3n model from preset.
    causal_lm = Gemma3nCausalLM.from_preset("gemma3n_instruct_1b")

    # Generate text.
    causal_lm.generate("What is the capital of France?", max_length=128)

    # === Vision + Text usage ===
    # Load a vision-text Gemma3n model from preset.
    causal_lm = Gemma3nCausalLM.from_preset("gemma3n_instruct_4b")

    # Generate with image input.
    image = np.ones((768, 768, 3), dtype="float32")
    causal_lm.generate({
        "prompts": "Describe this image: <start_of_image>",
        "images": image
    })

    # === Audio + Text usage ===
    # Load an audio-text Gemma3n model from preset.
    causal_lm = Gemma3nCausalLM.from_preset("gemma3n_instruct_4b_audio")

    # Generate with audio input.
    audio = np.ones((16000,), dtype="float32")
    causal_lm.generate({
        "prompts": "Transcribe this audio: <start_of_audio>",
        "audios": audio
    })

    # === Vision + Audio + Text usage ===
    # Generate with both image and audio.
    causal_lm.generate({
        "prompts": "Image: <start_of_image>, Audio: <start_of_audio>",
        "images": image,
        "audios": audio
    })
    ```
    """

    backbone_cls = Gemma3nBackbone
    preprocessor_cls = Gemma3nCausalLMPreprocessor

    def __init__(
        self,
        preprocessor,
        backbone,
        **kwargs,
    ):
        # === Layers ===
        self.preprocessor = preprocessor
        self.backbone = backbone

        # === Functional Model ===
        inputs = backbone._model_inputs_dict.copy()
        if "images" in inputs:
            if "vision_indices" not in inputs:
                inputs["vision_indices"] = keras.Input(
                    shape=(None,), dtype="int32", name="vision_indices"
                )
            if "vision_mask" not in inputs:
                inputs["vision_mask"] = keras.Input(
                    shape=(None,), dtype="bool", name="vision_mask"
                )
        if "input_features" in inputs:
            if "audio_indices" not in inputs:
                inputs["audio_indices"] = keras.Input(
                    shape=(None,), dtype="int32", name="audio_indices"
                )
            if "audio_mask" not in inputs:
                inputs["audio_mask"] = keras.Input(
                    shape=(None,), dtype="bool", name="audio_mask"
                )
        hidden_state = backbone(inputs)
        outputs = backbone.language_model.token_embedding(
            hidden_state, reverse=True
        )
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

    def compile(
        self,
        optimizer="auto",
        loss="auto",
        *,
        weighted_metrics="auto",
        sampler="greedy",
        **kwargs,
    ):
        super().compile(
            optimizer=optimizer,
            loss=loss,
            weighted_metrics=weighted_metrics,
            sampler=sampler,
            **kwargs,
        )

    def _normalize_generate_inputs(
        self,
        inputs,
    ):
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
            # Handle unbatched image input.
            if "images" in inputs and input_is_scalar:
                x = inputs["images"]
                if isinstance(x, np.ndarray) and len(x.shape) == 3:
                    inputs["images"] = [x]
                elif tf and isinstance(x, tf.Tensor) and x.shape.rank == 3:
                    inputs["images"] = x[tf.newaxis]
                elif isinstance(x, list):
                    inputs["images"] = [x]
            # Handle unbatched audio input.
            if "audios" in inputs and input_is_scalar:
                x = inputs["audios"]
                if isinstance(x, np.ndarray) and len(x.shape) == 1:
                    inputs["audios"] = [x]
                elif tf and isinstance(x, tf.Tensor) and x.shape.rank == 1:
                    inputs["audios"] = x[tf.newaxis]
                elif isinstance(x, list):
                    inputs["audios"] = [x]
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
        pixel_values=None,
        input_features=None,
        input_features_mask=None,
        vision_indices=None,
        audio_indices=None,
        vision_mask=None,
        audio_mask=None,
        padding_mask=None,
        cache_update_mask=None,
    ):
        """Forward pass of `Gemma3nCausalLM` with cache.

        `call_with_cache` adds an additional forward pass for the model for
        autoregressive inference. Unlike calling the model directly, this method
        allows caching previous key/value Tensors in multi-head attention layer,
        and avoids recomputing the outputs of seen tokens.

        Args:
            token_ids: A dense int Tensor with shape `(batch_size, max_length)`.
            cache: A dense float Tensor, the cache of key and value.
            cache_update_index: int, or int Tensor. The index of current inputs
                in the whole sequence.
            pixel_values: A dense float Tensor with shape
                `(batch_size, num_images, height, width, channels)`.
            input_features: A dense float Tensor with shape
                `(batch_size, num_audios, audio_seq_len, feature_size)`.
            input_features_mask: A dense bool Tensor with shape
                `(batch_size, num_audios, audio_seq_len)`.
            vision_indices: A dense int Tensor with shape
                `(batch_size, num_vision_tokens)`.
            audio_indices: A dense int Tensor with shape
                `(batch_size, num_audio_tokens)`.
            vision_mask: A dense bool Tensor with shape
                `(batch_size, max_length)`.
            audio_mask: A dense bool Tensor with shape
                `(batch_size, max_length)`.
            padding_mask: A dense int Tensor with shape
                `(batch_size, max_length)`.
            cache_update_mask: A dense bool Tensor for masking cache updates.

        Returns:
            A (logits, hidden_states, cache) tuple. Where `logits` is the
            language model logits for the input token_ids, `hidden_states` is
            the final hidden representation of the input tokens, and `cache` is
            the decoding cache.
        """
        # Build inputs dict for embedding processor.
        processor_inputs = {"token_ids": token_ids}
        if pixel_values is not None:
            processor_inputs["pixel_values"] = pixel_values
            processor_inputs["vision_indices"] = vision_indices
            processor_inputs["vision_mask"] = vision_mask
        if input_features is not None:
            processor_inputs["input_features"] = input_features
            processor_inputs["input_features_mask"] = input_features_mask
            processor_inputs["audio_indices"] = audio_indices
            processor_inputs["audio_mask"] = audio_mask
        # Get embeddings and per-layer inputs.
        inputs_embeds, per_layer_inputs = self.backbone.embedding_processor(
            processor_inputs
        )
        # Prepare attention mask for caching.
        batch_size = keras.ops.shape(token_ids)[0]
        max_length = keras.ops.shape(token_ids)[1]
        # Create causal attention mask.
        if padding_mask is None:
            padding_mask = keras.ops.ones(
                (batch_size, max_length), dtype="bool"
            )
        attention_mask = keras.ops.cast(padding_mask, dtype="bool")
        attention_mask = keras.ops.expand_dims(attention_mask, axis=1)
        attention_mask = keras.ops.expand_dims(attention_mask, axis=1)
        # Each decoder layer has a cache; we update them separately.
        hidden_states, new_cache = self.backbone.language_model(
            input_ids=token_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            cache=cache,
            cache_update_index=cache_update_index,
            cache_update_mask=cache_update_mask,
        )
        logits = self.backbone.language_model.token_embedding(
            hidden_states, reverse=True
        )
        return logits, hidden_states, new_cache

    def _build_cache(
        self,
        token_ids,
        pixel_values,
        input_features,
        input_features_mask,
        vision_indices,
        audio_indices,
        vision_mask,
        audio_mask,
        padding_mask,
    ):
        """Build an empty cache for use with `call_with_cache()`."""
        batch_size = keras.ops.shape(token_ids)[0]
        max_length = keras.ops.shape(token_ids)[1]
        num_layers = self.backbone.num_hidden_layers
        num_heads = self.backbone.num_key_value_heads
        head_dim = self.backbone.head_dim
        shape = [batch_size, num_layers, 2, num_heads, max_length, head_dim]
        cache = keras.ops.zeros(shape, dtype=self.compute_dtype)
        # Seed the cache.
        _, hidden_states, cache = self.call_with_cache(
            token_ids=token_ids,
            cache=cache,
            cache_update_index=0,
            pixel_values=pixel_values,
            input_features=input_features,
            input_features_mask=input_features_mask,
            vision_indices=vision_indices,
            audio_indices=audio_indices,
            vision_mask=vision_mask,
            audio_mask=audio_mask,
            padding_mask=padding_mask,
            cache_update_mask=None,
        )
        return hidden_states, cache

    def generate_step(self, inputs, stop_token_ids=[106]):
        """A compilable generation function for a single batch of inputs.

        This function represents the inner, XLA-compilable, generation function
        for a single batch of inputs. Inputs should have the same structure as
        model inputs, a dictionary with keys for token_ids, padding_mask, and
        optionally images, audios, vision_mask, audio_mask, etc.

        Args:
            inputs: A dictionary with keys for the model inputs including
                `"token_ids"`, `"padding_mask"`, and optionally `"images"`,
                `"audios"`, `"input_features"`, `"input_features_mask"`,
                `"vision_mask"`, `"audio_mask"`, `"vision_indices"`,
                `"audio_indices"`.
            stop_token_ids: Tuple of id's of end token's to stop on. If all
                sequences have produced a new stop token, generation
                will stop.
        """
        token_ids = inputs["token_ids"]
        padding_mask = inputs["padding_mask"]
        # Extract multimodal inputs.
        images = inputs.get("images", None)
        pixel_values = images
        input_features = inputs.get("input_features", None)
        input_features_mask = inputs.get("input_features_mask", None)
        vision_indices = inputs.get("vision_indices", None)
        audio_indices = inputs.get("audio_indices", None)
        vision_mask = inputs.get("vision_mask", None)
        audio_mask = inputs.get("audio_mask", None)
        audios = inputs.get("audios", None)
        # Handle unbatched inputs by adding batch dimension.
        if pixel_values is not None and len(keras.ops.shape(pixel_values)) == 4:
            pixel_values = keras.ops.expand_dims(pixel_values, axis=0)
        if audios is not None and len(keras.ops.shape(audios)) == 2:
            audios = keras.ops.expand_dims(audios, axis=0)
        if vision_mask is not None and len(keras.ops.shape(vision_mask)) == 1:
            vision_mask = keras.ops.expand_dims(vision_mask, axis=0)
        if (
            vision_indices is not None
            and len(keras.ops.shape(vision_indices)) == 1
        ):
            vision_indices = keras.ops.expand_dims(vision_indices, axis=0)
        if (
            input_features is not None
            and len(keras.ops.shape(input_features)) == 2
        ):
            input_features = keras.ops.expand_dims(input_features, axis=0)
        if (
            input_features_mask is not None
            and len(keras.ops.shape(input_features_mask)) == 1
        ):
            input_features_mask = keras.ops.expand_dims(
                input_features_mask, axis=0
            )
        if audio_mask is not None and len(keras.ops.shape(audio_mask)) == 1:
            audio_mask = keras.ops.expand_dims(audio_mask, axis=0)
        if (
            audio_indices is not None
            and len(keras.ops.shape(audio_indices)) == 1
        ):
            audio_indices = keras.ops.expand_dims(audio_indices, axis=0)
        # Create and seed cache with a single forward pass.
        hidden_states, cache = self._build_cache(
            token_ids,
            pixel_values,
            input_features,
            input_features_mask,
            vision_indices,
            audio_indices,
            vision_mask,
            audio_mask,
            padding_mask,
        )
        # Compute the lengths of all user inputted tokens ids.
        row_lengths = keras.ops.sum(
            keras.ops.cast(padding_mask, "int32"), axis=-1
        )
        # Start at the first index that has no user inputted id.
        index = keras.ops.min(row_lengths)

        def next(prompt, cache, index):
            # The cache index is the index of our previous token.
            cache_update_index = index - 1
            batch_size = keras.ops.shape(prompt)[0]
            prompt = keras.ops.slice(prompt, [0, index - 1], [batch_size, 1])
            sliced_cache_update_mask = keras.ops.slice(
                ~padding_mask, [0, index - 1], [batch_size, 1]
            )
            logits, hidden_states, cache = self.call_with_cache(
                token_ids=prompt,
                cache=cache,
                cache_update_index=cache_update_index,
                cache_update_mask=sliced_cache_update_mask,
            )
            return (
                keras.ops.squeeze(logits, axis=1),
                keras.ops.squeeze(hidden_states, axis=1),
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
        # Compute an output padding mask with the token ids we updated.
        if stop_token_ids is not None:
            # Build a mask of `stop_token_ids` locations not in the original
            # prompt (not in locations where `padding_mask` is True).
            end_locations = any_equal(
                token_ids, stop_token_ids, keras.ops.logical_not(padding_mask)
            )
            end_locations = keras.ops.cast(end_locations, "int32")
            # Use cumsum to get ones in all locations after end_locations.
            cumsum = keras.ops.cast(
                keras.ops.cumsum(end_locations, axis=-1), "int32"
            )
            overflow = cumsum - end_locations
            # Our padding mask is the inverse of these overflow locations.
            padding_mask = keras.ops.logical_not(
                keras.ops.cast(overflow, "bool")
            )
        else:
            # Without early stopping, all locations will have been updated.
            padding_mask = keras.ops.ones_like(token_ids, dtype="bool")
        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }

    def generate(
        self,
        inputs,
        max_length=None,
        stop_token_ids="auto",
        strip_prompt=False,
    ):
        # If `auto`, add end_of_turn as a stop token too.
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
            ]
            # Add end_of_turn token if available.
            end_of_turn_id = self.preprocessor.tokenizer.token_to_id(
                "<end_of_turn>"
            )
            if end_of_turn_id is not None:
                stop_token_ids.append(end_of_turn_id)
        return super().generate(
            inputs,
            max_length=max_length,
            stop_token_ids=stop_token_ids,
            strip_prompt=strip_prompt,
        )
