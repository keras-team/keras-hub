import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.qwen3_omni_moe.qwen3_omni_moe_backbone import Qwen3OmniMoeBackbone
from keras_hub.src.models.qwen3_omni_moe.qwen3_omni_moe_causal_lm_preprocessor import (
    Qwen3OmniMoeCausalLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export(
    "keras_hub.models.Qwen3OmniMoeCausalLM",
)
class Qwen3OmniMoeCausalLM(CausalLM):
    """An end-to-end Qwen3-Omni MoE model for causal language modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens. This task setup can be used to train the model unsupervised on plain
    text input, or to autoregressively generate plain text similar to the data
    used for training. This task can be used for pre-training or fine-tuning a
    Qwen3-Omni MoE model, simply by calling `fit()`.

    This model has a `generate()` method, which generates text based on a
    prompt. The generation strategy used is controlled by an additional
    `sampler` argument on `compile()`. You can recompile the model with
    different `keras_hub.samplers` objects to control the generation.
    By default, `"greedy"` sampling will be used.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to string inputs during
    `fit()`, `predict()`, `evaluate()`, and `generate()`. This is done by
    default when creating the model with `from_preset()`.

    The Qwen3-Omni MoE architecture leverages a Mixture of Experts (MoE) design
    with multimodal capabilities, supporting text, audio, and vision inputs.
    Each transformer layer uses a sparse set of experts to process tokens
    efficiently, making it suitable for large-scale multimodal AI tasks.

    Args:
        backbone: A `keras_hub.models.Qwen3OmniMoeBackbone` instance.
        preprocessor: A `keras_hub.models.Qwen3OmniMoeCausalLMPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.

    Examples:

    Use `generate()` to do text generation.
    ```python
    qwen3_omni_moe_lm = keras_hub.models.Qwen3OmniMoeCausalLM.from_preset(
        "qwen3_omni_moe_7b"
    )
    qwen3_omni_moe_lm.generate("I want to say", max_length=30)

    # Generate with batched prompts.
    qwen3_omni_moe_lm.generate(["This is a", "Where are you"], max_length=30)
    ```

    Compile the `generate()` function with a custom sampler.
    ```python
    qwen3_omni_moe_lm = keras_hub.models.Qwen3OmniMoeCausalLM.from_preset(
        "qwen3_omni_moe_7b"
    )
    qwen3_omni_moe_lm.compile(sampler="top_k")
    qwen3_omni_moe_lm.generate("I want to say", max_length=30)

    qwen3_omni_moe_lm.compile(sampler=keras_hub.samplers.BeamSampler(num_beams=2))
    qwen3_omni_moe_lm.generate("I want to say", max_length=30)
    ```

    Use `generate()` without preprocessing.
    ```python
    prompt = {
        # Token ids for "<bos> Qwen3-Omni is".
        "token_ids": np.array([[2, 12345, 678, 0, 0, 0, 0]] * 2),
        # Use `"padding_mask"` to indicate values that should not be overridden.
        "padding_mask": np.array([[1, 1, 1, 0, 0, 0, 0]] * 2),
    }

    qwen3_omni_moe_lm = keras_hub.models.Qwen3OmniMoeCausalLM.from_preset(
        "qwen3_omni_moe_7b",
        preprocessor=None,
    )
    qwen3_omni_moe_lm.generate(prompt)
    ```

    Call `fit()` on a single batch.
    ```python
    features = ["The quick brown fox jumped.", "I forgot my homework."]
    qwen3_omni_moe_lm = keras_hub.models.Qwen3OmniMoeCausalLM.from_preset(
        "qwen3_omni_moe_7b"
    )
    qwen3_omni_moe_lm.fit(x=features, batch_size=2)
    ```

    Call `fit()` with LoRA fine-tuning enabled.
    ```python
    features = ["The quick brown fox jumped.", "I forgot my homework."]
    qwen3_omni_moe_lm = keras_hub.models.Qwen3OmniMoeCausalLM.from_preset(
        "qwen3_omni_moe_7b"
    )
    qwen3_omni_moe_lm.backbone.enable_lora(rank=4)
    qwen3_omni_moe_lm.fit(x=features, batch_size=2)
    ```

    Call `fit()` without preprocessing.
    ```python
    x = {
        # Token ids for "<bos> Qwen3-Omni is a multimodal model<eos>"
        "token_ids": np.array([[2, 12345, 678, 543, 9876, 1, 0, 0]] * 2),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 0, 0]] * 2),
    }
    y = np.array([[12345, 678, 543, 9876, 1, 0, 0, 0]] * 2)
    sw = np.array([[1, 1, 1, 1, 1, 0, 0, 0]] * 2)

    qwen3_omni_moe_lm = keras_hub.models.Qwen3OmniMoeCausalLM.from_preset(
        "qwen3_omni_moe_7b",
        preprocessor=None,
    )
    qwen3_omni_moe_lm.fit(x=x, y=y, sample_weight=sw, batch_size=2)
    ```

    Custom backbone and vocabulary.
    ```python
    tokenizer = keras_hub.models.Qwen3OmniMoeTokenizer(
        proto="qwen3_omni_moe_vocab.spm",
    )
    preprocessor = keras_hub.models.Qwen3OmniMoeCausalLMPreprocessor(
        tokenizer=tokenizer,
        sequence_length=128,
    )
    backbone = keras_hub.models.Qwen3OmniMoeBackbone(
        vocabulary_size=151936,
        num_layers=32,
        num_query_heads=32,
        num_key_value_heads=4,
        hidden_dim=4096,
        intermediate_dim=11008,
        num_experts=8,
        num_experts_per_tok=2,
        head_dim=128,
        max_sequence_length=32768,
    )
    qwen3_omni_moe_lm = keras_hub.models.Qwen3OmniMoeCausalLM(
        backbone=backbone,
        preprocessor=preprocessor,
    )
    qwen3_omni_moe_lm.fit(x=features, batch_size=2)
    ```
    """

    backbone_cls = Qwen3OmniMoeBackbone
    preprocessor_cls = Qwen3OmniMoeCausalLMPreprocessor

    def __init__(self, backbone, preprocessor=None, **kwargs):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

        # === Functional Model ===
        # This must be "backbone.input" i.e. the full input structure,
        # rather than "backbone.inputs" which is the flattened list of inputs.
        inputs = backbone.input
        hidden_states = backbone(inputs)
        outputs = backbone.token_embedding(hidden_states, reverse=True)
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

    def call_with_cache(
        self,
        token_ids,
        cache,
        cache_update_index,
    ):
        """Forward pass of `Qwen3OmniMoeCausalLM` with cache.

        `call_with_cache` adds an additional forward pass for the model for
        autoregressive inference. Unlike calling the model directly, this method
        allows caching previous key/value Tensors in multi-head attention layer,
        and avoids recomputing the outputs of seen tokens.

        Args:
            token_ids: a dense int Tensor with shape `(batch_size, max_length)`.
            cache: a dense float Tensor, the cache of key and value.
            cache_update_index: int, or int Tensor. The index of current inputs
            in the whole sequence.

        Returns:
            A (logits, hidden_states, cache) tuple. Where `logits` is the
            language model logits for the input token_ids, `hidden_states` is
            the final hidden representation of the input tokens, and `cache` is
            the decoding cache.
        """
        x = self.backbone.token_embedding(token_ids)
        # Each decoder layer has a cache; we update them separately.
        updated_cache = []
        for i in range(self.backbone.num_layers):
            layer = self.backbone.transformer_decoder.layers[i]
            x, cache = layer(
                x,
                cache=cache[i],
                cache_update_index=cache_update_index,
            )
            updated_cache.append(cache)
        x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)
        return logits, x, updated_cache

    def compute_loss(
        self,
        x=None,
        y=None,
        y_pred=None,
        sample_weight=None,
    ):
        """Compute the loss of the model.

        Args:
            x: Input data.
            y: Target data.
            y_pred: Predictions returned by the model.
            sample_weight: Sample weights for the loss computation.

        Returns:
            The loss of the model.
        """
        if y is None:
            return None

        # If y_pred is provided, use it directly
        if y_pred is not None:
            # y_pred is already computed, just compute the loss
            y_pred = y_pred
        else:
            # Forward pass through the model
            y_pred = self(x)

        # Compute cross-entropy loss
        y_true = ops.cast(y, dtype="int32")
        y_pred = ops.cast(y_pred, dtype="float32")
        
        # Flatten for loss computation
        y_true_flat = ops.reshape(y_true, (-1,))
        y_pred_flat = ops.reshape(y_pred, (-1, ops.shape(y_pred)[-1]))
        
        # Compute cross-entropy loss
        loss = keras.losses.sparse_categorical_crossentropy(
            y_true_flat, y_pred_flat, from_logits=True
        )
        
        # Apply sample weights if provided
        if sample_weight is not None:
            sample_weight_flat = ops.reshape(sample_weight, (-1,))
            loss = loss * sample_weight_flat
        
        # Add auxiliary losses from MoE layers
        total_loss = ops.mean(loss)
        for auxiliary_loss in self.losses:
            total_loss += auxiliary_loss
            
        return total_loss
