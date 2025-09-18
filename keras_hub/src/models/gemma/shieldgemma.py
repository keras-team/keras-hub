import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.gemma import gemma_causal_lm
from keras_hub.src.models.task import Task


class ShieldGemmaViolationProbaility(keras.layers.Layer):
    """Relative probabilities for the 'Yes' (violating) and 'No' tokens."""

    def __init__(self, yes_token_idx, no_token_idx, **kw):
        super().__init__(**kw)
        self.yes_token_idx = yes_token_idx
        self.no_token_idx = no_token_idx

    def call(self, logits, padding_mask):
        last_prompt_index = keras.ops.cast(
            keras.ops.sum(padding_mask, axis=1) - 1, "int32"
        )
        last_logits = keras.ops.take(logits, last_prompt_index, axis=1)[:, 0]
        yes_logits = last_logits[:, self.yes_token_idx]
        no_logits = last_logits[:, self.no_token_idx]
        yes_no_logits = keras.ops.stack((yes_logits, no_logits), axis=1)
        return keras.ops.softmax(yes_no_logits, axis=1)


@keras_hub_export("keras_hub.models.ShieldGemma")
class ShieldGemma(Task):
    """A ShieldGemma model for safety content moderation, built on Gemma 2.

    ShieldGemma is a Gemma 2 variant fine-tuned to detect and predict violations
    of four harm types&mdash;Harrassment, Hate Speech, Dangerous Content, and
    Sexual Content&mdash;in text content from a user or model. Architecturally,
    the weights are the same as any other Gemma 2 class, but the prediction is
    augmented with a final layer that returns the probability that the provided
    content violates the harm type specified in the prompt. The probability is
    computed as the relative probabilities of the `Yes` (violating) and `No`
    (non-violating) tokens at the final prompt token, i.e., is the next most
    likley token a yes or a no.

    Links:

    *   https://arxiv.org/abs/2407.21772
    *   https://ai.google.dev/gemma/docs/shieldgemma/model_card
    *   https://ai.google.dev/responsible/docs/safeguards/shieldgemma
    *   https://www.kaggle.com/models/google/shieldgemma

    Args:
        gemma: A `keras_hub.models.GemmaCausalLM` initialized with ShieldGemma
            weights.

    Examples:

    Coming soon.
    """

    backbone_cls = gemma_causal_lm.GemmaCausalLM.backbone_cls
    preprocessor_cls = gemma_causal_lm.GemmaCausalLM.preprocessor_cls

    def __init__(self, gemma: gemma_causal_lm.GemmaCausalLM, **kwargs):
        # === Layers ===
        self.gemma = gemma
        self.backbone = self.gemma.backbone
        self.preprocessor = self.gemma.preprocessor
        self.yes_no_layer = ShieldGemmaViolationProbaility(
            yes_token_idx=self.preprocessor.tokenizer.token_to_id("Yes"),
            no_token_idx=self.preprocessor.tokenizer.token_to_id("No"),
        )

        # === Functional Model ===
        inputs = self.gemma.input
        logits = self.gemma(inputs)
        outputs = self.yes_no_layer(logits, inputs["padding_mask"])
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

    @classmethod
    def from_preset(cls, **kwargs):
        """Instantiate a `keras_hub.models.ShieldGemma` from a model preset."""
        gemma = gemma_causal_lm.GemmaCausalLM.from_preset(**kwargs)
        return cls(gemma)
