from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.phi3.phi3_causal_lm import Phi3CausalLM
from keras_hub.src.models.phi4.phi4_backbone import Phi4Backbone
from keras_hub.src.models.phi4.phi4_causal_lm_preprocessor import (
    Phi4CausalLMPreprocessor,
)


@keras_hub_export("keras_hub.models.Phi4CausalLM")
class Phi4CausalLM(Phi3CausalLM):
    """An end-to-end Phi4 model for causal language modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens. This task setup can be used to train the model unsupervised on
    plain text input, or to autoregressively generate plain text similar to
    the data used for training. This task can be used for pre-training or
    fine-tuning a Phi-4 model, simply by calling `fit()`.

    This model has a `generate()` method, which generates text based on a
    prompt. The generation strategy used is controlled by an additional
    `sampler` argument on `compile()`. You can recompile the model with
    different `keras_hub.samplers` objects to control the generation. By
    default, `"top_k"` sampling will be used.

    Args:
        backbone: A `keras_hub.models.Phi4Backbone` instance.
        preprocessor: A `keras_hub.models.Phi4CausalLMPreprocessor` or `None`.
            If `None`, this model will not apply preprocessing, and inputs
            should be preprocessed before calling the model.
    """

    backbone_cls = Phi4Backbone
    preprocessor_cls = Phi4CausalLMPreprocessor
