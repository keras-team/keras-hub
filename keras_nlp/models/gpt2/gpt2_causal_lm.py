# Copyright 2022 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GPT2 Causal LM (Language Model)."""

import copy

import tensorflow as tf
from tensorflow import keras

import keras_nlp
from keras_nlp.models.gpt2.gpt2_backbone import GPT2Backbone
from keras_nlp.models.gpt2.gpt2_causal_lm_preprocessor import (
    GPT2CausalLMPreprocessor,
)
from keras_nlp.models.gpt2.gpt2_presets import backbone_presets
from keras_nlp.utils.pipeline_model import PipelineModel
from keras_nlp.utils.python_utils import classproperty


@keras.utils.register_keras_serializable(package="keras_nlp")
class GPT2CausalLM(PipelineModel):
    """GPT2 Causal LM task model.

    Causal LM is predicting the next token based on previous tokens, which is
    the way GPT2 gets pretrained. Users can finetune `GPT2CausalLM` to generate
    text similar to the custom dataset. `GPT2CausalLM` also has a public method
    `generate()`, which generates text based on given prompt.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to raw inputs during
    `fit()`, `predict()`, and `evaluate()`. This is done by default when
    creating the model with `from_preset()`.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/openai/gpt-2).

    Args:
        backbone: A `keras_nlp.models.GPT2Backbone` instance.
        preprocessor: A `keras_nlp.models.GPT2CausalLMPreprocessor` or `None`.
            If `None`, this model will not apply preprocessing, and inputs
            should be preprocessed before calling the model.


    """

    def __init__(self, backbone, preprocessor=None, **kwargs):
        inputs = backbone.input
        x = backbone(inputs)
        outputs = tf.matmul(
            x,
            backbone.get_layer("token_embedding").embeddings,
            transpose_b=True,
        )
        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            include_preprocessing=preprocessor is not None,
            **kwargs,
        )

        self.preprocessor = preprocessor
        self.backbone = backbone

    def preprocess_samples(self, x, y=None, sample_weight=None):
        return self.preprocessor(x, y=y, sample_weight=sample_weight)

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)

    @classmethod
    def from_preset(
        cls,
        preset,
        load_weights=True,
        **kwargs,
    ):
        if "preprocessor" not in kwargs:
            kwargs["preprocessor"] = GPT2CausalLMPreprocessor.from_preset(
                preset
            )

        # Check if preset is backbone-only model.
        if preset in GPT2Backbone.presets:
            backbone = GPT2Backbone.from_preset(preset, load_weights)
            return cls(backbone, **kwargs)

        # Otherwise must be one of class presets.
        # Currently no classifier-level presets, so we raise ValueError.
        if preset not in cls.presets:
            raise ValueError(
                "`preset` must be one of "
                f"""{", ".join(cls.presets)}. Received: {preset}."""
            )

    def _get_token_probability(self, prompt, mask):
        model_inputs = {
            "token_ids": prompt,
            "padding_mask": mask,
        }
        return self(model_inputs)

    def generate(
        self,
        prompt,
        max_length,
        end_token="<|endoftext|>",
        sampler="top_k",
    ):
        """Generate text.

        This method generates text based on given `prompt`. Generation will
        continue until `max_length` is met, and all generated tokens after
        `end_token` will be truncated.

        Args:
            prompt: a string, string Tensor or string RaggedTensor. The prompt
                text for generation.
            max_length: int. The max length of generated sequence.
            end_token: string, defaults to "<|endoftext|>", which is the default
                end token of GPT2. The token marking the end of the sequence,
                tokens generated after the end token will be truncated.
        """
        end_token_id = self.preprocessor.tokenizer.token_to_id(end_token)

        if isinstance(sampler, str):
            sampler = keras_nlp.samplers.get(sampler)
        prompt = self.preprocessor.tokenizer(prompt)
        generated = sampler(
            prompt,
            self._get_token_probability,
            max_length=max_length,
            end_token_id=end_token_id,
        )
        return self.preprocessor.tokenizer.detokenize(generated)
