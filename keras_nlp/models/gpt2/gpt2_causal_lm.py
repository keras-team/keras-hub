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
"""BERT task specific models and heads."""

import copy

import tensorflow as tf
from tensorflow import keras

from keras_nlp.models.gpt2.gpt2_backbone import GPT2Backbone
from keras_nlp.models.gpt2.gpt2_preprocessor import GPT2CausalLMPreprocessor
from keras_nlp.models.gpt2.gpt2_presets import backbone_presets
from keras_nlp.samplers.beam_sampler import BeamSampler
from keras_nlp.samplers.greedy_sampler import GreedySampler
from keras_nlp.samplers.top_k_sampler import TopKSampler
from keras_nlp.samplers.top_p_sampler import TopPSampler
from keras_nlp.utils.pipeline_model import PipelineModel
from keras_nlp.utils.python_utils import classproperty


@keras.utils.register_keras_serializable(package="keras_nlp")
class GPT2CausalLM(PipelineModel):
    def __init__(self, backbone, preprocessor=None, **kwargs):

        inputs = backbone.input
        x = backbone(inputs)
        x = tf.matmul(
            x,
            backbone.get_layer("token_embedding").embeddings,
            transpose_b=True,
        )
        outputs = tf.keras.layers.Softmax()(x)
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

    def _get_generator(self, identifier):
        maps = {
            "greedy": GreedySampler(),
            "top_k": TopKSampler(k=5, from_logits=False),
            "top_p": TopPSampler(p=0.1, from_logits=False),
            "beam": BeamSampler(num_beams=5),
        }
        return maps[identifier]

    def _get_token_probability(self, prompt, mask):
        model_inputs = {
            "token_ids": prompt,
            "padding_mask": mask,
        }
        probs = self(model_inputs)
        return probs

    def generate(self, prompt, max_length, generator="top_k"):
        """Pick one method as the default generation algo."""
        if isinstance(generator, str):
            generator = self._get_generator(generator)
        prompt = self.preprocessor.tokenizer(prompt)
        generated = generator(self._get_token_probability, prompt, max_length)
        return self.preprocessor.tokenizer.detokenize(generated)
