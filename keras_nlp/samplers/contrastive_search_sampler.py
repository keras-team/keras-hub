# Copyright 2023 The KerasNLP Authors
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


import tensorflow as tf

# from absl import logging
from tensorflow import keras

# from keras_nlp.api_export import keras_nlp_export
from keras_nlp.samplers.sampler import Sampler
from keras_nlp.samplers.sampler import base_sampler_args_docstring
from keras_nlp.samplers.sampler import call_args_docstring
from keras_nlp.utils.python_utils import format_docstring


@format_docstring(
    base_sampler_args=base_sampler_args_docstring, call_args=call_args_docstring
)
@keras.utils.register_keras_serializable(package="keras_nlp")
class ContrastiveSampler(Sampler):
    """Contrastive Sampler class.

    Contrastive sampler is a variant of top-k sampler,it randomly selects a token from the
    tokens of top K probability, with  selection chance determined by the probability.
    adds a penalty based on max similarity with previously seen tokens.

      Args:
          num_beams: int. The number of beams that should be kept at each
              time-step. `num_beams` should be strictly positive.
          {{base_sampler_args}}

      Call Args:
          {{call_args}}

      Examples:
      ```python
      VOCAB_SIZE = 10

      # Create a dummy model to predict the next token.
      model = keras.Sequential(
          [
              keras.Input(shape=[None]),
              keras.layers.Embedding(
                  input_dim=VOCAB_SIZE,
                  output_dim=16,
              ),
              keras.layers.Dense(VOCAB_SIZE, activation="softmax"),
          ]
      )

      # Define a function that outputs the next token's probability for each token
      # in the input sequence.
      def token_probability_fn(inputs, mask):
          return model(inputs)

      prompt = tf.fill((8, 1), 1)

      sampler = keras_nlp.samplers.BeamSampler(num_beams=3)
      # Print the generated sequence (token ids).
      print(sampler(prompt, token_probability_fn, max_length=10))
      ```
    """

    def __init__(
        self,
        k=5,
        seed=None,
        jit_compile=True,
        run_eagerly=False,
    ):
        self.k = k
        self.seed = seed
        super().__init__(jit_compile=jit_compile, run_eagerly=run_eagerly)

    def get_next_token(self, next_token_probs):
        pass

    def sample(
        self,
        prompt,
        token_probability_fn,
        ask,
        batch_size,
        num_steps,
        current_index,
        from_logits=True,
    ):
        batch_size, max_length = tf.shape(prompt)[0], tf.shape(prompt)[1]
        max_length = tf.cast(max_length, num_steps.dtype)
        # The index of the last non-padding token in prompt. Since all sequences
        # are aligned to the right side, the index is the same for all.
        current_index = max_length - num_steps
        return batch_size, current_index

        def one_step(current_index, prompt, mask):
            #################################################################
            # 1. Get top-k tokens along with their probibility and representation.
            #
            top_k_pred, top_k_indices = tf.math.top_k(
                next_token_probs, k=self.k, sorted=False
            )
            # Your code goes here!
            #################################################################
            next_token = tf.random.categorical(
                tf.math.log(top_k_pred), 1, seed=self.seed
            )

            #################################################################
            # 2. Compute the penalty for each token in the top-k selection.
            #
            # Your code goes here!
            #################################################################

            #################################################################
            # 3. Update the corresponding index and mask.
            #
            # Your code goes here!
            #################################################################

            # Run a while loop till `max_length` of tokens has been generated.
            current_index, prompt, mask = tf.while_loop(
                cond=lambda current_index, prompt, mask: tf.less(
                    current_index, max_length
                ),
                body=one_step,
                loop_vars=(current_index, prompt, mask),
            )
            return prompt

        def get_config(self):
            config = super().get_config()

            config.update({"num_beams": self.num_beams})
            return config
