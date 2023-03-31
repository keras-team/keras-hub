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
"""Tests for Beam sampler."""

import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from keras_nlp.samplers.beam_sampler import BeamSampler


class BeamSamplerTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        super().setUp()
        # Use a simple alphabet of lowercase characters to [0, 26).
        self.int_lookup = {i: chr(i + ord("a")) for i in range(26)}
        self.char_lookup = {v: k for k, v in self.int_lookup.items()}
        self.batch_size = 1
        self.length = 12
        self.vocab_size = len(self.int_lookup)

        def next(prompt, state, index):
            # Return a distribution favoring the next char in state.
            logits = tf.one_hot(state[:, index], self.vocab_size) * 1e9
            return logits, state

        self.next = next
        self.sampler = BeamSampler(num_beams=5)
        self.sampler_all_beams = BeamSampler(num_beams=5, return_all_beams=True)

    def join_as_string(self, x):
        return ["".join([self.int_lookup[i] for i in s]) for s in x.numpy()]

    def test_stateless_call(self):
        def next(prompt, state, index):
            # Return a distribution favoring the first token in the vocab.
            logits = np.zeros((self.batch_size, self.vocab_size))
            logits[:, 0] = 1e9
            return tf.constant(logits, dtype="float32"), state

        prompt = tf.fill((self.batch_size, self.length), self.char_lookup["z"])
        output = self.sampler(
            next=next,
            prompt=prompt,
            index=5,
        )
        self.assertEqual(self.join_as_string(output), ["zzzzzaaaaaaa"])

    def test_stateful_call(self):
        state_chars = list("sequentially")
        state = tf.constant([[self.char_lookup[c] for c in state_chars]])
        prompt = tf.fill((self.batch_size, self.length), self.char_lookup["z"])
        output = self.sampler(
            next=self.next,
            prompt=prompt,
            state=state,
        )
        self.assertEqual(self.join_as_string(output), ["sequentially"])

    def test_return_all_beams(self):
        state_chars = list("sequentially")
        state = tf.constant([[self.char_lookup[c] for c in state_chars]])
        prompt = tf.fill((self.batch_size, self.length), self.char_lookup["z"])
        sorted_prompts, sorted_log_probs = self.sampler_all_beams(
            next=self.next,
            prompt=prompt,
            state=state,
        )

        self.assertEqual(
            sorted_prompts.shape, (self.batch_size, 5, self.length)
        )
        self.assertEqual(sorted_log_probs.shape, (self.batch_size, 5))
        self.assertTrue(
            tf.reduce_all(sorted_log_probs[:, 1:] <= sorted_log_probs[:, :-1])
        )
        self.assertEqual(
            self.join_as_string(sorted_prompts[:, 0, :]), ["sequentially"]
        )

    def test_early_stopping(self):
        state_chars = list("sequentially")
        state = tf.constant([[self.char_lookup[c] for c in state_chars]])
        prompt = tf.fill((self.batch_size, self.length), self.char_lookup["z"])
        output = self.sampler(
            next=self.next,
            prompt=prompt,
            state=state,
            end_token_id=self.char_lookup["t"],
        )
        self.assertEqual(self.join_as_string(output), ["sequentzzzzz"])

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_compilation(self, jit_compile):
        state_chars = list("sequentially")
        state = tf.constant([[self.char_lookup[c] for c in state_chars]])
        prompt = tf.fill((self.batch_size, self.length), self.char_lookup["z"])

        @tf.function(jit_compile=jit_compile)
        def generate(prompt, state):
            return self.sampler(self.next, prompt=prompt, state=state)

        output = generate(prompt, state)
        self.assertEqual(self.join_as_string(output), ["sequentially"])
