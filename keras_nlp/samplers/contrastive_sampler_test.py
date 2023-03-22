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
"""Tests for Top-K sampler."""

import tensorflow as tf
from absl.testing import parameterized

from keras_nlp.samplers.contrastive_sampler import ContrastiveSampler


class ContrastiveSamplerTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        super().setUp()
        # Use a simple alphabet of lowercase characters to [0, 26).
        self.int_lookup = {i: chr(i + ord("a")) for i in range(26)}
        self.char_lookup = {v: k for k, v in self.int_lookup.items()}
        self.batch_size = 1
        self.length = 13
        self.hidden_dim = 3
        self.vocab_size = len(self.int_lookup)
        self.init_hidden_states = tf.ones(
            [
                self.batch_size,
                self.length,
                self.hidden_dim,
            ]
        )

        def next(prompt, state, index):
            batch_size = tf.shape(prompt)[0]
            # Return a distribution favoring the next char in state.
            logits = tf.one_hot(state[:, index], self.vocab_size) * 1e9
            hidden_states = tf.ones([batch_size, 1, self.hidden_dim])
            return logits, state, hidden_states

        self.next = next
        self.sampler = ContrastiveSampler(k=5, alpha=0.2)

    def join_as_string(self, x):
        return ["".join([self.int_lookup[i] for i in s]) for s in x.numpy()]

    def test_stateless_call(self):
        def next(prompt, state, index):
            # Return a distribution favoring the first token in the vocab.
            batch_size = tf.shape(prompt)[0]
            logits = tf.one_hot(0, self.vocab_size) * 1e9
            logits = tf.reshape(
                tf.repeat([logits], batch_size, axis=0),
                [batch_size, -1],
            )
            hidden_states = tf.ones([batch_size, 1, self.hidden_dim])
            return logits, state, hidden_states

        prompt = tf.fill((self.batch_size, self.length), self.char_lookup["z"])
        output = self.sampler(
            next=next,
            prompt=prompt,
            index=5,
            init_hidden_states=self.init_hidden_states,
        )
        self.assertEqual(self.join_as_string(output), ["zzzzzaaaaaaaa"])

    def test_stateful_call(self):
        state_chars = list("zsequentiallyy")
        state = tf.constant([[self.char_lookup[c] for c in state_chars]])
        prompt = tf.fill((self.batch_size, self.length), self.char_lookup["z"])
        output = self.sampler(
            next=self.next,
            prompt=prompt,
            state=state,
            index=1,
            init_hidden_states=self.init_hidden_states,
        )
        self.assertEqual(self.join_as_string(output[:, 1:]), ["sequentially"])

    def test_early_stopping(self):
        state_chars = list("zsequentiallyy")
        state = tf.constant([[self.char_lookup[c] for c in state_chars]])
        prompt = tf.fill((self.batch_size, self.length), self.char_lookup["z"])
        output = self.sampler(
            next=self.next,
            prompt=prompt,
            state=state,
            end_token_id=self.char_lookup["t"],
            index=1,
            init_hidden_states=self.init_hidden_states,
        )
        self.assertEqual(self.join_as_string(output[:, 1:]), ["sequentzzzzz"])

    def test_outputs_in_top_k(self):
        def next(prompt, state, index):
            batch_size = tf.shape(prompt)[0]
            # Return a distribution where each id is progressively less likely.
            logits = tf.range(self.vocab_size, 0, -1, dtype="float32")
            logits = tf.repeat(logits[tf.newaxis, :], batch_size, axis=0)
            hidden_states = tf.ones([batch_size, 1, self.hidden_dim])
            return logits, state, hidden_states

        prompt = tf.fill((self.batch_size, self.length), self.char_lookup["z"])
        output = self.sampler(
            next=next,
            prompt=prompt,
            index=1,
            init_hidden_states=self.init_hidden_states,
        )
        output_ids = set(output[0, 1:].numpy())
        self.assertContainsSubset(output_ids, range(5))

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_compilation(self, jit_compile):
        state_chars = list("zsequentiallyy")
        state = tf.constant([[self.char_lookup[c] for c in state_chars]])
        prompt = tf.fill((self.batch_size, self.length), self.char_lookup["z"])

        @tf.function(jit_compile=jit_compile)
        def generate(prompt, state):
            return self.sampler(
                self.next,
                prompt=prompt,
                state=state,
                index=1,
                init_hidden_states=self.init_hidden_states,
            )

        output = generate(prompt, state)
        self.assertEqual(self.join_as_string(output[:, 1:]), ["sequentially"])
