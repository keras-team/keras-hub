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
"""Tests for Contrastive Sampler."""

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
        self.length = 12
        self.hidden_dim = 3
        self.vocab_size = len(self.int_lookup)
        self.hidden_states = tf.ones(
            [
                self.batch_size,
                self.length,
                self.hidden_dim,
            ]
        )

        def next(prompt, cache, index):
            batch_size = tf.shape(prompt)[0]
            # Return a distribution favoring the next char in cache.
            logits = tf.one_hot(cache[:, index], self.vocab_size) * 1e9
            hidden_states = tf.ones([batch_size, self.hidden_dim])
            return logits, hidden_states, cache

        self.next = next
        self.sampler = ContrastiveSampler(k=5, alpha=0.2, temperature=1.0)

    def join_as_string(self, x):
        return ["".join([self.int_lookup[i] for i in s]) for s in x.numpy()]

    def test_stateless_call(self):
        def next(prompt, cache, index):
            # Return a distribution favoring the first token in the vocab.
            batch_size = tf.shape(prompt)[0]
            logits = (
                tf.one_hot(
                    tf.zeros(batch_size, dtype=tf.int32),
                    self.vocab_size,
                )
                * 1e9
            )
            hidden_states = tf.ones([batch_size, self.hidden_dim])
            return logits, hidden_states, cache

        prompt = tf.fill((self.batch_size, self.length), self.char_lookup["z"])
        output = self.sampler(
            next=next,
            prompt=prompt,
            index=5,
            hidden_states=self.hidden_states,
        )
        self.assertEqual(self.join_as_string(output), ["zzzzzaaaaaaa"])

    def test_stateful_call(self):
        cache_chars = list("sequentiallyy")
        cache = tf.constant([[self.char_lookup[c] for c in cache_chars]])
        prompt = tf.fill((self.batch_size, self.length), self.char_lookup["s"])
        output = self.sampler(
            next=self.next,
            prompt=prompt,
            cache=cache,
            index=1,
            hidden_states=self.hidden_states,
        )
        self.assertEqual(self.join_as_string(output), ["sequentially"])

    def test_early_stopping(self):
        cache_chars = list("sequentiallyy")
        cache = tf.constant([[self.char_lookup[c] for c in cache_chars]])
        prompt = tf.fill((self.batch_size, self.length), self.char_lookup["s"])
        output = self.sampler(
            next=self.next,
            prompt=prompt,
            cache=cache,
            end_token_id=self.char_lookup["t"],
            index=0,
            hidden_states=self.hidden_states,
        )
        self.assertEqual(self.join_as_string(output), ["sequentsssss"])

    def test_outputs_in_top_k(self):
        def next(prompt, cache, index):
            batch_size = tf.shape(prompt)[0]
            # Return a distribution where each id is progressively less likely.
            logits = tf.range(self.vocab_size, 0, -1, dtype="float32")
            logits = tf.repeat(logits[tf.newaxis, :], batch_size, axis=0)
            hidden_states = tf.ones([batch_size, self.hidden_dim])
            return logits, hidden_states, cache

        prompt = tf.fill((self.batch_size, self.length), self.char_lookup["z"])
        output = self.sampler(
            next=next,
            prompt=prompt,
            index=1,
            hidden_states=self.hidden_states,
        )
        output_ids = set(output[0, 1:].numpy())
        self.assertContainsSubset(output_ids, range(5))

    def test_alpha_penalty(self):
        def next(prompt, cache, index):
            batch_size = tf.shape(prompt)[0]
            best_token_id = self.char_lookup["h"]
            logits = tf.ones([batch_size, self.vocab_size])
            # Favoring `best_token_id` in the logits.
            logits += (
                tf.one_hot(
                    tf.zeros(self.batch_size, dtype=tf.int32) + best_token_id,
                    self.vocab_size,
                )
                * 1e9
            )

            # Set the hidden states for `best_token_id` as [1, 1, ..., 1], so it
            # gets the max similarity penality score.
            mask_of_best_token = prompt[:, index - 1] == best_token_id
            random_states = tf.random.uniform([batch_size, self.hidden_dim]) * (
                1 - tf.cast(mask_of_best_token, dtype=tf.float32)[:, tf.newaxis]
            )
            hidden_states = (
                tf.ones([batch_size, self.hidden_dim])
                * tf.cast(mask_of_best_token, dtype=tf.float32)[:, tf.newaxis]
            )
            hidden_states = hidden_states + random_states
            return logits, hidden_states, cache

        prompt = tf.fill((1, self.length), self.char_lookup["z"])
        hidden_states = tf.ones([1, self.length, self.hidden_dim]) + 1e-5
        output = self.sampler(
            next=next,
            prompt=prompt,
            index=5,
            hidden_states=hidden_states,
        )
        self.assertEqual(self.join_as_string(output), ["zzzzzhhhhhhh"])

        sampler = ContrastiveSampler(k=5, alpha=1.0)
        output = sampler(
            next=next,
            prompt=prompt,
            index=5,
            hidden_states=hidden_states,
        )
        self.assertTrue("h" not in self.join_as_string(output))

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_compilation(self, jit_compile):
        cache_chars = list("sequentiallyy")
        cache = tf.constant([[self.char_lookup[c] for c in cache_chars]])
        prompt = tf.fill((self.batch_size, self.length), self.char_lookup["s"])

        @tf.function(jit_compile=jit_compile)
        def generate(prompt, cache):
            return self.sampler(
                self.next,
                prompt=prompt,
                cache=cache,
                index=1,
                hidden_states=self.hidden_states,
            )

        output = generate(prompt, cache)
        self.assertEqual(self.join_as_string(output), ["sequentially"])
