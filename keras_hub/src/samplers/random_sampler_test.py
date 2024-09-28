import numpy as np
from keras import ops

from keras_hub.src.samplers.random_sampler import RandomSampler
from keras_hub.src.tests.test_case import TestCase


class RandomSamplerTest(TestCase):
    def setUp(self):
        super().setUp()
        # Use a simple alphabet of lowercase characters to [0, 25].
        self.int_lookup = {i: chr(i + ord("a")) for i in range(26)}
        self.char_lookup = {v: k for k, v in self.int_lookup.items()}
        self.batch_size = 1
        self.length = 12
        self.vocab_size = len(self.int_lookup)

        def next(prompt, cache, index):
            # Dummy hidden states.
            hidden_states = ops.ones([self.batch_size, 5])
            # Return a distribution favoring the next char in state.
            logits = ops.one_hot(cache[:, index], self.vocab_size) * 1e9
            return logits, hidden_states, cache

        self.next = next
        self.sampler = RandomSampler(temperature=1.0)

    def join_as_string(self, x):
        x = ops.convert_to_numpy(x)
        return ["".join([self.int_lookup[i] for i in s]) for s in x]

    def test_stateless_call(self):
        def next(prompt, cache, index):
            # Dummy hidden states.
            hidden_states = ops.ones([self.batch_size, 5])
            # Return a distribution favoring the first token in the vocab.
            logits = np.zeros((self.batch_size, self.vocab_size))
            logits[:, 0] = 1e9
            return ops.array(logits), hidden_states, cache

        prompt = ops.full((self.batch_size, self.length), self.char_lookup["z"])
        output = self.sampler(
            next=next,
            prompt=prompt,
            index=5,
        )
        self.assertEqual(self.join_as_string(output), ["zzzzzaaaaaaa"])

    def test_stateful_call(self):
        cache_chars = list("sequentially")
        cache = ops.array([[self.char_lookup[c] for c in cache_chars]])
        prompt = ops.full((self.batch_size, self.length), self.char_lookup["z"])
        output = self.sampler(
            next=self.next,
            prompt=prompt,
            cache=cache,
        )
        self.assertEqual(self.join_as_string(output), ["sequentially"])

    def test_temperature(self):
        def next(prompt, cache, index):
            # Dummy hidden states.
            hidden_states = ops.ones([self.batch_size, 5])
            logits = ops.arange(self.vocab_size, 0, -1, dtype="float32")
            logits = ops.reshape(logits[None, :], (self.batch_size, -1))
            return ops.array(logits), hidden_states, cache

        prompt = ops.full((self.batch_size, self.length), self.char_lookup["z"])

        output = RandomSampler(temperature=1e-5)(
            next=next,
            prompt=prompt,
        )
        self.assertAllEqual(output, ops.zeros_like(output))

    def test_early_stopping(self):
        cache_chars = list("sequentially")
        cache = ops.array([[self.char_lookup[c] for c in cache_chars]])
        prompt = ops.full((self.batch_size, self.length), self.char_lookup["z"])
        output = self.sampler(
            next=self.next,
            prompt=prompt,
            cache=cache,
            stop_token_ids=[self.char_lookup["t"]],
        )
        self.assertEqual(self.join_as_string(output), ["sequentzzzzz"])
