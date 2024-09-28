import numpy as np
from keras import ops

from keras_hub.src.samplers.top_p_sampler import TopPSampler
from keras_hub.src.tests.test_case import TestCase


class TopPSamplerTest(TestCase):
    def setUp(self):
        super().setUp()
        # Use a simple alphabet of lowercase characters to [0, 26).
        self.int_lookup = {i: chr(i + ord("a")) for i in range(26)}
        self.char_lookup = {v: k for k, v in self.int_lookup.items()}
        self.batch_size = 1
        self.length = 12
        self.vocab_size = len(self.int_lookup)

        def next(prompt, cache, index):
            # Dummy hidden states.
            hidden_states = ops.ones([self.batch_size, 5])
            # Return a distribution favoring the next char in cache.
            logits = ops.one_hot(cache[:, index], self.vocab_size) * 1e9
            return logits, hidden_states, cache

        self.next = next
        self.sampler = TopPSampler(p=0.1)

    def join_as_string(self, x):
        x = ops.convert_to_numpy(x)
        return ["".join([self.int_lookup[i] for i in s]) for s in x]

    def test_stateless_call(self):
        def next(prompt, cache, index):
            # Dummy hidden states.
            hidden_states = ops.ones([self.batch_size, 5])
            # Return a distribution favoring the first token in the vocab.
            logits = (
                ops.one_hot(
                    ops.zeros(self.batch_size, dtype="int32"),
                    self.vocab_size,
                )
                * 1e9
            )
            return logits, hidden_states, cache

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

    def test_only_sample_from_top_k_tokens(self):
        def next(prompt, cache, index):
            # Dummy hidden states.
            hidden_states = ops.ones([self.batch_size, 5])
            # Return a distribution where each id is progressively less likely.
            logits = ops.arange(self.vocab_size, 0, -1, dtype="float32")
            logits = ops.repeat(logits[None, :], self.batch_size, axis=0)
            return logits, hidden_states, cache

        prompt = ops.full((self.batch_size, self.length), self.char_lookup["z"])
        output = TopPSampler(p=1, k=5)(
            next=next,
            prompt=prompt,
            index=5,
        )
        generated_str = self.join_as_string(output[:, 5:])[0]
        token_set = set(generated_str)
        self.assertContainsSubset(token_set, set("abcde"))

    def test_outputs_in_top_p(self):
        def next(prompt, cache, index):
            # Dummy hidden states.
            hidden_states = ops.ones([self.batch_size, 5])
            logits = np.zeros((self.batch_size, self.vocab_size))
            logits[:, :3] = 1.0
            return ops.array(logits), hidden_states, cache

        prompt = ops.full((self.batch_size, self.length), self.char_lookup["z"])
        output = TopPSampler(p=(2.0 / self.vocab_size))(
            next=next,
            prompt=prompt,
        )
        output_ids = set(ops.convert_to_numpy(output[0]))
        self.assertContainsSubset(output_ids, range(3))

    def test_temperature(self):
        def next(prompt, cache, index):
            # Dummy hidden states.
            hidden_states = ops.ones([self.batch_size, 5])
            logits = ops.arange(self.vocab_size, 0, -1, dtype="float32")
            logits = ops.reshape(logits[None, :], (self.batch_size, -1))
            return ops.array(logits), hidden_states, cache

        prompt = ops.full((self.batch_size, self.length), self.char_lookup["z"])

        output = TopPSampler(p=0.5, temperature=1e-9)(
            next=next,
            prompt=prompt,
        )
        self.assertAllEqual(output, ops.zeros_like(output))
