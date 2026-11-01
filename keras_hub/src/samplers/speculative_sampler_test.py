import keras
import numpy as np
from keras import ops

import keras_hub
from keras_hub.src.samplers.speculative_sampler import SpeculativeSampler
from keras_hub.src.tests.test_case import TestCase


class SpeculativeSamplerTest(TestCase):
    def setUp(self):
        super().setUp()
        # Use a simple alphabet of lowercase characters to [0, 26).
        self.int_lookup = {i: chr(i + ord("a")) for i in range(26)}
        self.char_lookup = {v: k for k, v in self.int_lookup.items()}
        self.batch_size = 1
        self.length = 12
        self.vocab_size = len(self.int_lookup)

    def _next_fn(self, target_chars):
        """Create a next function that outputs specific characters."""
        target_ids = ops.array([self.char_lookup[c] for c in target_chars])
        max_idx = len(target_chars) - 1

        def next_fn(prompt, cache, index):
            hidden_states = ops.ones([self.batch_size, 5])
            clamped_idx = ops.minimum(ops.cast(index, "int32"), max_idx)
            target_id = ops.take(target_ids, clamped_idx)
            logits = (
                ops.one_hot(
                    ops.broadcast_to(target_id, (self.batch_size,)),
                    self.vocab_size,
                )
                * 1e9
            )
            return logits, hidden_states, cache

        return next_fn

    def join_as_string(self, x):
        x = ops.convert_to_numpy(x)
        return ["".join([self.int_lookup[i] for i in s]) for s in x]

    def test_stateless_call(self):
        target_chars = list("helloworldss")
        target_next = self._next_fn(target_chars)
        draft_next = self._next_fn(target_chars)

        sampler = SpeculativeSampler(
            num_speculative_tokens=3,
            temperature=1.0,
        )
        prompt = ops.full((self.batch_size, self.length), self.char_lookup["z"])
        output = sampler(
            next=target_next,
            prompt=prompt,
            index=0,
            draft_next=draft_next,
        )
        self.assertTrue(self.join_as_string(output)[0].startswith("hello"))

    def test_all_accepted(self):
        target_chars = list("abcdefghijkl")
        target_next = self._next_fn(target_chars)
        draft_next = self._next_fn(target_chars)

        sampler = SpeculativeSampler(num_speculative_tokens=3, temperature=1.0)
        prompt = ops.full((self.batch_size, self.length), self.char_lookup["z"])
        output = sampler(
            next=target_next,
            prompt=prompt,
            index=0,
            draft_next=draft_next,
        )
        self.assertEqual(self.join_as_string(output), ["abcdefghijkl"])

    def test_partial_acceptance(self):
        target_chars = list("abcdefghijkl")
        draft_chars = list("abxdefghijkl")

        target_next = self._next_fn(target_chars)
        draft_next = self._next_fn(draft_chars)

        sampler = SpeculativeSampler(num_speculative_tokens=3, temperature=1.0)
        prompt = ops.full((self.batch_size, self.length), self.char_lookup["z"])
        output = sampler(
            next=target_next,
            prompt=prompt,
            index=0,
            draft_next=draft_next,
        )
        output_str = self.join_as_string(output)
        self.assertTrue(output_str[0].startswith("ab"))

    def test_early_stopping(self):
        target_chars = list("hellozzzzzzzz")
        target_next = self._next_fn(target_chars)
        draft_next = self._next_fn(target_chars)

        sampler = SpeculativeSampler(num_speculative_tokens=3, temperature=1.0)
        prompt = ops.full((self.batch_size, self.length), self.char_lookup["z"])
        output = sampler(
            next=target_next,
            prompt=prompt,
            index=0,
            stop_token_ids=[self.char_lookup["o"]],
            draft_next=draft_next,
        )
        output_str = self.join_as_string(output)
        self.assertTrue(
            "hello" in output_str[0] or output_str[0].startswith("hell")
        )

    def test_requires_draft_next(self):
        sampler = SpeculativeSampler(num_speculative_tokens=3)
        prompt = ops.full((self.batch_size, self.length), 0)

        def dummy_next(prompt, cache, index):
            return ops.zeros((1, self.vocab_size)), ops.zeros((1, 5)), cache

        with self.assertRaises(ValueError):
            sampler(next=dummy_next, prompt=prompt, index=0)

    def test_serialization(self):
        from keras_hub.src.samplers.top_p_sampler import TopPSampler

        base = TopPSampler(p=0.9, temperature=0.8)
        sampler = SpeculativeSampler(
            num_speculative_tokens=7,
            base_sampler=base,
            seed=42,
        )
        config = sampler.get_config()
        self.assertEqual(config["num_speculative_tokens"], 7)
        self.assertEqual(config["seed"], 42)
        self.assertIsNotNone(config["base_sampler"])

        restored = SpeculativeSampler.from_config(config)
        self.assertEqual(restored.num_speculative_tokens, 7)
        self.assertEqual(restored.seed, 42)

    def test_output_shape(self):
        target_chars = list("abcdefghijkl")
        target_next = self._next_fn(target_chars)
        draft_next = self._next_fn(target_chars)

        sampler = SpeculativeSampler(num_speculative_tokens=3)
        prompt = ops.full((self.batch_size, self.length), self.char_lookup["z"])
        output = sampler(
            next=target_next,
            prompt=prompt,
            index=0,
            draft_next=draft_next,
        )
        self.assertEqual(ops.shape(output)[0], self.batch_size)
        self.assertEqual(ops.shape(output)[1], self.length)

    def test_batched_call(self):
        batch_size = 2
        length = 8

        def batched_next(prompt, cache, index):
            hidden_states = ops.ones([batch_size, 5])
            logits = (
                ops.one_hot(
                    ops.zeros((batch_size,), dtype="int32"),
                    self.vocab_size,
                )
                * 1e9
            )
            return logits, hidden_states, cache

        sampler = SpeculativeSampler(num_speculative_tokens=2)
        prompt = ops.full((batch_size, length), self.char_lookup["z"])
        output = sampler(
            next=batched_next,
            prompt=prompt,
            index=0,
            draft_next=batched_next,
        )
        output_str = self.join_as_string(output)
        self.assertEqual(output_str, ["aaaaaaaa", "aaaaaaaa"])

    def test_ragged_batch_keeps_prompt_tokens(self):
        batch_size = 2
        length = 10
        z = self.char_lookup["z"]
        # Row 0 has a 2-token prompt. Row 1 has a 6-token prompt.
        prompt = np.full((batch_size, length), z, dtype="int32")
        prompt[1, :6] = [self.char_lookup[c] for c in "qrstuv"]
        mask = np.zeros((batch_size, length), dtype=bool)
        mask[0, :2] = True
        mask[1, :6] = True

        def next_fn(prompt, cache, index):
            logits = (
                ops.one_hot(ops.zeros((batch_size,), dtype="int32"), 26) * 1e9
            )
            return logits, ops.ones([batch_size, 5]), cache

        sampler = SpeculativeSampler(num_speculative_tokens=3)
        output = ops.convert_to_numpy(
            sampler(
                next=next_fn,
                prompt=ops.array(prompt),
                index=2,
                mask=ops.array(mask),
                draft_next=next_fn,
            )
        )
        self.assertAllEqual(output[mask], prompt[mask])
        self.assertAllEqual(output[~mask], np.zeros(int((~mask).sum())))

    def test_ragged_batch_counts_forced_prompt_positions_as_matched(self):
        if keras.config.backend() == "jax":
            self.skipTest("JAX traces the sampler loop once.")
        batch_size = 2
        length = 12
        z = self.char_lookup["z"]
        # Row 0 has a 2-token prompt. Row 1 has an 8-token prompt.
        prompt = np.full((batch_size, length), z, dtype="int32")
        prompt[1, :8] = [self.char_lookup[c] for c in "qrstuvwx"]
        mask = np.zeros((batch_size, length), dtype=bool)
        mask[0, :2] = True
        mask[1, :8] = True
        num_draft_calls = [0]

        def next_fn(prompt, cache, index):
            logits = (
                ops.one_hot(ops.zeros((batch_size,), dtype="int32"), 26) * 1e9
            )
            return logits, ops.ones([batch_size, 5]), cache

        def counting_draft_next(prompt, cache, index):
            num_draft_calls[0] += 1
            return next_fn(prompt, cache, index)

        sampler = SpeculativeSampler(num_speculative_tokens=3)
        output = ops.convert_to_numpy(
            sampler(
                next=next_fn,
                prompt=ops.array(prompt),
                index=2,
                mask=ops.array(mask),
                draft_next=counting_draft_next,
            )
        )
        self.assertAllEqual(output[mask], prompt[mask])
        # The draft and the target agree on every free position. Without
        # forced positions, the prompt of row 1 slows each cycle to one token.
        num_cycles = num_draft_calls[0] // 3
        self.assertLessEqual(num_cycles, 3)

    def test_one_shot_block_draft_next(self):
        # Some draft models (e.g. a bidirectional block-diffusion drafter)
        # produce every candidate from a single forward pass rather than
        # autoregressively. `draft_next` is still called once per candidate
        # by this sampler, so such a model can compute the whole block on
        # its first call within a cycle and serve cached slices on the
        # rest of that cycle's calls (a plain Python-level memo, safe
        # because a cycle's `for i in range(k)` calls happen within one
        # `body()` invocation) — this is the pattern
        # `MuseGlimmerCausalLM.generate_step` uses for its DFlash
        # assistant. Verify correctness of that pattern here with a plain
        # (non-model-specific) closure; per-cycle call-count savings are
        # only observable once `generate_step` runs inside a compiled
        # `generate_function` trace, not via a bare `sampler(...)` call
        # like this test makes, so this test checks output correctness
        # only.
        target_chars = list("abcdefghijkl")
        draft_chars = list("abcdefghijkl")
        target_next = self._next_fn(target_chars)
        block_logits = ops.one_hot(
            ops.array([self.char_lookup[c] for c in draft_chars[:3]]),
            self.vocab_size,
        )

        def one_shot_draft_next(prompt, draft_cache, index):
            anchor_pos = draft_cache
            position = ops.minimum(ops.maximum(index - anchor_pos - 1, 0), 2)
            logits = ops.take(block_logits, position, axis=0) * 1e9
            hidden_states = ops.ones([self.batch_size, 5])
            return logits, hidden_states, draft_cache

        sampler = SpeculativeSampler(num_speculative_tokens=3, temperature=1.0)
        prompt = ops.full((self.batch_size, self.length), self.char_lookup["z"])
        output = sampler(
            next=target_next,
            prompt=prompt,
            index=0,
            draft_next=one_shot_draft_next,
            draft_cache=-1,
        )
        self.assertEqual(self.join_as_string(output), ["abcdefghijkl"])

    def test_generic_draft_cache_continuation(self):
        # The 3-tuple `(draft_state, target_cache, anchor_pos)` convention
        # for refreshing `draft_cache` after each cycle is generic — it
        # must not require the model passed via `model=` to be any
        # particular class.
        target_chars = list("abcdefghijkl")
        target_next = self._next_fn(target_chars)
        draft_next = self._next_fn(target_chars)

        sampler = SpeculativeSampler(num_speculative_tokens=3, temperature=1.0)
        prompt = ops.full((self.batch_size, self.length), self.char_lookup["z"])
        initial_hidden = ops.zeros((self.batch_size, 1, 4))
        initial_target_cache = ops.zeros((self.batch_size, 1))
        output = sampler(
            next=target_next,
            prompt=prompt,
            cache=initial_target_cache,
            index=0,
            draft_next=draft_next,
            draft_cache=(initial_hidden, initial_target_cache, 0),
            model=object(),
        )
        self.assertEqual(self.join_as_string(output), ["abcdefghijkl"])

    def test_four_tuple_draft_cache_continuation(self):
        # The 4-tuple `(draft_state, target_cache, anchor_pos, drafter_cache)`
        # convention. After each cycle, the sampler replaces `target_cache`
        # with the verified cache, moves `anchor_pos` to the last verified
        # position, and replaces `draft_state` with the verify hidden state at
        # `min_accepted`. The sampler never changes `drafter_cache`.
        num_speculative_tokens = 3
        target_chars = list("abcdefghijkl")
        target_next = self._next_fn(target_chars)
        draft_next_fn = self._next_fn(target_chars)
        records = []

        def verify_next(prompt, cache, index, k):
            logits = ops.stack(
                [
                    target_next(prompt, cache, index + i)[0]
                    for i in range(k + 1)
                ],
                axis=1,
            )
            # Row `j` of the hidden states is `j`, so `min_accepted` is
            # visible in the next `draft_state`.
            hidden = ops.broadcast_to(
                ops.reshape(ops.cast(ops.arange(k + 1), "float32"), (1, -1, 1)),
                (self.batch_size, k + 1, 5),
            )
            return logits, hidden, cache + 1

        def draft_next(prompt, draft_cache, index):
            logits, _, _ = draft_next_fn(prompt, None, index)
            if keras.config.backend() == "torch":
                records.append((int(ops.convert_to_numpy(index)), draft_cache))
            return logits, ops.ones((self.batch_size, 5)), draft_cache

        sampler = SpeculativeSampler(
            num_speculative_tokens=num_speculative_tokens, temperature=1.0
        )
        prompt = ops.full((self.batch_size, self.length), self.char_lookup["z"])
        output = sampler(
            next=target_next,
            prompt=prompt,
            cache=ops.zeros((self.batch_size, 1)),
            index=0,
            draft_next=draft_next,
            draft_cache=(
                ops.zeros((self.batch_size, 1, 5)),
                ops.zeros((self.batch_size, 1)),
                0,
                ops.full((self.batch_size, 1), 7.0),
            ),
            verify_next=verify_next,
            model=object(),
        )
        self.assertEqual(self.join_as_string(output), ["abcdefghijkl"])

        # The sampler loop is traced on JAX and TensorFlow, so only torch
        # runs each cycle in Python and gives concrete values.
        if keras.config.backend() != "torch":
            return
        cycle_starts = records[::num_speculative_tokens]
        self.assertGreaterEqual(len(cycle_starts), 3)
        for cycle, (index, draft_cache) in enumerate(cycle_starts):
            draft_state, target_cache, anchor_pos, drafter_cache = draft_cache
            self.assertAllClose(drafter_cache, np.full((1, 1), 7.0))
            if cycle == 0:
                continue
            # Every draft token is right, so `min_accepted` is `k`.
            self.assertAllClose(
                draft_state, np.full((1, 1, 5), float(num_speculative_tokens))
            )
            self.assertAllClose(target_cache, np.full((1, 1), float(cycle)))
            self.assertEqual(int(ops.convert_to_numpy(anchor_pos)), index - 1)

    def test_short_circuit_rejection(self):
        # If token 2 is wrong, tokens 3+ should be rejected even if correct.
        target_chars = list("abcdefghijkl")
        draft_chars = list("abxdefghijkl")

        target_next = self._next_fn(target_chars)
        draft_next = self._next_fn(draft_chars)

        sampler = SpeculativeSampler(num_speculative_tokens=5, temperature=1.0)
        prompt = ops.full((self.batch_size, self.length), self.char_lookup["z"])
        output = sampler(
            next=target_next,
            prompt=prompt,
            index=0,
            draft_next=draft_next,
        )
        output_str = self.join_as_string(output)
        self.assertTrue(output_str[0].startswith("ab"))
        self.assertEqual(output_str[0][2], "c")

    def test_top_k_one_matches_target_sampler(self):
        # With `k=1`, the target distribution is a point mass, so a wrong
        # draft must never change the output.
        target_chars = list("abcdefghijkl")
        draft_chars = list("abxdefzhijkl")

        def soft_next(chars):
            fn = self._next_fn(chars)

            def next_fn(prompt, cache, index):
                logits, hidden, cache = fn(prompt, cache, index)
                return logits * 1e-9 + 2.0, hidden, cache

            return next_fn

        # Logits are 3.0 at the chosen character and 2.0 elsewhere.
        sampler = SpeculativeSampler(
            num_speculative_tokens=3,
            base_sampler=keras_hub.samplers.TopKSampler(k=1, seed=1),
            seed=2,
        )
        prompt = ops.full((self.batch_size, self.length), self.char_lookup["z"])
        for _ in range(5):
            output = sampler(
                next=soft_next(target_chars),
                prompt=prompt,
                index=0,
                draft_next=soft_next(draft_chars),
            )
            self.assertEqual(self.join_as_string(output), ["abcdefghijkl"])

    def _sample_first_token(self, base_sampler, draft_token, p):
        batch, k, length = 2000, 2, 4
        target_logits = np.log(np.asarray(p, "float32"))
        vocab = len(p)
        p_logits = ops.convert_to_tensor(
            np.broadcast_to(target_logits, (batch, k + 1, vocab)).copy()
        )
        q_logits = ops.convert_to_tensor(
            np.broadcast_to(np.eye(vocab)[draft_token] * 10.0, (batch, vocab))
            .astype("float32")
            .copy()
        )

        def draft_next(prompt, draft_cache, index):
            return q_logits, ops.zeros((batch, 1)), draft_cache

        def verify_next(prompt, cache, index, num_tokens):
            return p_logits, cache

        sampler = SpeculativeSampler(
            num_speculative_tokens=k, base_sampler=base_sampler, seed=3
        )
        output = sampler(
            next=None,
            prompt=ops.zeros((batch, length), "int32"),
            index=1,
            draft_next=draft_next,
            verify_next=verify_next,
        )
        tokens = ops.convert_to_numpy(output)[:, 1]
        return np.bincount(tokens, minlength=vocab) / batch

    def test_stochastic_matches_target_distribution(self):
        p = [0.1, 0.2, 0.3, 0.4]
        freq = self._sample_first_token(
            keras_hub.samplers.RandomSampler(seed=1), draft_token=1, p=p
        )
        self.assertLess(0.5 * np.abs(freq - np.array(p)).sum(), 0.05)

    def test_stochastic_applies_top_k_to_target(self):
        p = [0.1, 0.2, 0.3, 0.4]
        freq = self._sample_first_token(
            keras_hub.samplers.TopKSampler(k=2, seed=1), draft_token=1, p=p
        )
        expected = np.array([0.0, 0.0, 3 / 7, 4 / 7])
        self.assertLess(0.5 * np.abs(freq - expected).sum(), 0.05)

    def test_stochastic_bonus_token_stays_in_top_k_with_large_vocab(self):
        # `uniform + 1e-7` rounds to 1.0 in float32 for the largest uniform
        # values. A Gumbel-max draw then lets a masked token win.
        batch, k, length, vocab, top = 4, 3, 6, 200_000, 5
        logits = np.zeros((batch, k + 1, vocab), "float32")
        logits[..., :top] = 1.0
        p_logits = ops.convert_to_tensor(logits)
        q_logits = ops.zeros((batch, vocab))

        def draft_next(prompt, draft_cache, index):
            return q_logits, ops.zeros((batch, 1)), draft_cache

        def verify_next(prompt, cache, index, num_tokens):
            return p_logits, cache

        sampler = SpeculativeSampler(
            num_speculative_tokens=k,
            base_sampler=keras_hub.samplers.TopKSampler(k=top, seed=1),
            seed=2,
        )
        for _ in range(30):
            output = sampler(
                next=None,
                prompt=ops.zeros((batch, length), "int32"),
                index=1,
                draft_next=draft_next,
                verify_next=verify_next,
            )
            self.assertTrue(np.all(ops.convert_to_numpy(output) < top))

    def test_stop_token_in_uncommitted_draft_does_not_end_the_loop(self):
        # Row 0 accepts the drafts [1, 2, STOP] in the first cycle. Row 1
        # rejects its first draft and commits STOP. The loop advances by one
        # position. The stop token of row 0 is a draft at position 3 that
        # the next cycles verify again. It must not end the loop. Cycle 2
        # rejects the draft of row 0 at position 2, so the old STOP no longer
        # follows its prefix. Cycle 3 replaces it.
        length, vocab, stop = 6, 5, 4
        drafts = ops.convert_to_tensor([0, 1, 2, stop, 3, 3], "int32")
        # `targets[cycle][row]` holds the target token of each of the
        # k + 1 verified positions that start at index `cycle`.
        targets = np.full((length, 2, 4), 3, "int32")
        targets[1] = [[1, 2, stop, 0], [stop, 2, stop, 3]]
        targets[2] = [[0, 3, 3, 3], [2, stop, 3, 3]]
        targets[3] = [[3, 3, 3, 3], [stop, 3, 3, 3]]
        targets = ops.convert_to_tensor(targets)

        def draft_next(prompt, draft_cache, index):
            token = ops.take(drafts, index)
            logits = ops.one_hot(ops.broadcast_to(token, (2,)), vocab) * 1e9
            return logits, ops.zeros((2, 1)), draft_cache

        def verify_next(prompt, cache, index, num_tokens):
            logits = ops.one_hot(ops.take(targets, index, axis=0), vocab)
            return logits * 1e9, cache

        sampler = SpeculativeSampler(
            num_speculative_tokens=3,
            base_sampler=keras_hub.samplers.RandomSampler(seed=1),
        )
        mask = np.zeros((2, length), bool)
        mask[:, 0] = True
        output = ops.convert_to_numpy(
            sampler(
                next=None,
                prompt=ops.zeros((2, length), "int32"),
                index=1,
                mask=ops.convert_to_tensor(mask),
                stop_token_ids=[stop],
                draft_next=draft_next,
                verify_next=verify_next,
            )
        )
        self.assertEqual(output[1, 1], stop)
        self.assertAllEqual(output[0], [0, 1, 0, 3, 3, 3])

    def test_top_k_keeps_exactly_k_tokens_with_ties(self):
        probs = ops.convert_to_tensor(
            np.array([[0.3, 0.2, 0.2, 0.2, 0.05, 0.05]], "float32")
        )
        sampler = SpeculativeSampler(
            num_speculative_tokens=1,
            base_sampler=keras_hub.samplers.TopKSampler(k=2),
        )
        filtered = ops.convert_to_numpy(sampler._filter_probabilities(probs))
        # The lowest index of the tied tokens stays, like the plain sampler.
        self.assertAllClose(filtered, [[0.6, 0.4, 0.0, 0.0, 0.0, 0.0]])

    def test_top_p_keeps_the_nucleus_with_ties(self):
        probs = ops.convert_to_tensor(
            np.array([[0.3, 0.2, 0.2, 0.2, 0.05, 0.05]], "float32")
        )
        sampler = SpeculativeSampler(
            num_speculative_tokens=1,
            base_sampler=keras_hub.samplers.TopPSampler(p=0.6),
        )
        filtered = ops.convert_to_numpy(sampler._filter_probabilities(probs))
        # 0.3 + 0.2 = 0.5 is below 0.6. The next token crosses 0.6.
        expected = [[0.3 / 0.7, 0.2 / 0.7, 0.2 / 0.7, 0.0, 0.0, 0.0]]
        self.assertAllClose(filtered, expected)
