import numpy as np
from keras import ops

from keras_hub.src.samplers.entropy_bound_sampler import EntropyBoundSampler
from keras_hub.src.tests.test_case import TestCase


class EntropyBoundSamplerTest(TestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 2
        self.canvas_length = 6
        self.vocab_size = 10

        self.sampler = EntropyBoundSampler(
            entropy_bound=0.1,
            confidence_threshold=0.005,
            stability_threshold=1,
            seed=42,
        )

    def _make_peaked_logits(self, token_ids):
        """Return logits strongly peaked at `token_ids` (B, L, V)."""
        one_hot = ops.one_hot(
            ops.array(token_ids, dtype="int32"), self.vocab_size
        )
        return ops.cast(one_hot, "float32") * 1e9

    def _make_uniform_logits(self):
        """Return uniform logits (B, L, V) — maximum entropy."""
        return ops.zeros(
            (self.batch_size, self.canvas_length, self.vocab_size),
            dtype="float32",
        )

    def _sample_step(self, sampler, canvas, logits, step, state=None):
        if state is None:
            state = sampler.initialize_state(canvas)
        return sampler._sample_step(canvas, logits, step, state)

    def test_infers_vocabulary_size_from_logits(self):
        sampler = EntropyBoundSampler(seed=42)
        canvas = ops.zeros((self.batch_size, self.canvas_length), dtype="int32")
        logits = self._make_uniform_logits()

        new_canvas, stop, _, _ = self._sample_step(
            sampler, canvas, logits, step=0
        )

        self.assertEqual(new_canvas.shape, canvas.shape)
        self.assertEqual(stop.shape, (self.batch_size,))

    def test_confident_tokens_are_committed(self):
        token_ids = np.ones(
            (self.batch_size, self.canvas_length), dtype="int32"
        )
        canvas = ops.array(token_ids)
        logits = self._make_peaked_logits(token_ids)

        new_canvas, _, _, _ = self._sample_step(
            self.sampler, canvas, logits, step=0
        )

        expected = ops.argmax(logits, axis=-1)
        self.assertAllEqual(new_canvas, expected)

    def test_output_shape(self):
        canvas = ops.zeros((self.batch_size, self.canvas_length), dtype="int32")
        logits = self._make_uniform_logits()

        new_canvas, stop, _, _ = self._sample_step(
            self.sampler, canvas, logits, step=0
        )

        self.assertEqual(
            new_canvas.shape, (self.batch_size, self.canvas_length)
        )
        stop_np = ops.convert_to_numpy(stop)
        self.assertEqual(stop_np.shape, (self.batch_size,))

    def test_uncommitted_positions_are_renoised(self):
        sampler = EntropyBoundSampler(
            entropy_bound=0.0,
            seed=0,
        )
        canvas = ops.zeros((self.batch_size, self.canvas_length), dtype="int32")
        logits = self._make_uniform_logits()

        new_canvas, _, _, _ = self._sample_step(sampler, canvas, logits, step=0)
        new_canvas_np = ops.convert_to_numpy(new_canvas)

        # All tokens must be within [0, vocab_size).
        self.assertTrue(np.all(new_canvas_np >= 0))
        self.assertTrue(np.all(new_canvas_np < self.vocab_size))

    def test_no_stop_on_step_zero(self):
        token_ids = np.zeros(
            (self.batch_size, self.canvas_length), dtype="int32"
        )
        canvas = ops.array(token_ids)
        logits = self._make_peaked_logits(token_ids)

        _, stop, _, _ = self._sample_step(self.sampler, canvas, logits, step=0)

        self.assertFalse(ops.convert_to_numpy(ops.any(stop)))

    def test_stop_when_confident_and_stable(self):
        sampler = EntropyBoundSampler(
            entropy_bound=1.0,
            confidence_threshold=1.0,
            stability_threshold=1,
            seed=0,
        )
        token_ids = np.zeros(
            (self.batch_size, self.canvas_length), dtype="int32"
        )
        canvas = ops.array(token_ids)
        logits = self._make_peaked_logits(token_ids)

        _, stop0, _, state = self._sample_step(sampler, canvas, logits, step=0)
        _, stop1, _, _ = self._sample_step(
            sampler, canvas, logits, step=1, state=state
        )

        self.assertFalse(ops.convert_to_numpy(ops.any(stop0)))
        self.assertTrue(ops.convert_to_numpy(ops.all(stop1)))

    def test_explicit_tensor_state(self):
        sampler = EntropyBoundSampler(
            entropy_bound=1.0,
            confidence_threshold=1.0,
            stability_threshold=1,
            seed=0,
        )
        token_ids = np.zeros(
            (self.batch_size, self.canvas_length), dtype="int32"
        )
        canvas = ops.array(token_ids)
        logits = self._make_peaked_logits(token_ids)
        state = sampler.initialize_state(canvas)

        _, stop0, _, state = self._sample_step(
            sampler, canvas, logits, step=0, state=state
        )
        _, stop1, _, _ = self._sample_step(
            sampler, canvas, logits, step=1, state=state
        )

        self.assertFalse(ops.convert_to_numpy(ops.any(stop0)))
        self.assertTrue(ops.convert_to_numpy(ops.all(stop1)))

    def test_no_stop_when_argmax_changes(self):
        sampler = EntropyBoundSampler(
            entropy_bound=1.0,
            confidence_threshold=1.0,
            stability_threshold=1,
            seed=0,
        )
        token_ids_a = np.zeros(
            (self.batch_size, self.canvas_length), dtype="int32"
        )
        token_ids_b = np.ones(
            (self.batch_size, self.canvas_length), dtype="int32"
        )
        canvas = ops.zeros((self.batch_size, self.canvas_length), dtype="int32")
        logits_a = self._make_peaked_logits(token_ids_a)
        logits_b = self._make_peaked_logits(token_ids_b)

        _, _, _, state = self._sample_step(sampler, canvas, logits_a, step=0)
        _, stop, _, _ = self._sample_step(
            sampler, canvas, logits_b, step=1, state=state
        )

        self.assertFalse(ops.convert_to_numpy(ops.any(stop)))

    def test_get_config(self):
        config = self.sampler.get_config()

        self.assertEqual(config["entropy_bound"], 0.1)
        self.assertEqual(config["confidence_threshold"], 0.005)
        self.assertEqual(config["stability_threshold"], 1)
        self.assertEqual(config["seed"], 42)

    def test_from_config(self):
        config = self.sampler.get_config()
        restored = EntropyBoundSampler.from_config(config)

        self.assertEqual(restored.entropy_bound, self.sampler.entropy_bound)
        self.assertEqual(
            restored.confidence_threshold, self.sampler.confidence_threshold
        )
        self.assertEqual(
            restored.stability_threshold, self.sampler.stability_threshold
        )
        self.assertEqual(restored.seed, self.sampler.seed)

    def test_stability_threshold_respected(self):
        # stability_threshold=2 requires two consecutive stable steps before
        # stopping is allowed.
        sampler = EntropyBoundSampler(
            entropy_bound=1.0,
            confidence_threshold=1.0,
            stability_threshold=2,
            seed=0,
        )
        token_ids = np.zeros(
            (self.batch_size, self.canvas_length), dtype="int32"
        )
        canvas = ops.array(token_ids)
        logits = self._make_peaked_logits(token_ids)

        _, stop0, _, state = self._sample_step(sampler, canvas, logits, step=0)
        _, stop1, _, state = self._sample_step(
            sampler, canvas, logits, step=1, state=state
        )
        _, stop2, _, _ = self._sample_step(
            sampler, canvas, logits, step=2, state=state
        )

        self.assertFalse(ops.convert_to_numpy(ops.any(stop0)))
        self.assertFalse(
            ops.convert_to_numpy(ops.any(stop1))
        )  # only 1 stable step
        self.assertTrue(ops.convert_to_numpy(ops.all(stop2)))  # 2 stable steps

    def test_reproducibility(self):
        # Two samplers with the same seed must produce identical outputs.
        sampler_a = EntropyBoundSampler(
            entropy_bound=0.5,
            seed=7,
        )
        sampler_b = EntropyBoundSampler(
            entropy_bound=0.5,
            seed=7,
        )
        canvas = ops.zeros((self.batch_size, self.canvas_length), dtype="int32")
        logits = self._make_uniform_logits()

        canvas_a, _, _, _ = self._sample_step(sampler_a, canvas, logits, step=0)
        canvas_b, _, _, _ = self._sample_step(sampler_b, canvas, logits, step=0)

        self.assertAllEqual(canvas_a, canvas_b)

    def test_per_row_stop(self):
        # Row 0 keeps the same argmax across both steps; row 1's argmax
        # changes.  After step 1, only row 0 should have its stop flag set.
        sampler = EntropyBoundSampler(
            entropy_bound=1.0,
            confidence_threshold=1.0,
            stability_threshold=1,
            seed=0,
        )
        # Step 0: both rows peaked at token 0.
        token_ids_step0 = np.zeros(
            (self.batch_size, self.canvas_length), dtype="int32"
        )
        # Step 1: row 0 stays at token 0, row 1 moves to token 1.
        token_ids_step1 = np.array(
            [[0] * self.canvas_length, [1] * self.canvas_length],
            dtype="int32",
        )
        canvas = ops.zeros((self.batch_size, self.canvas_length), dtype="int32")

        _, _, _, state = self._sample_step(
            sampler,
            canvas,
            self._make_peaked_logits(token_ids_step0),
            step=0,
        )
        _, stop, _, _ = self._sample_step(
            sampler,
            canvas,
            self._make_peaked_logits(token_ids_step1),
            step=1,
            state=state,
        )

        stop_np = ops.convert_to_numpy(stop)
        self.assertTrue(stop_np[0])  # row 0: argmax stable → stops
        self.assertFalse(stop_np[1])  # row 1: argmax changed → does not stop

    def test_finished_rows_are_frozen_while_other_rows_continue(self):
        class ScriptedSampler(EntropyBoundSampler):
            def _sample_step(self, canvas, logits, step, state):
                next_canvas = canvas + 1
                next_argmax = canvas
                next_stop = ops.stack([step >= 1, step >= 3])
                return next_canvas, next_stop, next_argmax, state

        sampler = ScriptedSampler(seed=0)
        canvas = ops.zeros((2, 3), dtype="int32")

        def next(canvas, prev_logits, step):
            if prev_logits is None:
                return ops.zeros((2, 3, 2), dtype="float32")
            return prev_logits + 1

        output = sampler(next=next, canvas=canvas, max_steps=5)

        self.assertAllEqual(output[0], [1, 1, 1])
        self.assertAllEqual(output[1], [3, 3, 3])
