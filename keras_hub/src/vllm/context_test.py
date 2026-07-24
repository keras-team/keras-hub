"""CPU unit tests for the thread-local vLLM context (no TPU needed)."""

import threading

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.vllm import context as vllm_context


class VllmContextTest(TestCase):
    def tearDown(self):
        vllm_context.clear_vllm_context()
        super().tearDown()

    def test_inactive_by_default(self):
        vllm_context.clear_vllm_context()
        self.assertIsNone(vllm_context.get_vllm_context())

    def test_set_get_roundtrip(self):
        vllm_context.set_vllm_context(
            paged_attention_func="FUNC",
            positions="POS",
            kv_caches=["C0"],
        )
        ctx = vllm_context.get_vllm_context()
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx.paged_attention_func, "FUNC")
        self.assertEqual(ctx.positions, "POS")
        self.assertEqual(ctx.kv_caches, ["C0"])
        self.assertTrue(ctx.active)

    def test_clear_resets_everything(self):
        vllm_context.set_vllm_context(
            paged_attention_func="FUNC", positions="POS", kv_caches=["C0"]
        )
        vllm_context.clear_vllm_context()
        self.assertIsNone(vllm_context.get_vllm_context())
        # Every field named in _INACTIVE_STATE is back at its inactive
        # value, so nothing can survive a "clear".
        for name, value in vllm_context._INACTIVE_STATE.items():
            self.assertEqual(
                getattr(vllm_context._vllm_context, name), value, name
            )

    def test_kv_caches_copied_and_lifecycle_fields_initialized(self):
        caches = ["C0", "C1"]
        vllm_context.set_vllm_context(kv_caches=caches, positions="POS")
        ctx = vllm_context.get_vllm_context()
        # Layer counter starts fresh; updated caches seed from the input.
        self.assertEqual(ctx.layer_index, 0)
        self.assertEqual(ctx.positions, "POS")
        self.assertEqual(ctx.updated_kv_caches, caches)
        # But they are independent copies: mutating the context must not leak
        # back into the caller's list (and vice versa).
        self.assertIsNot(ctx.kv_caches, caches)
        self.assertIsNot(ctx.updated_kv_caches, ctx.kv_caches)
        ctx.updated_kv_caches[0] = "NEW"
        self.assertEqual(caches, ["C0", "C1"])

    def test_reset_on_new_forward_step(self):
        # A second forward step must reset the per-step layer counter, even if
        # the previous step was never explicitly cleared.
        vllm_context.set_vllm_context(kv_caches=["C0"])
        vllm_context.get_vllm_context().layer_index = 5
        vllm_context.set_vllm_context(kv_caches=["C0"])
        self.assertEqual(vllm_context.get_vllm_context().layer_index, 0)

    def test_scope_clears_on_normal_exit(self):
        with vllm_context.vllm_context_scope(paged_attention_func="KERNEL"):
            self.assertIsNotNone(vllm_context.get_vllm_context())
        self.assertIsNone(vllm_context.get_vllm_context())

    def test_scope_clears_on_exception(self):
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with vllm_context.vllm_context_scope(paged_attention_func="KERNEL"):
                self.assertIsNotNone(vllm_context.get_vllm_context())
                raise RuntimeError("boom")
        self.assertIsNone(vllm_context.get_vllm_context())

    def test_nested_scopes_restore_outer(self):
        # The inner scope's exit must hand back the outer scope's state, not
        # wipe it — otherwise the rest of the outer forward would silently
        # fall back to the dense path.
        with vllm_context.vllm_context_scope(
            paged_attention_func="OUTER", kv_caches=["O0"]
        ):
            with vllm_context.vllm_context_scope(paged_attention_func="INNER"):
                ctx = vllm_context.get_vllm_context()
                self.assertEqual(ctx.paged_attention_func, "INNER")
                self.assertIsNone(ctx.kv_caches)
            ctx = vllm_context.get_vllm_context()
            self.assertIsNotNone(ctx)
            self.assertEqual(ctx.paged_attention_func, "OUTER")
            self.assertEqual(ctx.kv_caches, ["O0"])
        self.assertIsNone(vllm_context.get_vllm_context())

    def test_context_is_thread_local(self):
        # A context set on the main thread must be invisible to another thread,
        # so concurrent requests can never see each other's serving state.
        vllm_context.set_vllm_context(paged_attention_func="FUNC")
        seen = {}

        def worker():
            seen["ctx"] = vllm_context.get_vllm_context()

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        self.assertIsNone(seen["ctx"])
        # The main thread still has its active context.
        self.assertIsNotNone(vllm_context.get_vllm_context())
