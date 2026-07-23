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
            block_tables="BT",
            slot_mapping="SM",
            attention_metadata="META",
            paged_attention_func="FUNC",
            mesh="MESH",
        )
        ctx = vllm_context.get_vllm_context()
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx.block_tables, "BT")
        self.assertEqual(ctx.slot_mapping, "SM")
        self.assertEqual(ctx.attention_metadata, "META")
        self.assertEqual(ctx.paged_attention_func, "FUNC")
        self.assertEqual(ctx.mesh, "MESH")
        self.assertTrue(ctx.active)

    def test_clear_resets_everything(self):
        vllm_context.set_vllm_context("BT", "SM", "META", "FUNC", "MESH")
        vllm_context.clear_vllm_context()
        self.assertIsNone(vllm_context.get_vllm_context())
        # The singleton's fields are reset too.
        self.assertIsNone(vllm_context._vllm_context.mesh)
        self.assertIsNone(vllm_context._vllm_context.paged_attention_func)
        self.assertIsNone(vllm_context._vllm_context.kv_caches)
        self.assertEqual(vllm_context._vllm_context.layer_index, 0)

    def test_kv_caches_copied_and_lifecycle_fields_initialized(self):
        caches = ["C0", "C1"]
        vllm_context.set_vllm_context(
            None, None, kv_caches=caches, positions="POS"
        )
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
        vllm_context.set_vllm_context(None, None, kv_caches=["C0"])
        vllm_context.get_vllm_context().layer_index = 5
        vllm_context.set_vllm_context(None, None, kv_caches=["C0"])
        self.assertEqual(vllm_context.get_vllm_context().layer_index, 0)

    def test_scope_clears_on_exception(self):
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with vllm_context.vllm_context_scope(
                None, None, paged_attention_func="KERNEL"
            ):
                self.assertIsNotNone(vllm_context.get_vllm_context())
                raise RuntimeError("boom")
        self.assertIsNone(vllm_context.get_vllm_context())

    def test_context_is_thread_local(self):
        # A context set on the main thread must be invisible to another thread,
        # so concurrent requests can never see each other's serving state.
        vllm_context.set_vllm_context("BT", "SM", "META", "FUNC", "MESH")
        seen = {}

        def worker():
            seen["ctx"] = vllm_context.get_vllm_context()

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        self.assertIsNone(seen["ctx"])
        # The main thread still has its active context.
        self.assertIsNotNone(vllm_context.get_vllm_context())
