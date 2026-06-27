"""CPU unit tests for the thread-local vLLM context (no TPU/keras needed)."""

from keras_hub.src.vllm import context as vllm_context


def teardown_function(_):
    vllm_context.clear_vllm_context()


def test_inactive_by_default():
    vllm_context.clear_vllm_context()
    assert vllm_context.get_vllm_context() is None


def test_set_get_roundtrip():
    vllm_context.set_vllm_context(
        block_tables="BT",
        slot_mapping="SM",
        attention_metadata="META",
        paged_attention_func="FUNC",
        mesh="MESH",
    )
    ctx = vllm_context.get_vllm_context()
    assert ctx is not None
    assert ctx.block_tables == "BT"
    assert ctx.slot_mapping == "SM"
    assert ctx.attention_metadata == "META"
    assert ctx.paged_attention_func == "FUNC"
    assert ctx.mesh == "MESH"
    assert ctx.active is True


def test_clear_resets_everything():
    vllm_context.set_vllm_context("BT", "SM", "META", "FUNC", "MESH")
    vllm_context.clear_vllm_context()
    assert vllm_context.get_vllm_context() is None
    # The singleton's fields are reset too.
    from keras_hub.src.vllm.context import _vllm_context

    assert _vllm_context.mesh is None
    assert _vllm_context.paged_attention_func is None
