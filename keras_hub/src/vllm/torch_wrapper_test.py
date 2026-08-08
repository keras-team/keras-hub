"""CPU unit tests for the GPU (torch) serving wrapper.

These drive `KerasHubTorchModel` with a recording stand-in for vLLM's
`Attention` layer and a fake backbone, so no GPU, vLLM, or preset download
is required.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from keras import ops

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.vllm import torch_wrapper
from keras_hub.src.vllm.attention import vllm_paged_attention
from keras_hub.src.vllm.torch_wrapper import KerasHubPresetLoader
from keras_hub.src.vllm.torch_wrapper import KerasHubTorchModel

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

LAYERS, HEADS, KV_HEADS, HEAD_DIM = 2, 4, 2, 8


class FakeAttention(nn.Module):
    """Stand-in for vLLM's Attention: records init args, echoes the query."""

    def __init__(self, num_heads, head_size, scale=None, **kwargs):
        super().__init__()
        self.init_args = dict(
            num_heads=num_heads, head_size=head_size, scale=scale, **kwargs
        )
        self.calls = []

    def forward(self, q, k, v):
        self.calls.append((q, k, v))
        return q


def fake_vllm_config():
    hf_config = SimpleNamespace(
        keras_hub_preset="test_preset",
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        torch_dtype="float32",
    )
    model_config = SimpleNamespace(hf_config=hf_config, dtype="float32")
    return SimpleNamespace(model_config=model_config, cache_config=None)


class RoutedBackbone:
    """Fake backbone whose 'attention layers' call the real bridge."""

    def __init__(self, num_layers):
        self.num_layers = num_layers

    def __call__(self, inputs, training=False):
        token_ids = inputs["token_ids"]
        num_tokens = ops.shape(token_ids)[0]
        hidden = ops.zeros((num_tokens, 1, HEADS * HEAD_DIM))
        for _ in range(self.num_layers):
            q = ops.zeros((num_tokens, 1, HEADS, HEAD_DIM))
            kv = ops.zeros((num_tokens, 1, KV_HEADS, HEAD_DIM))
            out = vllm_paged_attention(
                q, kv, kv, HEAD_DIM**-0.5, num_kv_heads=KV_HEADS
            )
            self_check = out is not None  # route must be on-path in tests
            assert self_check
        return hidden


class KerasHubTorchModelTest(TestCase):
    def setUp(self):
        super().setUp()
        self._real_attention_cls = torch_wrapper._attention_cls
        torch_wrapper._attention_cls = lambda: FakeAttention

    def tearDown(self):
        torch_wrapper._attention_cls = self._real_attention_cls
        super().tearDown()

    def build_wrapper(self):
        return KerasHubTorchModel(fake_vllm_config(), prefix="model")

    def test_builds_one_attention_module_per_layer(self):
        wrapper = self.build_wrapper()
        self.assertLen(wrapper.layers, LAYERS)
        for i, layer in enumerate(wrapper.layers):
            self.assertEqual(layer.init_args["num_heads"], HEADS)
            self.assertEqual(layer.init_args["head_size"], HEAD_DIM)
            self.assertEqual(layer.init_args["num_kv_heads"], KV_HEADS)
            self.assertEqual(layer.init_args["scale"], HEAD_DIM**-0.5)
            # The engine binds paged KV caches by prefix; they must be
            # unique and stable.
            self.assertEqual(
                layer.init_args["prefix"], f"model.layers.{i}.attn"
            )

    def test_forward_dispatches_once_per_layer(self):
        wrapper = self.build_wrapper()
        wrapper.backbone = RoutedBackbone(LAYERS)
        hidden = wrapper.forward(
            np.zeros((3,), dtype="int32"), np.arange(3, dtype="int32")
        )
        # (num_tokens, 1, hidden) squeezed to (num_tokens, hidden).
        self.assertEqual(tuple(hidden.shape), (3, HEADS * HEAD_DIM))
        for layer in wrapper.layers:
            self.assertLen(layer.calls, 1)

    def test_forward_raises_when_a_layer_skips_dispatch(self):
        wrapper = self.build_wrapper()
        wrapper.backbone = RoutedBackbone(LAYERS - 1)
        with self.assertRaisesRegex(RuntimeError, "skipped"):
            wrapper.forward(
                np.zeros((3,), dtype="int32"), np.arange(3, dtype="int32")
            )

    def test_published_function_rejects_mismatched_geometry(self):
        wrapper = self.build_wrapper()
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            wrapper._paged_attention(
                wrapper.layers[0],
                None,
                None,
                None,
                scale=HEAD_DIM**-0.5,
                head_size=HEAD_DIM,
                num_heads=HEADS + 1,  # route disagrees with the config
                num_kv_heads=KV_HEADS,
            )

    def test_published_function_rejects_window_and_cap(self):
        # Per-layer options are constructor arguments on vLLM's Attention;
        # families that need them are not served by this wrapper yet.
        wrapper = self.build_wrapper()
        for option in (dict(sliding_window=128), dict(soft_cap=50.0)):
            with self.assertRaisesRegex(RuntimeError, "not supported"):
                wrapper._paged_attention(
                    wrapper.layers[0],
                    None,
                    None,
                    None,
                    scale=HEAD_DIM**-0.5,
                    head_size=HEAD_DIM,
                    num_heads=HEADS,
                    num_kv_heads=KV_HEADS,
                    **option,
                )

    def test_embed_and_logits_use_the_tied_embedding(self):
        wrapper = self.build_wrapper()

        class FakeEmbedding:
            def __call__(self, x, reverse=False):
                return ("reverse" if reverse else "forward", x)

        wrapper.backbone = SimpleNamespace(token_embedding=FakeEmbedding())
        self.assertEqual(wrapper.embed_input_ids("ids")[0], "forward")
        self.assertEqual(wrapper.compute_logits("hidden")[0], "reverse")

    def test_loader_fills_the_model_from_the_preset(self):
        loaded = []
        model = SimpleNamespace(load_preset=lambda: loaded.append(True))
        KerasHubPresetLoader().load_weights(model, model_config=None)
        self.assertTrue(loaded)
