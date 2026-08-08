"""Tests for the per-layer sliding-window pattern in the config writer."""

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.vllm.registry import _sliding_window_per_layer


class SlidingWindowPerLayerTest(TestCase):
    def test_disabled_or_incomplete_returns_none(self):
        self.assertIsNone(
            _sliding_window_per_layer("GemmaBackbone", 4, 256, False)
        )
        self.assertIsNone(
            _sliding_window_per_layer("GemmaBackbone", 4, None, True)
        )
        self.assertIsNone(
            _sliding_window_per_layer("GemmaBackbone", None, 256, True)
        )

    def test_gemma_windows_even_layers(self):
        # gemma_backbone.py: `use_sliding_window_attention and (i % 2 == 0)`
        self.assertEqual(
            _sliding_window_per_layer("GemmaBackbone", 4, 256, True),
            [256, None, 256, None],
        )

    def test_gemma3_runs_five_local_one_global(self):
        # gemma3_backbone.py: `use_sliding_window_attention and (i % 6 < 5)`
        windows = _sliding_window_per_layer("Gemma3Backbone", 12, 512, True)
        self.assertEqual(windows[:6], [512] * 5 + [None])
        self.assertEqual(windows[6:], [512] * 5 + [None])

    def test_other_families_window_uniformly(self):
        self.assertEqual(
            _sliding_window_per_layer("QwenBackbone", 3, 128, True),
            [128, 128, 128],
        )
