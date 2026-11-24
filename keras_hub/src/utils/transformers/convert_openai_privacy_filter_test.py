import json
import os
import re
import tempfile

import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_backbone import (  # noqa: E501
    OpenAIPrivacyFilterBackbone,
)
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_token_classifier import (  # noqa: E501
    OpenAIPrivacyFilterTokenClassifier,
)
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_tokenizer import (  # noqa: E501
    O200K_SPLIT_PATTERN,
)
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_tokenizer import (  # noqa: E501
    OpenAIPrivacyFilterTokenizer,
)
from keras_hub.src.models.token_classifier import TokenClassifier
from keras_hub.src.tests.mocks.mock_bpe_vocabulary import get_bpe_fixture
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.keras_utils import running_on_gpu
from keras_hub.src.utils.transformers.convert_openai_privacy_filter import (
    convert_backbone_config,
)
from keras_hub.src.utils.transformers.convert_openai_privacy_filter import (
    convert_head,
)
from keras_hub.src.utils.transformers.convert_openai_privacy_filter import (
    convert_task_config,
)
from keras_hub.src.utils.transformers.convert_openai_privacy_filter import (
    convert_tokenizer,
)
from keras_hub.src.utils.transformers.convert_openai_privacy_filter import (
    convert_weights,
)

TINY_PRESET = "hf://yujiepan/openai-privacy-filter-tiny-random"

# The keys and values of the real `openai/privacy-filter` config.
HF_CONFIG = {
    "vocab_size": 200064,
    "num_hidden_layers": 8,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "hidden_size": 640,
    "intermediate_size": 640,
    "head_dim": 64,
    "num_local_experts": 128,
    "num_experts_per_tok": 4,
    "rms_norm_eps": 1e-5,
    "sliding_window": 128,
    "attention_dropout": 0.0,
    "attention_bias": True,
    "rope_parameters": {
        "rope_type": "yarn",
        "rope_theta": 150000.0,
        "factor": 32.0,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "original_max_position_embeddings": 4096,
        "truncate": False,
    },
    "id2label": {str(i): f"L{i}" for i in range(33)},
    "classifier_dropout": 0.0,
}


def _write_json(directory, name, data):
    with open(os.path.join(directory, name), "w", encoding="utf-8") as f:
        json.dump(data, f)


def _convert_tokenizer_json(directory, merges=None, pre_tokenizer=None, **kw):
    """Write a minimal `tokenizer.json` to `directory` and convert it."""
    vocab, fixture_merges = get_bpe_fixture()
    if pre_tokenizer is None:
        pre_tokenizer = {
            "pretokenizers": [{"pattern": {"Regex": O200K_SPLIT_PATTERN}}]
        }
    _write_json(
        directory,
        "tokenizer.json",
        {
            "model": {
                "vocab": vocab,
                "merges": fixture_merges if merges is None else merges,
            },
            "added_tokens": [
                {"content": "<|endoftext|>", "id": vocab["<|endoftext|>"]},
                {"content": "<|startoftext|>", "id": len(vocab)},
            ],
            "pre_tokenizer": pre_tokenizer,
        },
    )
    return convert_tokenizer(OpenAIPrivacyFilterTokenizer, directory, **kw)


class TestTask(TestCase):
    def test_convert_config(self):
        hf_config = HF_CONFIG
        # `intermediate_size` differs from `hidden_size`, so a swap shows.
        config = convert_backbone_config(
            {**hf_config, "intermediate_size": 1280}
        )
        self.assertEqual(
            config,
            {
                "vocabulary_size": 200064,
                "num_layers": 8,
                "num_query_heads": 14,
                "num_key_value_heads": 2,
                "hidden_dim": 640,
                "intermediate_dim": 1280,
                "head_dim": 64,
                "num_experts": 128,
                "top_k": 4,
                "rope_max_wavelength": 150000.0,
                "rope_scaling_factor": 32.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "original_max_position_embeddings": 4096,
                "truncate": False,
                "swiglu_alpha": 1.702,
                "swiglu_limit": 7.0,
                "rms_norm_eps": 1e-5,
                "sliding_window": 128,
                "attention_dropout": 0.0,
            },
        )

        # Test overriding rope parameters
        hf_config_override = {
            **hf_config,
            "rope_parameters": {
                **hf_config["rope_parameters"],
                "beta_fast": 16.0,
            },
        }
        config_override = convert_backbone_config(hf_config_override)
        self.assertEqual(config_override["beta_fast"], 16.0)

        self.assertEqual(
            convert_task_config(hf_config),
            {"num_classes": 33, "dropout": 0.0},
        )
        # Values the backbone hardcodes must be validated, not ignored.
        for key, value in [
            ("rope_type", "linear"),
        ]:
            bad = {
                **hf_config,
                "rope_parameters": {**hf_config["rope_parameters"], key: value},
            }
            with self.assertRaises(ValueError):
                convert_backbone_config(bad)
        with self.assertRaises(ValueError):
            convert_backbone_config({**hf_config, "attention_bias": False})

    def test_load_weights_false_skips_head(self):
        # A config-only directory has no `model.safetensors`, so reading the
        # head here would raise `FileNotFoundError`.
        preset_dir = self.get_temp_dir()
        _write_json(
            preset_dir,
            "config.json",
            {
                **HF_CONFIG,
                "architectures": ["OpenAIPrivacyFilterForTokenClassification"],
                "model_type": "openai_privacy_filter",
                "vocab_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "hidden_size": 8,
                "intermediate_size": 16,
                "head_dim": 4,
                "num_local_experts": 2,
                "num_experts_per_tok": 1,
                "sliding_window": 4,
                "id2label": {"0": "O", "1": "B", "2": "I"},
            },
        )
        model = OpenAIPrivacyFilterTokenClassifier.from_preset(
            preset_dir, load_weights=False, preprocessor=None
        )
        self.assertEqual(model.num_classes, 3)

    def test_convert_tokenizer_registers_added_tokens(self):
        vocab, _ = get_bpe_fixture()
        start_id = len(vocab)
        preset_dir = self.get_temp_dir()
        tokenizer = _convert_tokenizer_json(preset_dir)
        for python_workflow in (True, False):
            tokenizer._allow_python_workflow = python_workflow
            self.assertAllEqual(tokenizer("<|startoftext|>"), [start_id])
        restored = OpenAIPrivacyFilterTokenizer.from_config(
            tokenizer.get_config()
        )
        self.assertEqual(
            list(restored.unsplittable_tokens),
            ["<|endoftext|>", "<|startoftext|>"],
        )

        # An explicit `unsplittable_tokens` is kept as given.
        tokenizer = _convert_tokenizer_json(
            preset_dir, unsplittable_tokens=["<|endoftext|>"]
        )
        self.assertEqual(tokenizer.unsplittable_tokens, ["<|endoftext|>"])

    def test_convert_tokenizer_merges_and_pre_tokenizer_errors(self):
        # `get_temp_dir` is one directory per test, so each file is
        # converted before the next one overwrites it.
        preset_dir = self.get_temp_dir()
        _, merges = get_bpe_fixture()
        from_strings = _convert_tokenizer_json(preset_dir)
        pairs = [merge.split(" ") for merge in merges]
        from_pairs = _convert_tokenizer_json(preset_dir, merges=pairs)
        self.assertEqual(from_strings.merges, merges)
        self.assertEqual(from_pairs.merges, merges)
        text = "airplane at airport"
        self.assertAllEqual(from_pairs(text), from_strings(text))

        # A different split regex is rejected, naming the one found.
        bad_regex = {"pretokenizers": [{"pattern": {"Regex": "\\s+"}}]}
        with self.assertRaisesRegex(ValueError, re.escape(repr("\\s+"))):
            _convert_tokenizer_json(preset_dir, pre_tokenizer=bad_regex)
        # A bare `Split` instead of a `Sequence` gets a `ValueError` with
        # context, not a `KeyError`.
        single_split = {"type": "Split", "pattern": {"Regex": "\\s+"}}
        with self.assertRaisesRegex(ValueError, "Expected `pre_tokenizer`"):
            _convert_tokenizer_json(preset_dir, pre_tokenizer=single_split)

    def test_convert_weights_round_trip(self):
        # Runs the converter on an in-memory HF-style state dict, so the
        # weight mapping is checked in CI without a download. No projection
        # is square and `intermediate` differs from `hidden`, and every HF
        # tensor holds distinct values, so a dropped transpose or a swapped
        # or mis-sourced key changes the ported values.
        hidden, heads, kv_heads, head_dim = 6, 2, 1, 4
        intermediate, experts, vocab = 5, 2, 10
        backbone = OpenAIPrivacyFilterBackbone(
            vocabulary_size=vocab,
            num_layers=1,
            num_query_heads=heads,
            num_key_value_heads=kv_heads,
            hidden_dim=hidden,
            intermediate_dim=intermediate,
            head_dim=head_dim,
            num_experts=experts,
            top_k=1,
            sliding_window=4,
        )
        task = OpenAIPrivacyFilterTokenClassifier(
            backbone=backbone, num_classes=3
        )
        layer = backbone.encoder_layers[0]
        attn = layer.self_attention
        moe = layer.sparse_moe_block
        prefix = "model.layers.0."

        def same(w):
            return w

        def transpose(w):
            return w.T

        def qkv_kernel(num_heads):
            # HF `(out, in)` to Keras `(in, heads, head_dim)`.
            return lambda w: w.T.reshape(hidden, num_heads, head_dim)

        def qkv_bias(num_heads):
            return lambda w: w.reshape(num_heads, head_dim)

        def o_kernel(w):
            # HF `(hidden, heads * head_dim)` to `(heads, head_dim, hidden)`.
            return w.T.reshape(heads, head_dim, hidden)

        q_size, kv_size = heads * head_dim, kv_heads * head_dim
        # (Keras variable, HF key, HF shape, HF to Keras)
        mapping = [
            (
                backbone.token_embedding.embeddings,
                "model.embed_tokens.weight",
                (vocab, hidden),
                same,
            ),
            (
                layer.input_layernorm.scale,
                prefix + "input_layernorm.weight",
                (hidden,),
                same,
            ),
            (
                attn.query_dense.kernel,
                prefix + "self_attn.q_proj.weight",
                (q_size, hidden),
                qkv_kernel(heads),
            ),
            (
                attn.query_dense.bias,
                prefix + "self_attn.q_proj.bias",
                (q_size,),
                qkv_bias(heads),
            ),
            (
                attn.key_dense.kernel,
                prefix + "self_attn.k_proj.weight",
                (kv_size, hidden),
                qkv_kernel(kv_heads),
            ),
            (
                attn.key_dense.bias,
                prefix + "self_attn.k_proj.bias",
                (kv_size,),
                qkv_bias(kv_heads),
            ),
            (
                attn.value_dense.kernel,
                prefix + "self_attn.v_proj.weight",
                (kv_size, hidden),
                qkv_kernel(kv_heads),
            ),
            (
                attn.value_dense.bias,
                prefix + "self_attn.v_proj.bias",
                (kv_size,),
                qkv_bias(kv_heads),
            ),
            (
                attn.output_dense.kernel,
                prefix + "self_attn.o_proj.weight",
                (hidden, q_size),
                o_kernel,
            ),
            (
                attn.output_dense.bias,
                prefix + "self_attn.o_proj.bias",
                (hidden,),
                same,
            ),
            (attn.sinks, prefix + "self_attn.sinks", (heads,), same),
            (
                layer.post_attention_layernorm.scale,
                prefix + "post_attention_layernorm.weight",
                (hidden,),
                same,
            ),
            (
                moe.router.router_dense.kernel,
                prefix + "mlp.router.weight",
                (experts, hidden),
                transpose,
            ),
            (
                moe.router.router_dense.bias,
                prefix + "mlp.router.bias",
                (experts,),
                same,
            ),
            (
                moe.experts.gate_up_proj,
                prefix + "mlp.experts.gate_up_proj",
                (experts, hidden, 2 * intermediate),
                same,
            ),
            (
                moe.experts.gate_up_proj_bias,
                prefix + "mlp.experts.gate_up_proj_bias",
                (experts, 2 * intermediate),
                same,
            ),
            (
                moe.experts.down_proj,
                prefix + "mlp.experts.down_proj",
                (experts, intermediate, hidden),
                same,
            ),
            (
                moe.experts.down_proj_bias,
                prefix + "mlp.experts.down_proj_bias",
                (experts, hidden),
                same,
            ),
            (
                backbone.final_layernorm.scale,
                "model.norm.weight",
                (hidden,),
                same,
            ),
            (
                task._classifier_dense.kernel,
                "score.weight",
                (3, hidden),
                transpose,
            ),
            (task._classifier_dense.bias, "score.bias", (3,), same),
        ]
        rng = np.random.RandomState(0)
        hf_state_dict = {
            key: rng.uniform(size=shape) for _, key, shape, _ in mapping
        }

        class FakeLoader:
            def __init__(self):
                self.ported_keys = []

            def port_weight(self, keras_variable, hf_weight_key, hook_fn=None):
                self.ported_keys.append(hf_weight_key)
                tensor = hf_state_dict[hf_weight_key]
                if hook_fn:
                    tensor = hook_fn(tensor, keras_variable.shape)
                keras_variable.assign(tensor)

        loader = FakeLoader()
        convert_weights(backbone, loader, {})
        convert_head(task, loader, {})

        # Every HF tensor is read exactly once, `mapping` covers every
        # trainable Keras weight, and each one holds its own HF tensor.
        self.assertEqual(sorted(loader.ported_keys), sorted(hf_state_dict))
        self.assertLen(mapping, len(task.trainable_weights))
        for variable, key, _, to_keras in mapping:
            self.assertAllClose(
                ops.convert_to_numpy(variable),
                to_keras(hf_state_dict[key]),
                msg=key,
            )

    @pytest.mark.large
    def test_convert_tiny_preset(self):
        text = "My name is John Smith."
        model = OpenAIPrivacyFilterTokenClassifier.from_preset(TINY_PRESET)
        logits = model.predict([text])
        # One logit per BIOES label in the checkpoint's `id2label`.
        self.assertEqual(logits.shape, (1, 512, 33))

        torch = pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")
        if not hasattr(transformers, "OpenAIPrivacyFilterConfig"):
            pytest.skip(
                "Needs OpenAIPrivacyFilterConfig, installed "
                f"transformers={transformers.__version__} does not have it."
            )
        hf_repo = TINY_PRESET.removeprefix("hf://")
        hf_model = (
            transformers.AutoModelForTokenClassification.from_pretrained(
                hf_repo
            )
            .to(device="cpu", dtype=torch.float32)
            .eval()
        )
        ids = ops.convert_to_numpy(model.preprocessor.tokenizer(text)).tolist()
        with torch.no_grad():
            hf_logits = (
                hf_model(input_ids=torch.tensor([ids], device="cpu"))
                .logits.cpu()
                .numpy()
            )
        # A100 worst drift 2.4e-3 (jax and tensorflow), at least 3x headroom.
        atol = 8e-3 if running_on_gpu() else 1e-4
        self.assertAllClose(
            logits[0, : len(ids)], hf_logits[0, : len(ids)], atol=atol
        )

    @pytest.mark.large
    def test_class_detection(self):
        model = TokenClassifier.from_preset(TINY_PRESET, load_weights=False)
        self.assertIsInstance(model, OpenAIPrivacyFilterTokenClassifier)
        model = Backbone.from_preset(TINY_PRESET, load_weights=False)
        self.assertIsInstance(model, OpenAIPrivacyFilterBackbone)

    @pytest.mark.large
    def test_logits_match_hf(self):
        # Build a small random HF model, convert it, and compare the logits.
        # The sliding window is shorter than the input and the second row is
        # padded, so both masks are exercised.
        torch = pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")
        if not hasattr(transformers, "OpenAIPrivacyFilterConfig"):
            pytest.skip(
                "Needs OpenAIPrivacyFilterConfig, installed "
                f"transformers={transformers.__version__} does not have it."
            )
        config = transformers.OpenAIPrivacyFilterConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_local_experts=8,
            num_experts_per_tok=2,
            sliding_window=3,
            initializer_range=0.2,
            num_labels=33,
            pad_token_id=0,
            eos_token_id=0,
        )
        torch.manual_seed(0)
        hf_model = transformers.OpenAIPrivacyFilterForTokenClassification(
            config
        ).eval()
        with tempfile.TemporaryDirectory() as preset_dir:
            hf_model.save_pretrained(preset_dir)
            model = OpenAIPrivacyFilterTokenClassifier.from_preset(
                preset_dir, preprocessor=None
            )

        token_ids = np.array(
            [list(range(1, 13)), list(range(20, 28)) + [0] * 4], dtype="int32"
        )
        padding_mask = (np.arange(12) < np.array([[12], [8]])).astype("int32")
        logits = ops.convert_to_numpy(
            model({"token_ids": token_ids, "padding_mask": padding_mask})
        )
        with torch.no_grad():
            hf_logits = hf_model(
                input_ids=torch.tensor(token_ids),
                attention_mask=torch.tensor(padding_mask),
            ).logits.numpy()
        real = padding_mask.astype(bool)
        # A100 worst drift 7.67e-3 (jax), at least 3x headroom.
        atol = 3e-2 if running_on_gpu() else 1e-4
        self.assertAllClose(logits[real], hf_logits[real], atol=atol)
