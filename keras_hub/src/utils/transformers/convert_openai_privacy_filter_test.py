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
from keras_hub.src.models.token_classifier import TokenClassifier
from keras_hub.src.tests.test_case import TestCase
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
    convert_weights,
)

TINY_PRESET = "hf://yujiepan/openai-privacy-filter-tiny-random"


class TestTask(TestCase):
    def test_convert_config(self):
        # The keys and values of the real `openai/privacy-filter` config.
        hf_config = {
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
        }
        config = convert_backbone_config(hf_config)
        self.assertEqual(config["rope_max_wavelength"], 150000.0)
        self.assertEqual(config["rope_scaling_factor"], 32.0)
        self.assertEqual(config["num_experts"], 128)
        self.assertEqual(config["top_k"], 4)
        self.assertEqual(config["sliding_window"], 128)
        self.assertEqual(convert_task_config(hf_config), {"num_classes": 33})
        # Values the backbone hardcodes must be validated, not ignored.
        for key, value in [
            ("rope_type", "linear"),
            ("beta_fast", 8.0),
            ("truncate", True),
        ]:
            bad = {
                **hf_config,
                "rope_parameters": {**hf_config["rope_parameters"], key: value},
            }
            with self.assertRaises(ValueError):
                convert_backbone_config(bad)
        with self.assertRaises(ValueError):
            convert_backbone_config({**hf_config, "attention_bias": False})

    def test_convert_weights_round_trip(self):
        # Runs the converter on an in-memory HF-style state dict, so the
        # weight mapping is checked in CI without a download.
        backbone = OpenAIPrivacyFilterBackbone(
            vocabulary_size=10,
            num_layers=1,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=8,
            intermediate_dim=16,
            head_dim=4,
            num_experts=2,
            top_k=1,
            sliding_window=4,
        )
        task = OpenAIPrivacyFilterTokenClassifier(
            backbone=backbone, num_classes=3
        )

        config = {
            "vocab_size": 10,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "hidden_size": 8,
            "intermediate_size": 16,
            "head_dim": 4,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
        }

        rng = np.random.RandomState(0)
        hf_state_dict = {}
        hf_state_dict["model.embed_tokens.weight"] = rng.uniform(size=(10, 8))
        hf_state_dict["model.layers.0.input_layernorm.weight"] = rng.uniform(
            size=(8,)
        )
        hf_state_dict["model.layers.0.self_attn.q_proj.weight"] = rng.uniform(
            size=(8, 8)
        )
        hf_state_dict["model.layers.0.self_attn.q_proj.bias"] = rng.uniform(
            size=(8,)
        )
        hf_state_dict["model.layers.0.self_attn.k_proj.weight"] = rng.uniform(
            size=(4, 8)
        )
        hf_state_dict["model.layers.0.self_attn.k_proj.bias"] = rng.uniform(
            size=(4,)
        )
        hf_state_dict["model.layers.0.self_attn.v_proj.weight"] = rng.uniform(
            size=(4, 8)
        )
        hf_state_dict["model.layers.0.self_attn.v_proj.bias"] = rng.uniform(
            size=(4,)
        )
        hf_state_dict["model.layers.0.self_attn.o_proj.weight"] = rng.uniform(
            size=(8, 8)
        )
        hf_state_dict["model.layers.0.self_attn.o_proj.bias"] = rng.uniform(
            size=(8,)
        )
        hf_state_dict["model.layers.0.self_attn.sinks"] = rng.uniform(size=(2,))
        hf_state_dict["model.layers.0.post_attention_layernorm.weight"] = (
            rng.uniform(size=(8,))
        )
        hf_state_dict["model.layers.0.mlp.router.weight"] = rng.uniform(
            size=(2, 8)
        )
        hf_state_dict["model.layers.0.mlp.router.bias"] = rng.uniform(size=(2,))
        hf_state_dict["model.layers.0.mlp.experts.gate_up_proj"] = rng.uniform(
            size=(2, 8, 32)
        )
        hf_state_dict["model.layers.0.mlp.experts.gate_up_proj_bias"] = (
            rng.uniform(size=(2, 32))
        )
        hf_state_dict["model.layers.0.mlp.experts.down_proj"] = rng.uniform(
            size=(2, 16, 8)
        )
        hf_state_dict["model.layers.0.mlp.experts.down_proj_bias"] = (
            rng.uniform(size=(2, 8))
        )
        hf_state_dict["model.norm.weight"] = rng.uniform(size=(8,))
        hf_state_dict["score.weight"] = rng.uniform(size=(3, 8))
        hf_state_dict["score.bias"] = rng.uniform(size=(3,))

        class FakeLoader:
            def port_weight(self, keras_variable, hf_weight_key, hook_fn=None):
                tensor = hf_state_dict[hf_weight_key]
                if hook_fn:
                    tensor = hook_fn(tensor, keras_variable.shape)
                keras_variable.assign(tensor)

        loader = FakeLoader()
        convert_weights(backbone, loader, config)
        convert_head(task, loader, config)

        # Each HF projection is `(out, in)`; the Keras kernels are
        # `(in, heads, head_dim)`. Checking q, k and v separately means a
        # dropped transpose or a swapped k/v mapping both fail here.
        attn = backbone.encoder_layers[0].self_attention
        for name, dense, heads in [
            ("q_proj", attn.query_dense, 2),
            ("k_proj", attn.key_dense, 1),
            ("v_proj", attn.value_dense, 1),
        ]:
            hf_w = hf_state_dict[f"model.layers.0.self_attn.{name}.weight"]
            self.assertAllClose(
                ops.convert_to_numpy(dense.kernel),
                hf_w.T.reshape(8, heads, 4),
            )
        moe = backbone.encoder_layers[0].sparse_moe_block
        self.assertAllClose(
            ops.convert_to_numpy(moe.router.router_dense.kernel),
            hf_state_dict["model.layers.0.mlp.router.weight"].T,
        )
        self.assertAllClose(
            ops.convert_to_numpy(moe.experts.gate_up_proj),
            hf_state_dict["model.layers.0.mlp.experts.gate_up_proj"],
        )
        self.assertAllClose(
            ops.convert_to_numpy(task._classifier_dense.kernel),
            hf_state_dict["score.weight"].T,
        )

    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = OpenAIPrivacyFilterTokenClassifier.from_preset(TINY_PRESET)
        logits = model.predict(["My name is John Smith."])
        # One logit per BIOES label in the checkpoint's `id2label`.
        self.assertEqual(logits.shape[-1], 33)

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
            pytest.skip("Needs a transformers release with this model.")
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
        self.assertAllClose(logits[real], hf_logits[real], atol=1e-4)
