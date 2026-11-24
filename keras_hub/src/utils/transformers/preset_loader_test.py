import json
import os
import tempfile
import types

import numpy as np
from safetensors.numpy import save_file

from keras_hub.src.models.bert.bert_text_classifier_preprocessor import (
    BertTextClassifierPreprocessor,
)
from keras_hub.src.models.modern_bert.modern_bert_masked_lm import (
    ModernBertMaskedLM,
)
from keras_hub.src.models.vit.vit_image_classifier import ViTImageClassifier
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_bert
from keras_hub.src.utils.transformers import convert_vit
from keras_hub.src.utils.transformers.preset_loader import (
    TransformersPresetLoader,
)


class TransformersPresetLoaderTest(TestCase):
    def _tiny_vit_config(self, **overrides):
        config = {
            "model_type": "vit",
            "architectures": ["ViTForImageClassification"],
            "image_size": 16,
            "patch_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "hidden_size": 8,
            "intermediate_size": 16,
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "qkv_bias": True,
            "id2label": {"0": "cat", "1": "dog"},
        }
        config.update(overrides)
        return config

    def _write_preset(self, config, weights=None, extra_files=None):
        preset_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
        with open(os.path.join(preset_dir, "config.json"), "w") as f:
            json.dump(config, f)
        for name, content in (extra_files or {}).items():
            full_path = os.path.join(preset_dir, name)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                if isinstance(content, str):
                    f.write(content)
                else:
                    json.dump(content, f)
        if weights is not None:
            save_file(weights, os.path.join(preset_dir, "model.safetensors"))
        return preset_dir

    def test_load_task_skips_convert_head_when_load_weights_false(self):
        vit_config = self._tiny_vit_config()
        saved_head_kernel_hf = np.full((2, 8), 7.5, dtype="float32")
        saved_head_bias = np.full((2,), -3.5, dtype="float32")
        preset_dir = self._write_preset(
            vit_config,
            weights={
                "classifier.weight": saved_head_kernel_hf,
                "classifier.bias": saved_head_bias,
            },
        )
        task = ViTImageClassifier.from_preset(preset_dir, load_weights=False)
        self.assertNotAllClose(task.output_dense.kernel, saved_head_kernel_hf.T)

        config_only_dir = self._write_preset(vit_config)
        try:
            task_no_file = ViTImageClassifier.from_preset(
                config_only_dir, load_weights=False
            )
        except FileNotFoundError as e:
            self.fail(
                f"Unexpected FileNotFoundError on load_weights=False: {e}"
            )
        self.assertIsInstance(task_no_file, ViTImageClassifier)

    def test_load_task_config_forwards_cls_by_arity(self):
        vit_config = self._tiny_vit_config()
        preset_dir = self._write_preset(vit_config)

        seen_cls = []

        def load_task_config_3(preset, config, cls=None):
            seen_cls.append(cls)
            return {}

        loader_3 = TransformersPresetLoader(preset_dir, vit_config)
        loader_3.converter = types.SimpleNamespace(
            backbone_cls=convert_vit.backbone_cls,
            convert_backbone_config=convert_vit.convert_backbone_config,
            load_task_config=load_task_config_3,
        )
        task_3 = loader_3.load_task(
            ViTImageClassifier,
            load_weights=False,
            load_task_weights=False,
            num_classes=2,
        )
        self.assertIsInstance(task_3, ViTImageClassifier)
        self.assertEqual(seen_cls, [ViTImageClassifier])

        seen_2 = []

        def load_task_config_2(preset, config):
            seen_2.append((preset, config["model_type"]))
            return {"dropout": 0.25}

        loader_2 = TransformersPresetLoader(preset_dir, vit_config)
        loader_2.converter = types.SimpleNamespace(
            backbone_cls=convert_vit.backbone_cls,
            convert_backbone_config=convert_vit.convert_backbone_config,
            load_task_config=load_task_config_2,
        )
        task_2 = loader_2.load_task(
            ViTImageClassifier,
            load_weights=False,
            load_task_weights=False,
            num_classes=2,
        )
        self.assertEqual(seen_2, [(preset_dir, "vit")])
        self.assertEqual(task_2.dropout, 0.25)

    def test_load_preprocessor_config_forwards_cls_by_arity(self):
        bert_config = {
            "model_type": "bert",
            "vocab_size": 6,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "hidden_size": 8,
            "intermediate_size": 16,
            "max_position_embeddings": 16,
        }
        preset_dir = self._write_preset(
            bert_config,
            extra_files={
                "vocab.txt": "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\n",
                "tokenizer_config.json": {"do_lower_case": True},
            },
        )

        seen_cls = []

        def load_preprocessor_config_3(preset, config, cls=None):
            seen_cls.append(cls)
            return {"sequence_length": 12}

        loader_3 = TransformersPresetLoader(preset_dir, bert_config)
        loader_3.converter = types.SimpleNamespace(
            backbone_cls=convert_bert.backbone_cls,
            convert_tokenizer=convert_bert.convert_tokenizer,
            load_preprocessor_config=load_preprocessor_config_3,
        )
        pre_3 = loader_3.load_preprocessor(BertTextClassifierPreprocessor)
        self.assertIsInstance(pre_3, BertTextClassifierPreprocessor)
        self.assertEqual(seen_cls, [BertTextClassifierPreprocessor])
        self.assertEqual(pre_3.sequence_length, 12)

        seen_2 = []

        def load_preprocessor_config_2(preset, config):
            seen_2.append((preset, config["model_type"]))
            return {"sequence_length": 10}

        loader_2 = TransformersPresetLoader(preset_dir, bert_config)
        loader_2.converter = types.SimpleNamespace(
            backbone_cls=convert_bert.backbone_cls,
            convert_tokenizer=convert_bert.convert_tokenizer,
            load_preprocessor_config=load_preprocessor_config_2,
        )
        pre_2 = loader_2.load_preprocessor(BertTextClassifierPreprocessor)
        self.assertEqual(seen_2, [(preset_dir, "bert")])
        self.assertEqual(pre_2.sequence_length, 10)

    def test_load_task_handles_missing_or_empty_architectures(self):
        base_modern_bert = {
            "model_type": "modernbert",
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "local_attention": 4,
            "global_attn_every_n_layers": 1,
            "global_rope_theta": 10000.0,
            "local_rope_theta": 10000.0,
            "norm_eps": 1e-5,
        }
        for arch_variant in (None, [], "absent"):
            vit_cfg = self._tiny_vit_config()
            mb_cfg = dict(base_modern_bert)
            if arch_variant == "absent":
                del vit_cfg["architectures"]
            else:
                vit_cfg["architectures"] = arch_variant
                mb_cfg["architectures"] = arch_variant

            vit_dir = self._write_preset(vit_cfg)
            mb_dir = self._write_preset(mb_cfg)
            try:
                vit_task = ViTImageClassifier.from_preset(
                    vit_dir, num_classes=2, load_weights=False
                )
                mb_task = ModernBertMaskedLM.from_preset(
                    mb_dir, preprocessor=None, load_weights=False
                )
            except (KeyError, IndexError, TypeError) as e:
                self.fail(
                    f"Unexpected error for architectures={arch_variant!r}: {e}"
                )
            self.assertIsInstance(vit_task, ViTImageClassifier)
            self.assertIsInstance(mb_task, ModernBertMaskedLM)
