import tempfile
import types

import numpy as np

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
from keras_hub.src.utils.transformers.convert_mobile_bert_test import (
    write_hf_preset,
)
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
        return write_hf_preset(
            preset_dir, config, weights=weights, extra_files=extra_files
        )

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

    def _assert_config_hook_signatures(
        self,
        preset_dir,
        model_type,
        run_hook,
        expected_cls,
        hook_return,
        attr_name,
        expected_val,
    ):
        seen = []

        def hook_2(preset, config):
            seen.append(("2pos", preset, config["model_type"]))
            return dict(hook_return)

        def hook_3_default(preset, config, cls=None):
            seen.append(("3default", preset, config["model_type"], cls))
            return dict(hook_return)

        def hook_3_req(preset, config, cls):
            seen.append(("3req", preset, config["model_type"], cls))
            return dict(hook_return)

        def hook_varargs(preset, config, *args):
            seen.append(("varargs", preset, config["model_type"], args))
            return dict(hook_return)

        def hook_varkw(preset, config, **kwargs):
            seen.append(("varkw", preset, config["model_type"], kwargs))
            return dict(hook_return)

        def hook_kwonly(preset, config, *, cls=None):
            seen.append(("kwonly", preset, config["model_type"], cls))
            return dict(hook_return)

        for hook in (
            hook_2,
            hook_3_default,
            hook_3_req,
            hook_varargs,
            hook_varkw,
            hook_kwonly,
        ):
            try:
                built = run_hook(hook)
            except TypeError as e:
                self.fail(f"Unexpected TypeError for {hook.__name__}: {e}")
            self.assertIsInstance(built, expected_cls)
            self.assertEqual(
                getattr(built, attr_name), expected_val, msg=hook.__name__
            )

        self.assertEqual(
            seen,
            [
                ("2pos", preset_dir, model_type),
                ("3default", preset_dir, model_type, expected_cls),
                ("3req", preset_dir, model_type, expected_cls),
                ("varargs", preset_dir, model_type, (expected_cls,)),
                ("varkw", preset_dir, model_type, {}),
                ("kwonly", preset_dir, model_type, expected_cls),
            ],
        )

    def test_load_task_config_forwards_cls_by_arity(self):
        vit_config = self._tiny_vit_config()
        preset_dir = self._write_preset(vit_config)

        def run_task_hook(hook):
            loader = TransformersPresetLoader(preset_dir, vit_config)
            loader.converter = types.SimpleNamespace(
                backbone_cls=convert_vit.backbone_cls,
                convert_backbone_config=convert_vit.convert_backbone_config,
                load_task_config=hook,
            )
            return loader.load_task(
                ViTImageClassifier,
                load_weights=False,
                load_task_weights=False,
                num_classes=2,
            )

        self._assert_config_hook_signatures(
            preset_dir,
            "vit",
            run_task_hook,
            ViTImageClassifier,
            {"dropout": 0.25},
            "dropout",
            0.25,
        )

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

        def run_preprocessor_hook(hook):
            loader = TransformersPresetLoader(preset_dir, bert_config)
            loader.converter = types.SimpleNamespace(
                backbone_cls=convert_bert.backbone_cls,
                convert_tokenizer=convert_bert.convert_tokenizer,
                load_preprocessor_config=hook,
            )
            return loader.load_preprocessor(BertTextClassifierPreprocessor)

        self._assert_config_hook_signatures(
            preset_dir,
            "bert",
            run_preprocessor_hook,
            BertTextClassifierPreprocessor,
            {"sequence_length": 12},
            "sequence_length",
            12,
        )

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
                vit_loader_task = TransformersPresetLoader(
                    vit_dir, vit_cfg
                ).load_task(
                    ViTImageClassifier,
                    load_weights=False,
                    load_task_weights=True,
                    num_classes=2,
                )
                mb_task = ModernBertMaskedLM.from_preset(
                    mb_dir, preprocessor=None, load_weights=False
                )
            except (KeyError, IndexError, TypeError) as e:
                self.fail(
                    f"Unexpected error for architectures={arch_variant!r}: {e}"
                )
            self.assertIsInstance(vit_task, ViTImageClassifier)
            self.assertIsInstance(vit_loader_task, ViTImageClassifier)
            self.assertIsInstance(mb_task, ModernBertMaskedLM)
