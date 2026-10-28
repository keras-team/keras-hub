import json
import os
import re
import tempfile

import numpy as np
import pytest
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.masked_lm import MaskedLM
from keras_hub.src.models.modern_bert.modern_bert_backbone import (
    ModernBertBackbone,
)
from keras_hub.src.models.modern_bert.modern_bert_masked_lm import (
    ModernBertMaskedLM,
)
from keras_hub.src.models.modern_bert.modern_bert_text_classifier import (
    ModernBertTextClassifier,
)
from keras_hub.src.models.modern_bert.modern_bert_text_embedder import (
    ModernBertTextEmbedder,
)
from keras_hub.src.models.modern_bert.modern_bert_text_embedder_preprocessor import (  # noqa: E501
    ModernBertTextEmbedderPreprocessor,
)
from keras_hub.src.models.modern_bert.modern_bert_tokenizer import (
    ModernBertTokenizer,
)
from keras_hub.src.models.text_classifier import TextClassifier
from keras_hub.src.models.text_embedder import TextEmbedder
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.keras_utils import running_on_gpu
from keras_hub.src.utils.transformers.convert_modern_bert import (
    convert_tokenizer,
)
from keras_hub.src.utils.transformers.convert_modern_bert import (
    load_preprocessor_config,
)
from keras_hub.src.utils.transformers.convert_modern_bert import (
    load_task_config,
)

HF_MODEL_ID = "answerdotai/ModernBERT-base"
HF_PRESET = f"hf://{HF_MODEL_ID}"

# ModernBERT sentence-transformers checkpoints. Both store their tokenizer
# merges as `["a", "b"]` pairs, unlike `answerdotai/ModernBERT-base`.
NOMIC_EMBED_MODEL_ID = "nomic-ai/modernbert-embed-base"  # Mean, normalized.
GTE_MODEL_ID = "Alibaba-NLP/gte-modernbert-base"  # CLS, not normalized.

# `local_attention=128` gives a sliding-window radius of 64. A sequence
# shorter than that makes the window mask all-ones, so the local layers
# become indistinguishable from the global ones and the comparison silently
# stops covering the alternating-attention logic. 256 is the shortest length
# that exercises it.
SEQ_LEN = 256

# A tiny byte-level BPE vocabulary for the offline tokenizer test, as in the
# ModernBERT tokenizer tests.
TINY_MERGES = [
    "Ġ a",
    "Ġ t",
    "Ġ i",
    "Ġ b",
    "a i",
    "p l",
    "n e",
    "Ġa t",
    "p o",
    "r t",
    "Ġt h",
    "ai r",
    "pl a",
    "po rt",
    "Ġai r",
    "Ġa i",
    "pla ne",
]
TINY_SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]"]


def _tiny_tokenizer_json(pair_merges):
    """A minimal `tokenizer.json`, with merges as pairs or as strings."""
    tokens = []
    for merge in TINY_MERGES:
        a, b = merge.split(" ")
        tokens.extend([a, b, a + b])
    vocab = {token: i for i, token in enumerate(sorted(set(tokens)))}
    added_tokens = [
        {"id": len(vocab) + i, "content": token, "special": True}
        for i, token in enumerate(TINY_SPECIAL_TOKENS)
    ]
    if pair_merges:
        merges = [merge.split(" ") for merge in TINY_MERGES]
    else:
        merges = list(TINY_MERGES)
    return {
        "added_tokens": added_tokens,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False},
        "model": {"type": "BPE", "vocab": vocab, "merges": merges},
    }


class TestConvertModernBert(TestCase):
    def _write_preset(self, files):
        """Write `{relative_path: json_content}` to a local preset dir."""
        preset = tempfile.mkdtemp(dir=self.get_temp_dir())
        for path, content in files.items():
            full_path = os.path.join(preset, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                json.dump(content, f)
        return preset

    def _tiny_backbone(self):
        return ModernBertBackbone(
            vocabulary_size=10,
            hidden_dim=8,
            intermediate_dim=16,
            num_layers=2,
            num_heads=2,
            local_attention_window=4,
            global_attn_every_n_layers=2,
        )

    def test_load_task_config(self):
        transformer = {"type": "sentence_transformers.models.Transformer"}
        normalize = {"type": "sentence_transformers.models.Normalize"}

        # No sentence-transformers files: nothing is overridden.
        self.assertEqual(load_task_config(self._write_preset({}), {}), {})

        # Legacy boolean flags, with a `Normalize` module.
        preset = self._write_preset(
            {
                "modules.json": [transformer, normalize],
                "1_Pooling/config.json": {
                    "pooling_mode_cls_token": False,
                    "pooling_mode_mean_tokens": True,
                    "pooling_mode_max_tokens": False,
                },
            }
        )
        self.assertEqual(
            load_task_config(preset, {}),
            {"normalize": True, "pooling_mode": "mean"},
        )

        # Newer string form, without a `Normalize` module.
        preset = self._write_preset(
            {
                "modules.json": [transformer],
                "1_Pooling/config.json": {"pooling_mode": "cls"},
            }
        )
        self.assertEqual(
            load_task_config(preset, {}),
            {"normalize": False, "pooling_mode": "cls"},
        )

        # As in sentence-transformers, the string form takes precedence over
        # the flags, and a config with no mode enabled means mean pooling.
        preset = self._write_preset(
            {
                "1_Pooling/config.json": {
                    "pooling_mode": "max",
                    "pooling_mode_mean_tokens": True,
                }
            }
        )
        self.assertEqual(load_task_config(preset, {}), {"pooling_mode": "max"})
        preset = self._write_preset(
            {"1_Pooling/config.json": {"pooling_mode_mean_tokens": False}}
        )
        self.assertEqual(load_task_config(preset, {}), {"pooling_mode": "mean"})

    @parameterized.named_parameters(
        (
            "flag_weightedmean",
            {"pooling_mode_weightedmean_tokens": True},
            "weightedmean",
        ),
        (
            "string_weightedmean",
            {"pooling_mode": "weightedmean"},
            "weightedmean",
        ),
        ("string_lasttoken", {"pooling_mode": "lasttoken"}, "lasttoken"),
        (
            "flag_multiple",
            {"pooling_mode_cls_token": True, "pooling_mode_mean_tokens": True},
            "cls+mean",
        ),
        ("string_multiple", {"pooling_mode": ["mean", "max"]}, "mean+max"),
    )
    def test_load_task_config_names_unsupported_pooling(
        self, pooling_config, pooling_mode
    ):
        """Unsupported pooling raises an error naming it, in both formats."""
        preset = self._write_preset({"1_Pooling/config.json": pooling_config})
        kwargs = load_task_config(preset, {})
        self.assertEqual(kwargs, {"pooling_mode": pooling_mode})
        with self.assertRaisesRegex(
            ValueError, f"Invalid pooling_mode: '{re.escape(pooling_mode)}'"
        ):
            ModernBertTextEmbedder(backbone=self._tiny_backbone(), **kwargs)

    def test_convert_tokenizer_accepts_pair_merges(self):
        """Merges stored as `["a", "b"]` pairs load like `"a b"` strings."""
        string_tokenizer = convert_tokenizer(
            ModernBertTokenizer,
            self._write_preset(
                {"tokenizer.json": _tiny_tokenizer_json(pair_merges=False)}
            ),
        )
        pair_tokenizer = convert_tokenizer(
            ModernBertTokenizer,
            self._write_preset(
                {"tokenizer.json": _tiny_tokenizer_json(pair_merges=True)}
            ),
        )
        self.assertEqual(pair_tokenizer.merges, TINY_MERGES)

        text = " airplane at airport"
        expected = ["Ġair", "plane", "Ġat", "Ġair", "port"]
        pair_ids = ops.convert_to_numpy(pair_tokenizer(text))
        self.assertEqual(
            [pair_tokenizer.id_to_token(int(i)) for i in pair_ids], expected
        )
        self.assertAllEqual(
            pair_ids, ops.convert_to_numpy(string_tokenizer(text))
        )

    def test_load_preprocessor_config(self):
        transformers_config = {"max_position_embeddings": 8192}
        modules = [{"type": "sentence_transformers.models.Transformer"}]

        # `max_seq_length` from `sentence_bert_config.json` is used as-is.
        preset = self._write_preset(
            {
                "modules.json": modules,
                "sentence_bert_config.json": {"max_seq_length": 512},
            }
        )
        self.assertEqual(
            load_preprocessor_config(preset, transformers_config),
            {"sequence_length": 512},
        )

        # Without it, sentence-transformers uses `max_position_embeddings`
        # (`Alibaba-NLP/gte-modernbert-base` ships no `sentence_bert_config`
        # and an effectively unset tokenizer `model_max_length`)...
        for files in (
            {"modules.json": modules},
            {
                "modules.json": modules,
                "tokenizer_config.json": {"model_max_length": int(1e30)},
            },
        ):
            self.assertEqual(
                load_preprocessor_config(
                    self._write_preset(files), transformers_config
                ),
                {"sequence_length": 8192},
            )

        # ...capped at the tokenizer's `model_max_length`.
        preset = self._write_preset(
            {
                "modules.json": modules,
                "tokenizer_config.json": {"model_max_length": 1024},
            }
        )
        self.assertEqual(
            load_preprocessor_config(preset, transformers_config),
            {"sequence_length": 1024},
        )

        # Not a sentence-transformers checkpoint (no `modules.json`, e.g.
        # `answerdotai/ModernBERT-base`): the preprocessor default is kept.
        self.assertEqual(
            load_preprocessor_config(
                self._write_preset({}), transformers_config
            ),
            {},
        )

    @pytest.mark.extra_large
    def test_convert_modern_bert_base(self):
        """Compare backbone hidden states against HF.

        Loads through `from_preset` rather than hand-driving
        `SafetensorLoader`, so `preset_loader.py` dispatch and
        `convert_backbone_config` are both in the path.
        """
        import torch
        from transformers import AutoModel

        hf_model = AutoModel.from_pretrained(HF_MODEL_ID)
        hf_model.eval()

        backbone = ModernBertBackbone.from_preset(HF_PRESET)

        np.random.seed(42)
        batch_size = 2
        vocabulary_size = hf_model.config.vocab_size
        token_ids = np.random.randint(
            100, vocabulary_size - 100, size=(batch_size, SEQ_LEN)
        )
        padding_mask = np.ones((batch_size, SEQ_LEN), dtype=np.int64)

        with torch.no_grad():
            hf_outputs = (
                hf_model(
                    input_ids=torch.from_numpy(token_ids),
                    attention_mask=torch.from_numpy(padding_mask),
                )
                .last_hidden_state.cpu()
                .numpy()
            )

        keras_outputs = backbone(
            {
                "token_ids": token_ids,
                "padding_mask": padding_mask.astype(bool),
            }
        )

        keras_outputs = ops.convert_to_numpy(keras_outputs)

        # Divergence grows with sequence length; ~5e-4 is expected at 256, so
        # this leaves headroom without being vacuous.
        self.assertAllClose(hf_outputs, keras_outputs, atol=2e-3, rtol=2e-3)

    @pytest.mark.extra_large
    def test_convert_masked_lm_end_to_end(self):
        """Fill-mask parity with the tokenizer in the loop.

        The backbone comparison above feeds identical ids to both models,
        which bypasses `ModernBertTokenizer` entirely. This test starts from
        raw text on both sides, so a tokenization difference (such as the
        `[MASK]` `lstrip` handling) shows up as a prediction mismatch.
        """
        import torch
        from transformers import AutoModelForMaskedLM
        from transformers import AutoTokenizer

        hf_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        hf_model = AutoModelForMaskedLM.from_pretrained(HF_MODEL_ID)
        hf_model.eval()

        keras_model = ModernBertMaskedLM.from_preset(HF_PRESET)
        keras_tokenizer = keras_model.preprocessor.tokenizer

        prompts = [
            "The capital of France is [MASK].",
            "She went to the [MASK] to buy milk.",
            "[MASK] is the largest planet in our solar system.",
        ]

        for prompt in prompts:
            hf_inputs = hf_tokenizer(prompt, return_tensors="pt")
            keras_ids = [int(i) for i in keras_tokenizer([prompt])[0]]

            hf_body_ids = [
                i
                for i in hf_inputs["input_ids"][0].tolist()
                if i
                not in (
                    hf_tokenizer.cls_token_id,
                    hf_tokenizer.sep_token_id,
                )
            ]
            self.assertAllEqual(
                keras_ids,
                hf_body_ids,
                msg=f"Tokenization diverged for {prompt!r}",
            )

            with torch.no_grad():
                hf_logits = hf_model(**hf_inputs).logits

            hf_mask_position = (
                (hf_inputs["input_ids"][0] == hf_tokenizer.mask_token_id)
                .nonzero()[0]
                .item()
            )
            hf_top1 = int(hf_logits[0, hf_mask_position].argmax())

            # Drive the KerasHub forward pass from KerasHub's own
            # tokenization, so this stays an end-to-end check.
            keras_input_ids = np.array(
                [
                    [keras_tokenizer.cls_token_id]
                    + keras_ids
                    + [keras_tokenizer.sep_token_id]
                ]
            )
            keras_mask_position = int(
                np.argmax(keras_input_ids[0] == keras_tokenizer.mask_token_id)
            )

            keras_logits = keras_model(
                {
                    "token_ids": keras_input_ids,
                    "padding_mask": np.ones_like(keras_input_ids, dtype=bool),
                    "mask_positions": np.array([[keras_mask_position]]),
                }
            )
            keras_logits = ops.convert_to_numpy(keras_logits)
            keras_top1 = int(keras_logits[0, 0].argmax())

            self.assertEqual(
                keras_top1,
                hf_top1,
                msg=(
                    f"Fill-mask prediction diverged for {prompt!r}: "
                    f"KerasHub predicted "
                    f"{hf_tokenizer.decode(keras_top1)!r}, HF predicted "
                    f"{hf_tokenizer.decode(hf_top1)!r}"
                ),
            )

    @pytest.mark.large
    def test_class_detection(self):
        model = MaskedLM.from_preset(HF_PRESET, load_weights=False)
        self.assertIsInstance(model, ModernBertMaskedLM)

        model = Backbone.from_preset(HF_PRESET, load_weights=False)
        self.assertIsInstance(model, ModernBertBackbone)

    @pytest.mark.large
    def test_text_embedder_class_detection(self):
        """Sentence-transformers settings are read from the checkpoint."""
        model = TextEmbedder.from_preset(
            f"hf://{NOMIC_EMBED_MODEL_ID}", load_weights=False
        )
        self.assertIsInstance(model, ModernBertTextEmbedder)
        self.assertEqual(model.pooling_mode, "mean")
        self.assertTrue(model.normalize)
        # From `sentence_bert_config.json`.
        self.assertEqual(model.preprocessor.sequence_length, 8192)

        model = TextEmbedder.from_preset(
            f"hf://{GTE_MODEL_ID}", load_weights=False
        )
        self.assertIsInstance(model, ModernBertTextEmbedder)
        self.assertEqual(model.pooling_mode, "cls")
        self.assertFalse(model.normalize)
        # No `sentence_bert_config.json`, so `max_position_embeddings`, as
        # sentence-transformers does.
        self.assertEqual(model.preprocessor.sequence_length, 8192)

    @pytest.mark.large
    def test_text_embedder_preprocessor_sequence_length_override(self):
        preprocessor = ModernBertTextEmbedderPreprocessor.from_preset(
            f"hf://{NOMIC_EMBED_MODEL_ID}", sequence_length=128
        )
        self.assertEqual(preprocessor.sequence_length, 128)

    @pytest.mark.large
    def test_other_tasks_load_from_embedding_checkpoint(self):
        """Embedder-only task kwargs must not leak into other tasks."""
        model = TextClassifier.from_preset(
            f"hf://{GTE_MODEL_ID}", num_classes=2, load_weights=False
        )
        self.assertIsInstance(model, ModernBertTextClassifier)
        # `pooling_mode` and `normalize` from the checkpoint's
        # sentence-transformers files are not set on, or saved by, the model.
        for name in ("pooling_mode", "normalize"):
            self.assertFalse(hasattr(model, name))
            self.assertNotIn(name, model.get_config())
        # And the classifier still runs end to end.
        model.preprocessor.sequence_length = 16
        logits = model.predict(["The quick brown fox jumped."])
        self.assertEqual(logits.shape, (1, 2))
        self.assertTrue(np.all(np.isfinite(logits)))

    @pytest.mark.extra_large
    def test_text_embedder_numerics(self):
        """Compare sentence embeddings against HF, starting from raw text.

        HF hidden states are pooled as each checkpoint's sentence-transformers
        config specifies. The batch mixes short texts with one longer than the
        local attention window, so it is padded and exercises both the local
        and the global attention layers.
        """
        import torch
        from transformers import AutoModel
        from transformers import AutoTokenizer

        texts = [
            "search_query: What is TSNE?",
            "search_query: Who is Laurens van der Maaten?",
            "search_document: "
            + " ".join(
                [
                    "TSNE is a dimensionality reduction algorithm created "
                    "by Laurens van der Maaten."
                ]
                * 12
            ),
        ]

        for hf_model_id, pooling_mode, normalize in (
            (NOMIC_EMBED_MODEL_ID, "mean", True),
            (GTE_MODEL_ID, "cls", False),
        ):
            hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
            # Compare in float32. `gte-modernbert-base` is stored in float16,
            # which recent `transformers` versions would otherwise run in.
            hf_model = AutoModel.from_pretrained(hf_model_id).float()
            hf_model.eval()
            hf_inputs = hf_tokenizer(texts, padding=True, return_tensors="pt")
            with torch.no_grad():
                hidden = hf_model(**hf_inputs).last_hidden_state
            if pooling_mode == "mean":
                mask = hf_inputs["attention_mask"].unsqueeze(-1).float()
                hf_embeddings = (hidden * mask).sum(1) / mask.sum(1)
            else:
                hf_embeddings = hidden[:, 0]
            if normalize:
                hf_embeddings = torch.nn.functional.normalize(
                    hf_embeddings, dim=-1
                )
            hf_embeddings = hf_embeddings.numpy()

            embedder = ModernBertTextEmbedder.from_preset(f"hf://{hf_model_id}")
            self.assertEqual(embedder.pooling_mode, pooling_mode)
            self.assertEqual(embedder.normalize, normalize)
            # Avoid padding every input to the 8192-token default.
            embedder.preprocessor.sequence_length = 512

            # Tokenization parity, ignoring padding.
            keras_inputs = embedder.preprocessor(texts)
            keras_ids = ops.convert_to_numpy(keras_inputs["token_ids"])
            keras_mask = ops.convert_to_numpy(keras_inputs["padding_mask"])
            hf_ids = hf_inputs["input_ids"].numpy()
            hf_mask = hf_inputs["attention_mask"].numpy()
            for i in range(len(texts)):
                self.assertAllEqual(
                    keras_ids[i][keras_mask[i].astype(bool)],
                    hf_ids[i][hf_mask[i].astype(bool)],
                )

            keras_embeddings = embedder.encode_text(texts)
            # On CPU the max abs difference is ~1e-7 for nomic and ~1e-5 for
            # gte, whose unnormalized values reach ~17, so 1e-4 still catches
            # real regressions. GPU matmuls (e.g. TF32) are less precise.
            tol = 5e-3 if running_on_gpu() else 1e-4
            self.assertAllClose(
                hf_embeddings, keras_embeddings, atol=tol, rtol=tol
            )
