import json
import os
import tempfile

import keras
import numpy as np
import pytest
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.masked_lm import MaskedLM
from keras_hub.src.models.mobile_bert.mobile_bert_backbone import (
    MobileBertBackbone,
)
from keras_hub.src.models.mobile_bert.mobile_bert_masked_lm import (
    MobileBertMaskedLM,
)
from keras_hub.src.models.mobile_bert.mobile_bert_masked_lm_preprocessor import (  # noqa: E501
    MobileBertMaskedLMPreprocessor,
)
from keras_hub.src.models.mobile_bert.mobile_bert_text_classifier import (
    MobileBertTextClassifier,
)
from keras_hub.src.models.mobile_bert.mobile_bert_text_classifier_preprocessor import (  # noqa: E501
    MobileBertTextClassifierPreprocessor,
)
from keras_hub.src.models.mobile_bert.mobile_bert_text_embedder import (
    MobileBertTextEmbedder,
)
from keras_hub.src.models.mobile_bert.mobile_bert_text_embedder_preprocessor import (  # noqa: E501
    MobileBertTextEmbedderPreprocessor,
)
from keras_hub.src.models.mobile_bert.mobile_bert_tokenizer import (
    MobileBertTokenizer,
)
from keras_hub.src.models.text_classifier import TextClassifier
from keras_hub.src.models.text_embedder import TextEmbedder
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers.convert_mobile_bert import (
    convert_backbone_config,
)

# (name, lowercase, strip_accents, expected lowercase, expected strip_accents,
# expected token for "CAFÉ").
TOKENIZER_CASES = (
    ("lower_default_strip", True, None, True, True, "cafe"),
    ("cased_default_strip", False, None, False, False, "[UNK]"),
    ("cased_strip", False, True, False, True, "CAFE"),
    # Lowercased but not stripped: "café" is not in the vocabulary.
    ("lower_no_strip", True, False, True, False, "[UNK]"),
)


class ConvertMobileBertTest(TestCase):
    @pytest.mark.large
    def test_convert_mnli_preset(self):
        model = MobileBertBackbone.from_preset(
            "hf://typeform/mobilebert-uncased-mnli"
        )
        self.assertIsInstance(model, MobileBertBackbone)

    @pytest.mark.large
    def test_class_detection(self):
        model = Backbone.from_preset(
            "hf://typeform/mobilebert-uncased-mnli",
            load_weights=False,
        )
        self.assertIsInstance(model, MobileBertBackbone)

        token_ids = np.zeros((2, 8), dtype="int32")
        segment_ids = np.zeros((2, 8), dtype="int32")
        padding_mask = np.ones((2, 8), dtype="int32")

        outputs = model(
            {
                "token_ids": token_ids,
                "segment_ids": segment_ids,
                "padding_mask": padding_mask,
            }
        )
        self.assertEqual(outputs["sequence_output"].shape, (2, 8, 512))
        self.assertEqual(outputs["pooled_output"].shape, (2, 512))

    @pytest.mark.large
    def test_task_class_detection(self):
        preset = "hf://typeform/mobilebert-uncased-mnli"
        inputs = {
            "token_ids": np.zeros((2, 8), dtype="int32"),
            "segment_ids": np.zeros((2, 8), dtype="int32"),
            "padding_mask": np.ones((2, 8), dtype="int32"),
        }
        classifier = TextClassifier.from_preset(
            preset, load_weights=False, num_classes=3
        )
        self.assertIsInstance(classifier, MobileBertTextClassifier)
        self.assertEqual(classifier.backbone.vocabulary_size, 30522)
        self.assertTrue(
            np.isfinite(ops.convert_to_numpy(classifier(inputs))).all()
        )
        # A classifier checkpoint has no `cls.predictions.*`; with
        # `load_weights=False` no safetensors file is opened at all.
        masked_lm = MaskedLM.from_preset(preset, load_weights=False)
        self.assertIsInstance(masked_lm, MobileBertMaskedLM)
        logits = masked_lm({**inputs, "mask_positions": inputs["token_ids"]})
        self.assertTrue(np.isfinite(ops.convert_to_numpy(logits)).all())
        embedder = TextEmbedder.from_preset(preset, load_weights=False)
        self.assertIsInstance(embedder, MobileBertTextEmbedder)
        self.assertEqual(embedder.backbone.vocabulary_size, 30522)
        self.assertTrue(
            np.isfinite(ops.convert_to_numpy(embedder(inputs))).all()
        )

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "cafe", "CAFE"]
    vocab += ["<unk>", "你好", "@@s", "fox"]

    def _write_minimal_preset(
        self,
        tmp_dir,
        files=None,
        config=None,
        weights=None,
        write_vocab=True,
    ):
        if write_vocab:
            with open(os.path.join(tmp_dir, "vocab.txt"), "w") as f:
                f.write("\n".join(self.vocab) + "\n")
        cfg_payload = {"model_type": "mobilebert"} if config is None else config
        with open(os.path.join(tmp_dir, "config.json"), "w") as f:
            json.dump(cfg_payload, f)
        for filename, payload in (files or {}).items():
            full_path = os.path.join(tmp_dir, filename)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                json.dump(payload, f)
        if weights is not None:
            safetensors = pytest.importorskip("safetensors.numpy")
            safetensors.save_file(
                weights, os.path.join(tmp_dir, "model.safetensors")
            )

    def _tiny_hf_config(self, **overrides):
        # Use distinct non-square dimensions (vocab=11, emb=6, bottleneck=8,
        # hidden=14, intermediate=18) so transposed vs un-transposed weights
        # and extra_kernel (8, 11) vs embeddings (11, 6) are unambiguous.
        config = {
            "model_type": "mobilebert",
            "architectures": ["MobileBertForPreTraining"],
            "vocab_size": len(self.vocab),
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "hidden_size": 14,
            "embedding_size": 6,
            "intra_bottleneck_size": 8,
            "true_hidden_size": 8,
            "intermediate_size": 18,
            "num_feedforward_networks": 2,
            "hidden_act": "relu",
            "normalization_type": "no_norm",
            "layer_norm_eps": 1e-12,
            "classifier_activation": True,
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 16,
            "type_vocab_size": 2,
            "trigram_input": True,
            "use_bottleneck": True,
            "key_query_shared_bottleneck": True,
            "use_bottleneck_attention": False,
        }
        config.update(overrides)
        return config

    def _build_synthetic_hf_weights(
        self,
        config,
        include_pooler=None,
        include_mlm=None,
        prefix="mobilebert.",
    ):
        """Construct explicit PyTorch-shaped HF safetensor arrays."""
        vocab_size = config["vocab_size"]
        emb_dim = config["embedding_size"]
        hidden_dim = config["hidden_size"]
        bneck_dim = config["intra_bottleneck_size"]
        inter_dim = config["intermediate_size"]
        num_layers = config["num_hidden_layers"]
        num_ffn = config["num_feedforward_networks"]
        max_pos = config["max_position_embeddings"]
        num_seg = config["type_vocab_size"]

        arch = (
            config["architectures"][0]
            if ("architectures" in config and config["architectures"])
            else None
        )
        if include_pooler is None:
            include_pooler = bool(config["classifier_activation"]) and arch in (
                None,
                "MobileBertModel",
                "MobileBertForPreTraining",
                "MobileBertForSequenceClassification",
                "MobileBertForMultipleChoice",
                "MobileBertForNextSentencePrediction",
            )
        if include_mlm is None:
            include_mlm = arch in (
                "MobileBertForPreTraining",
                "MobileBertForMaskedLM",
            )

        counter = 0

        def make_tensor(shape):
            nonlocal counter
            counter += 1
            size = int(np.prod(shape))
            vals = np.linspace(
                0.01 * counter, 0.01 * counter + 0.5, size, dtype="float32"
            )
            return np.ascontiguousarray(vals.reshape(shape))

        weights = {
            f"{prefix}embeddings.word_embeddings.weight": make_tensor(
                (vocab_size, emb_dim)
            ),
            f"{prefix}embeddings.embedding_transformation.weight": make_tensor(
                (hidden_dim, 3 * emb_dim)
            ),
            f"{prefix}embeddings.embedding_transformation.bias": make_tensor(
                (hidden_dim,)
            ),
            f"{prefix}embeddings.position_embeddings.weight": make_tensor(
                (max_pos, hidden_dim)
            ),
            f"{prefix}embeddings.token_type_embeddings.weight": make_tensor(
                (num_seg, hidden_dim)
            ),
            f"{prefix}embeddings.LayerNorm.weight": make_tensor((hidden_dim,)),
            f"{prefix}embeddings.LayerNorm.bias": make_tensor((hidden_dim,)),
        }

        for i in range(num_layers):
            lp = f"{prefix}encoder.layer.{i}."
            weights.update(
                {
                    f"{lp}bottleneck.input.dense.weight": make_tensor(
                        (bneck_dim, hidden_dim)
                    ),
                    f"{lp}bottleneck.input.dense.bias": make_tensor(
                        (bneck_dim,)
                    ),
                    f"{lp}bottleneck.input.LayerNorm.weight": make_tensor(
                        (bneck_dim,)
                    ),
                    f"{lp}bottleneck.input.LayerNorm.bias": make_tensor(
                        (bneck_dim,)
                    ),
                    f"{lp}bottleneck.attention.dense.weight": make_tensor(
                        (bneck_dim, hidden_dim)
                    ),
                    f"{lp}bottleneck.attention.dense.bias": make_tensor(
                        (bneck_dim,)
                    ),
                    f"{lp}bottleneck.attention.LayerNorm.weight": make_tensor(
                        (bneck_dim,)
                    ),
                    f"{lp}bottleneck.attention.LayerNorm.bias": make_tensor(
                        (bneck_dim,)
                    ),
                    f"{lp}attention.self.query.weight": make_tensor(
                        (bneck_dim, bneck_dim)
                    ),
                    f"{lp}attention.self.query.bias": make_tensor((bneck_dim,)),
                    f"{lp}attention.self.key.weight": make_tensor(
                        (bneck_dim, bneck_dim)
                    ),
                    f"{lp}attention.self.key.bias": make_tensor((bneck_dim,)),
                    f"{lp}attention.self.value.weight": make_tensor(
                        (bneck_dim, hidden_dim)
                    ),
                    f"{lp}attention.self.value.bias": make_tensor((bneck_dim,)),
                    f"{lp}attention.output.dense.weight": make_tensor(
                        (bneck_dim, bneck_dim)
                    ),
                    f"{lp}attention.output.dense.bias": make_tensor(
                        (bneck_dim,)
                    ),
                    f"{lp}attention.output.LayerNorm.weight": make_tensor(
                        (bneck_dim,)
                    ),
                    f"{lp}attention.output.LayerNorm.bias": make_tensor(
                        (bneck_dim,)
                    ),
                }
            )
            for j in range(num_ffn - 1):
                weights.update(
                    {
                        f"{lp}ffn.{j}.intermediate.dense.weight": make_tensor(
                            (inter_dim, bneck_dim)
                        ),
                        f"{lp}ffn.{j}.intermediate.dense.bias": make_tensor(
                            (inter_dim,)
                        ),
                        f"{lp}ffn.{j}.output.dense.weight": make_tensor(
                            (bneck_dim, inter_dim)
                        ),
                        f"{lp}ffn.{j}.output.dense.bias": make_tensor(
                            (bneck_dim,)
                        ),
                        f"{lp}ffn.{j}.output.LayerNorm.weight": make_tensor(
                            (bneck_dim,)
                        ),
                        f"{lp}ffn.{j}.output.LayerNorm.bias": make_tensor(
                            (bneck_dim,)
                        ),
                    }
                )
            weights.update(
                {
                    f"{lp}intermediate.dense.weight": make_tensor(
                        (inter_dim, bneck_dim)
                    ),
                    f"{lp}intermediate.dense.bias": make_tensor((inter_dim,)),
                    f"{lp}output.dense.weight": make_tensor(
                        (bneck_dim, inter_dim)
                    ),
                    f"{lp}output.dense.bias": make_tensor((bneck_dim,)),
                    f"{lp}output.LayerNorm.weight": make_tensor((bneck_dim,)),
                    f"{lp}output.LayerNorm.bias": make_tensor((bneck_dim,)),
                    f"{lp}output.bottleneck.dense.weight": make_tensor(
                        (hidden_dim, bneck_dim)
                    ),
                    f"{lp}output.bottleneck.dense.bias": make_tensor(
                        (hidden_dim,)
                    ),
                    f"{lp}output.bottleneck.LayerNorm.weight": make_tensor(
                        (hidden_dim,)
                    ),
                    f"{lp}output.bottleneck.LayerNorm.bias": make_tensor(
                        (hidden_dim,)
                    ),
                }
            )

        if include_pooler:
            weights[f"{prefix}pooler.dense.weight"] = make_tensor(
                (hidden_dim, hidden_dim)
            )
            weights[f"{prefix}pooler.dense.bias"] = make_tensor((hidden_dim,))

        if include_mlm:
            weights.update(
                {
                    "cls.predictions.transform.dense.weight": make_tensor(
                        (hidden_dim, hidden_dim)
                    ),
                    "cls.predictions.transform.dense.bias": make_tensor(
                        (hidden_dim,)
                    ),
                    "cls.predictions.transform.LayerNorm.weight": make_tensor(
                        (hidden_dim,)
                    ),
                    "cls.predictions.transform.LayerNorm.bias": make_tensor(
                        (hidden_dim,)
                    ),
                    "cls.predictions.dense.weight": make_tensor(
                        (hidden_dim - emb_dim, vocab_size)
                    ),
                    "cls.predictions.bias": make_tensor((vocab_size,)),
                }
            )
        return weights

    def _write_synthetic_checkpoint(
        self,
        tmp_dir,
        config,
        include_pooler=None,
        include_mlm=None,
        prefix="mobilebert.",
    ):
        weights = self._build_synthetic_hf_weights(
            config,
            include_pooler=include_pooler,
            include_mlm=include_mlm,
            prefix=prefix,
        )
        self._write_minimal_preset(tmp_dir, config=config, weights=weights)
        return weights

    def _assert_ported_weights_equal(
        self, model, hf_weights, prefix="mobilebert."
    ):
        """Verify every ported Keras variable directly against HF NumPy data."""
        backbone = getattr(model, "backbone", model)
        checked = set()

        def check(var, key, transpose=False):
            expected = hf_weights[key]
            if transpose:
                expected = np.transpose(expected)
            self.assertAllEqual(ops.convert_to_numpy(var), expected, msg=key)
            checked.add(key)

        check(
            backbone.token_embedding.embeddings,
            f"{prefix}embeddings.word_embeddings.weight",
        )
        check(
            backbone.embedding_projection.dense.kernel,
            f"{prefix}embeddings.embedding_transformation.weight",
            transpose=True,
        )
        check(
            backbone.embedding_projection.dense.bias,
            f"{prefix}embeddings.embedding_transformation.bias",
        )
        check(
            backbone.position_embedding.position_embeddings,
            f"{prefix}embeddings.position_embeddings.weight",
        )
        check(
            backbone.segment_embedding.embeddings,
            f"{prefix}embeddings.token_type_embeddings.weight",
        )
        check(
            backbone.embeddings_layer_norm.gamma,
            f"{prefix}embeddings.LayerNorm.weight",
        )
        check(
            backbone.embeddings_layer_norm.beta,
            f"{prefix}embeddings.LayerNorm.bias",
        )

        for i, block in enumerate(backbone.transformer_layers):
            lp = f"{prefix}encoder.layer.{i}."
            check(
                block.bottleneck_input.dense.kernel,
                f"{lp}bottleneck.input.dense.weight",
                transpose=True,
            )
            check(
                block.bottleneck_input.dense.bias,
                f"{lp}bottleneck.input.dense.bias",
            )
            check(
                block.bottleneck_input.layer_norm.gamma,
                f"{lp}bottleneck.input.LayerNorm.weight",
            )
            check(
                block.bottleneck_input.layer_norm.beta,
                f"{lp}bottleneck.input.LayerNorm.bias",
            )
            check(
                block.bottleneck_attention.dense.kernel,
                f"{lp}bottleneck.attention.dense.weight",
                transpose=True,
            )
            check(
                block.bottleneck_attention.dense.bias,
                f"{lp}bottleneck.attention.dense.bias",
            )
            check(
                block.bottleneck_attention.layer_norm.gamma,
                f"{lp}bottleneck.attention.LayerNorm.weight",
            )
            check(
                block.bottleneck_attention.layer_norm.beta,
                f"{lp}bottleneck.attention.LayerNorm.bias",
            )
            check(
                block.attention.query_dense.kernel,
                f"{lp}attention.self.query.weight",
                transpose=True,
            )
            check(
                block.attention.query_dense.bias,
                f"{lp}attention.self.query.bias",
            )
            check(
                block.attention.key_dense.kernel,
                f"{lp}attention.self.key.weight",
                transpose=True,
            )
            check(
                block.attention.key_dense.bias,
                f"{lp}attention.self.key.bias",
            )
            check(
                block.attention.value_dense.kernel,
                f"{lp}attention.self.value.weight",
                transpose=True,
            )
            check(
                block.attention.value_dense.bias,
                f"{lp}attention.self.value.bias",
            )
            check(
                block.attention.output_dense.kernel,
                f"{lp}attention.output.dense.weight",
                transpose=True,
            )
            check(
                block.attention.output_dense.bias,
                f"{lp}attention.output.dense.bias",
            )
            check(
                block.attention.output_layer_norm.gamma,
                f"{lp}attention.output.LayerNorm.weight",
            )
            check(
                block.attention.output_layer_norm.beta,
                f"{lp}attention.output.LayerNorm.bias",
            )
            for j in range(backbone.num_feedforward_networks - 1):
                ffn = block.ffn_layers[j]
                check(
                    ffn.intermediate_dense.kernel,
                    f"{lp}ffn.{j}.intermediate.dense.weight",
                    transpose=True,
                )
                check(
                    ffn.intermediate_dense.bias,
                    f"{lp}ffn.{j}.intermediate.dense.bias",
                )
                check(
                    ffn.output_dense.kernel,
                    f"{lp}ffn.{j}.output.dense.weight",
                    transpose=True,
                )
                check(
                    ffn.output_dense.bias,
                    f"{lp}ffn.{j}.output.dense.bias",
                )
                check(
                    ffn.output_layer_norm.gamma,
                    f"{lp}ffn.{j}.output.LayerNorm.weight",
                )
                check(
                    ffn.output_layer_norm.beta,
                    f"{lp}ffn.{j}.output.LayerNorm.bias",
                )
            ffn_last = block.ffn_layers[-1]
            check(
                ffn_last.intermediate_dense.kernel,
                f"{lp}intermediate.dense.weight",
                transpose=True,
            )
            check(
                ffn_last.intermediate_dense.bias,
                f"{lp}intermediate.dense.bias",
            )
            check(
                ffn_last.output_dense.kernel,
                f"{lp}output.dense.weight",
                transpose=True,
            )
            check(ffn_last.output_dense.bias, f"{lp}output.dense.bias")
            check(
                ffn_last.output_layer_norm.gamma,
                f"{lp}output.LayerNorm.weight",
            )
            check(
                ffn_last.output_layer_norm.beta,
                f"{lp}output.LayerNorm.bias",
            )
            check(
                block.output_bottleneck_dense.kernel,
                f"{lp}output.bottleneck.dense.weight",
                transpose=True,
            )
            check(
                block.output_bottleneck_dense.bias,
                f"{lp}output.bottleneck.dense.bias",
            )
            check(
                block.output_bottleneck_layer_norm.gamma,
                f"{lp}output.bottleneck.LayerNorm.weight",
            )
            check(
                block.output_bottleneck_layer_norm.beta,
                f"{lp}output.bottleneck.LayerNorm.bias",
            )

        pooler_key = f"{prefix}pooler.dense.weight"
        if backbone.pooled_dense is not None and pooler_key in hf_weights:
            check(backbone.pooled_dense.kernel, pooler_key, transpose=True)
            check(backbone.pooled_dense.bias, f"{prefix}pooler.dense.bias")

        if (
            isinstance(model, MobileBertMaskedLM)
            and "cls.predictions.bias" in hf_weights
        ):
            head = model.masked_lm_head
            check(
                head.transform_dense.kernel,
                "cls.predictions.transform.dense.weight",
                transpose=True,
            )
            check(
                head.transform_dense.bias,
                "cls.predictions.transform.dense.bias",
            )
            check(
                head.transform_layer_norm.gamma,
                "cls.predictions.transform.LayerNorm.weight",
            )
            check(
                head.transform_layer_norm.beta,
                "cls.predictions.transform.LayerNorm.bias",
            )
            check(head.extra_kernel, "cls.predictions.dense.weight")
            check(head.bias, "cls.predictions.bias")

        return checked

    @pytest.mark.large
    def test_tokenizer_from_preset(self):
        tokenizer = MobileBertTokenizer.from_preset(
            "hf://typeform/mobilebert-uncased-mnli"
        )
        self.assertIsInstance(tokenizer, MobileBertTokenizer)
        outputs = tokenizer("The quick brown fox")
        self.assertAllEqual(outputs, [1996, 4248, 2829, 4419])
        self.assertAllEqual(tokenizer("Café"), [7668])

    def _tokenizer_files(self, fmt, lowercase, strip_accents):
        if fmt == "json":
            return {
                "tokenizer.json": {
                    "normalizer": {
                        "type": "BertNormalizer",
                        "lowercase": lowercase,
                        "strip_accents": strip_accents,
                    }
                }
            }
        return {
            "tokenizer_config.json": {
                "do_lower_case": lowercase,
                "strip_accents": strip_accents,
            }
        }

    def _assert_tokenizer(
        self, files, exp_lower, exp_strip, exp_token, **kwargs
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_minimal_preset(tmp_dir, files)
            tokenizer = MobileBertTokenizer.from_preset(tmp_dir, **kwargs)
        self.assertEqual(tokenizer.lowercase, exp_lower)
        self.assertEqual(tokenizer.strip_accents, exp_strip)
        self.assertAllEqual(tokenizer("CAFÉ"), [self.vocab.index(exp_token)])
        return tokenizer

    @parameterized.named_parameters(
        (f"{fmt}_{name}", fmt, lowercase, strip_accents, lower, strip, token)
        for fmt in ("json", "config")
        for name, lowercase, strip_accents, lower, strip, token in (
            TOKENIZER_CASES
        )
    )
    def test_convert_tokenizer(
        self, fmt, lowercase, strip_accents, exp_lower, exp_strip, exp_token
    ):
        self._assert_tokenizer(
            self._tokenizer_files(fmt, lowercase, strip_accents),
            exp_lower,
            exp_strip,
            exp_token,
        )

    def test_convert_tokenizer_config_beats_json(self):
        self._assert_tokenizer(
            {
                **self._tokenizer_files("config", False, False),
                **self._tokenizer_files("json", True, True),
            },
            False,
            False,
            "[UNK]",
        )

    def test_convert_tokenizer_falls_through_to_tokenizer_json(self):
        # `tokenizer_config.json` present but without `do_lower_case`.
        self._assert_tokenizer(
            {
                "tokenizer_config.json": {"model_max_length": 512},
                **self._tokenizer_files("json", False, True),
            },
            False,
            True,
            "CAFE",
        )

    def test_convert_tokenizer_fallback_defaults(self):
        tokenizer = self._assert_tokenizer({}, True, True, "cafe")
        self.assertTrue(tokenizer.split_on_cjk)
        self.assertEqual(tokenizer.oov_token, "[UNK]")
        self.assertEqual(tokenizer.suffix_indicator, "##")

    def test_convert_tokenizer_kwargs_and_fast_vocab(self):
        # Passing `lowercase=False` without `strip_accents` on an uncased
        # `(True, None)` checkpoint must derive `strip_accents=False`.
        self._assert_tokenizer(
            self._tokenizer_files("config", True, None),
            False,
            False,
            "[UNK]",
            lowercase=False,
        )
        # Passing explicit `vocabulary=` kwarg must not collide with
        # preset vocab resolution.
        custom_vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "CAFE"]
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_minimal_preset(tmp_dir, {})
            tok = MobileBertTokenizer.from_preset(
                tmp_dir, vocabulary=custom_vocab, lowercase=False
            )
        self.assertEqual(tok.vocabulary_size(), len(custom_vocab))
        self.assertAllEqual(tok("CAFE"), [custom_vocab.index("CAFE")])

        # Fast-only `tokenizer.json` directory without `vocab.txt`.
        fast_payload = {
            "tokenizer.json": {
                "normalizer": {
                    "type": "BertNormalizer",
                    "lowercase": True,
                    "strip_accents": True,
                },
                "model": {
                    "type": "WordPiece",
                    "vocab": {t: idx for idx, t in enumerate(self.vocab)},
                },
            }
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_minimal_preset(
                tmp_dir, files=fast_payload, write_vocab=False
            )
            fast_tok = MobileBertTokenizer.from_preset(tmp_dir)
        self.assertAllEqual(fast_tok("CAFÉ"), [self.vocab.index("cafe")])

        # Missing both `vocab.txt` and `tokenizer.json` raises
        # `FileNotFoundError`.
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_minimal_preset(tmp_dir, files={}, write_vocab=False)
            with self.assertRaises(FileNotFoundError):
                MobileBertTokenizer.from_preset(tmp_dir)

    def test_convert_tokenizer_forwards_wordpiece_fields(self):
        files = {
            "tokenizer_config.json": {
                "tokenize_chinese_chars": False,
                # Special tokens may be serialized as `AddedToken` dicts.
                "unk_token": {"content": "<unk>", "lstrip": False},
            },
            "tokenizer.json": {"model": {"continuing_subword_prefix": "@@"}},
        }
        tokenizer = self._assert_tokenizer(files, True, True, "cafe")
        self.assertFalse(tokenizer.split_on_cjk)
        self.assertEqual(tokenizer.oov_token, "<unk>")
        self.assertEqual(tokenizer.suffix_indicator, "@@")
        # One token, not one per CJK character.
        self.assertAllEqual(tokenizer("你好"), [self.vocab.index("你好")])
        self.assertAllEqual(
            tokenizer("cafes"),
            [self.vocab.index("cafe"), self.vocab.index("@@s")],
        )
        self.assertAllEqual(tokenizer("zzz"), [self.vocab.index("<unk>")])

    @parameterized.named_parameters(
        ("trigram_input", {"trigram_input": False}),
        ("use_bottleneck", {"use_bottleneck": False}),
        (
            "key_query_shared_bottleneck",
            {"key_query_shared_bottleneck": False},
        ),
        ("use_bottleneck_attention", {"use_bottleneck_attention": True}),
        ("true_hidden_size", {"true_hidden_size": 4}),
        ("normalization_type", {"normalization_type": "rms_norm"}),
        ("num_feedforward_networks", {"num_feedforward_networks": 0}),
    )
    def test_backbone_config_rejects_unsupported_variants(self, overrides):
        (field,) = overrides
        with self.assertRaisesRegex(ValueError, field):
            convert_backbone_config(self._tiny_hf_config(**overrides))

    def test_backbone_config_pooler_follows_architecture(self):
        # (architectures, classifier_activation, expected use_pooler_dense).
        cases = (
            (["MobileBertForMaskedLM"], True, False),
            (["MobileBertForQuestionAnswering"], True, False),
            (["MobileBertForSequenceClassification"], True, True),
            (["MobileBertForPreTraining"], True, True),
            (["MobileBertForPreTraining"], False, False),
            (None, True, True),
            ([], True, True),
            ("absent", True, True),
        )
        for architectures, classifier_activation, expected in cases:
            config = self._tiny_hf_config(
                classifier_activation=classifier_activation
            )
            if architectures == "absent":
                del config["architectures"]
            else:
                config["architectures"] = architectures
            self.assertEqual(
                convert_backbone_config(config)["use_pooler_dense"],
                expected,
                msg=f"{architectures} {classifier_activation}",
            )

    def test_round_trip_for_pretraining_checkpoint(self):
        config = self._tiny_hf_config()
        with tempfile.TemporaryDirectory() as tmp_dir:
            hf_weights = self._write_synthetic_checkpoint(tmp_dir, config)
            model = MobileBertMaskedLM.from_preset(tmp_dir)
        self.assertIsInstance(model.backbone.pooled_dense, keras.layers.Dense)
        ported = self._assert_ported_weights_equal(model, hf_weights)
        # Pooler and MLM head were both ported, i.e. nothing was skipped.
        self.assertEqual(ported, set(hf_weights))

    def test_masked_lm_checkpoint_without_pooler(self):
        # `MobileBertForMaskedLM` is built with `add_pooling_layer=False`, so
        # no `pooler.dense.*` ships even though `classifier_activation=True`.
        config = self._tiny_hf_config(architectures=["MobileBertForMaskedLM"])
        with tempfile.TemporaryDirectory() as tmp_dir:
            hf_weights = self._write_synthetic_checkpoint(tmp_dir, config)
            self.assertNotIn("mobilebert.pooler.dense.weight", hf_weights)
            model = MobileBertMaskedLM.from_preset(tmp_dir)
        self.assertIsNone(model.backbone.pooled_dense)
        ported = self._assert_ported_weights_equal(model, hf_weights)
        self.assertEqual(ported, set(hf_weights))
        inputs = {
            "token_ids": np.ones((1, 4), dtype="int32"),
            "segment_ids": np.zeros((1, 4), dtype="int32"),
            "padding_mask": np.ones((1, 4), dtype="int32"),
        }
        outputs = model.backbone(inputs)
        self.assertAllClose(
            outputs["pooled_output"], outputs["sequence_output"][:, 0, :]
        )

    def test_backbone_without_pooler_weights(self):
        # `MobileBertModel(cfg, add_pooling_layer=False)` keeps
        # `classifier_activation=True` in `config.json` while omitting
        # `pooler.dense.*` from `model.safetensors`.
        config = self._tiny_hf_config(architectures=["MobileBertModel"])
        for prefix in ("mobilebert.", ""):
            with tempfile.TemporaryDirectory() as tmp_dir:
                hf_weights = self._write_synthetic_checkpoint(
                    tmp_dir,
                    config,
                    include_pooler=False,
                    include_mlm=False,
                    prefix=prefix,
                )
                model = MobileBertBackbone.from_preset(tmp_dir)
            self.assertIsInstance(model.pooled_dense, keras.layers.Dense)
            ported = self._assert_ported_weights_equal(
                model, hf_weights, prefix=prefix
            )
            self.assertEqual(ported, set(hf_weights))

    def test_classifier_checkpoint_without_mlm_head(self):
        config = self._tiny_hf_config(
            architectures=["MobileBertForSequenceClassification"]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            hf_weights = self._write_synthetic_checkpoint(tmp_dir, config)
            self.assertNotIn("cls.predictions.bias", hf_weights)
            masked_lm = MobileBertMaskedLM.from_preset(tmp_dir)
            classifier = MobileBertTextClassifier.from_preset(
                tmp_dir, num_classes=2
            )
        # The backbone (incl. pooler) is ported; the MLM head stays random.
        ported = self._assert_ported_weights_equal(masked_lm, hf_weights)
        self.assertEqual(ported, set(hf_weights))
        self._assert_ported_weights_equal(classifier, hf_weights)
        inputs = {
            "token_ids": np.ones((1, 4), dtype="int32"),
            "segment_ids": np.zeros((1, 4), dtype="int32"),
            "padding_mask": np.ones((1, 4), dtype="int32"),
            "mask_positions": np.zeros((1, 2), dtype="int32"),
        }
        logits = ops.convert_to_numpy(masked_lm(inputs))
        self.assertEqual(logits.shape, (1, 2, len(self.vocab)))
        self.assertTrue(np.isfinite(logits).all())

    def test_classifier_activation_false_checkpoint(self):
        config = self._tiny_hf_config(classifier_activation=False)
        with tempfile.TemporaryDirectory() as tmp_dir:
            hf_weights = self._write_synthetic_checkpoint(tmp_dir, config)
            self.assertNotIn("mobilebert.pooler.dense.weight", hf_weights)
            model = MobileBertBackbone.from_preset(tmp_dir)
        self.assertIsNone(model.pooled_dense)
        ported = self._assert_ported_weights_equal(model, hf_weights)
        # Only the backbone was loaded; the fixture's `cls.predictions.*`
        # head tensors are out of scope here.
        backbone_keys = {k for k in hf_weights if not k.startswith("cls.")}
        self.assertEqual(ported, backbone_keys)

    def test_hf_architecture_none_empty_absent(self):
        for arch_variant in (None, [], "absent"):
            config = self._tiny_hf_config()
            if arch_variant == "absent":
                del config["architectures"]
            else:
                config["architectures"] = arch_variant

            with tempfile.TemporaryDirectory() as tmp_dir:
                self._write_minimal_preset(tmp_dir, config=config)
                bb = Backbone.from_preset(tmp_dir, load_weights=False)
                mlm = MaskedLM.from_preset(tmp_dir, load_weights=False)
                clf = TextClassifier.from_preset(
                    tmp_dir, load_weights=False, num_classes=2
                )
                emb = TextEmbedder.from_preset(tmp_dir, load_weights=False)
            self.assertIsInstance(bb, MobileBertBackbone)
            self.assertIsInstance(mlm, MobileBertMaskedLM)
            self.assertIsInstance(clf, MobileBertTextClassifier)
            self.assertIsInstance(emb, MobileBertTextEmbedder)

        # Bare `architectures=None` checkpoint with and without MLM weights.
        config_none = self._tiny_hf_config(architectures=None)
        with tempfile.TemporaryDirectory() as tmp_dir:
            hf_mlm = self._write_synthetic_checkpoint(
                tmp_dir, config_none, include_pooler=True, include_mlm=True
            )
            loaded_mlm = MobileBertMaskedLM.from_preset(tmp_dir)
        self.assertEqual(
            self._assert_ported_weights_equal(loaded_mlm, hf_mlm), set(hf_mlm)
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            hf_bb = self._write_synthetic_checkpoint(
                tmp_dir, config_none, include_pooler=True, include_mlm=False
            )
            loaded_bb_mlm = MobileBertMaskedLM.from_preset(tmp_dir)
        self.assertEqual(
            self._assert_ported_weights_equal(loaded_bb_mlm, hf_bb), set(hf_bb)
        )

    def test_task_and_preprocessor_config(self):
        # 1. Plain HF checkpoint with `max_position_embeddings=16` clamps all
        # preprocessors so `.predict(["cafe"])` does not overflow positions.
        config = self._tiny_hf_config(
            architectures=["MobileBertModel"], classifier_dropout=0.25
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_synthetic_checkpoint(
                tmp_dir, config, include_pooler=True, include_mlm=False
            )
            clf_pre = MobileBertTextClassifierPreprocessor.from_preset(tmp_dir)
            emb_pre = MobileBertTextEmbedderPreprocessor.from_preset(tmp_dir)
            mlm_pre = MobileBertMaskedLMPreprocessor.from_preset(tmp_dir)
            clf = MobileBertTextClassifier.from_preset(tmp_dir, num_classes=2)
            emb = MobileBertTextEmbedder.from_preset(tmp_dir)
        self.assertEqual(clf_pre.sequence_length, 16)
        self.assertEqual(emb_pre.sequence_length, 16)
        self.assertEqual(mlm_pre.sequence_length, 16)
        self.assertEqual(clf.dropout, 0.25)
        pred = emb.predict(["cafe"])
        self.assertEqual(pred.shape, (1, 14))
        self.assertTrue(np.isfinite(pred).all())

        # 2. Sentence-Transformers checkpoint (`sentence_bert_config.json`,
        # `1_Pooling/config.json`, `modules.json`).
        st_files = {
            "sentence_bert_config.json": {"max_seq_length": 12},
            "1_Pooling/config.json": {
                "pooling_mode_cls_token": True,
                "pooling_mode_mean_tokens": False,
                "pooling_mode_max_tokens": False,
            },
            "modules.json": [{"type": "sentence_transformers.models.Pooling"}],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            hf_w = self._build_synthetic_hf_weights(
                config, include_pooler=True, include_mlm=False
            )
            self._write_minimal_preset(
                tmp_dir, files=st_files, config=config, weights=hf_w
            )
            st_emb = MobileBertTextEmbedder.from_preset(tmp_dir)
        self.assertEqual(st_emb.pooling_mode, "cls")
        self.assertFalse(st_emb.normalize)
        self.assertEqual(st_emb.preprocessor.sequence_length, 12)

        # 3. `sentence_bert_config.json` without `max_seq_length` falls through
        # to `modules.json` + `tokenizer_config.json` and never returns None.
        fallthrough_files = {
            "sentence_bert_config.json": {"do_lower_case": True},
            "modules.json": [
                {"type": "sentence_transformers.models.Normalize"}
            ],
            "tokenizer_config.json": {"model_max_length": 10},
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_minimal_preset(
                tmp_dir, files=fallthrough_files, config=config
            )
            ft_pre = MobileBertTextEmbedderPreprocessor.from_preset(tmp_dir)
            ft_emb = MobileBertTextEmbedder.from_preset(
                tmp_dir, load_weights=False
            )
        self.assertEqual(ft_pre.sequence_length, 10)
        self.assertTrue(ft_emb.normalize)
