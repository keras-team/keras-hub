import numpy as np
import pytest
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
from keras_hub.src.models.text_classifier import TextClassifier
from keras_hub.src.models.text_embedder import TextEmbedder
from keras_hub.src.tests.test_case import TestCase

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


class TestConvertModernBert(TestCase):
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

    @pytest.mark.large
    def test_text_embedder_preprocessor_sequence_length_override(self):
        preprocessor = ModernBertTextEmbedderPreprocessor.from_preset(
            f"hf://{NOMIC_EMBED_MODEL_ID}", sequence_length=128
        )
        self.assertEqual(preprocessor.sequence_length, 128)

    @pytest.mark.large
    def test_other_tasks_load_from_embedding_checkpoint(self):
        """Embedder-only task kwargs must not break other ModernBERT tasks."""
        model = TextClassifier.from_preset(
            f"hf://{GTE_MODEL_ID}", num_classes=2, load_weights=False
        )
        self.assertIsInstance(model, ModernBertTextClassifier)

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
            self.assertAllClose(
                hf_embeddings, keras_embeddings, atol=2e-3, rtol=2e-3
            )
