import importlib.util
import json
import os
import subprocess
import sys

import keras
import numpy as np
import pytest
from absl.testing import parameterized

import keras_hub
from keras_hub.src.models.modern_bert import modern_bert_masked_lm_preprocessor
from keras_hub.src.models.modern_bert.modern_bert_backbone import (
    ModernBertBackbone,
)
from keras_hub.src.models.modern_bert.modern_bert_masked_lm import (
    ModernBertMaskedLM,
)
from keras_hub.src.models.modern_bert.modern_bert_tokenizer import (
    ModernBertTokenizer,
)
from keras_hub.src.tests.test_case import TestCase

# Saves task weights with a known decoder bias. Run in a subprocess so the
# weights can be saved under a different backend than the test's.
SAVE_TASK_WEIGHTS_SCRIPT = """
import json
import sys

import numpy as np

import keras_hub

config, path = sys.argv[1:]
backbone = keras_hub.models.ModernBertBackbone.from_config(json.loads(config))
model = keras_hub.models.ModernBertMaskedLM(backbone=backbone)
bias = model.mlm_head_decoder_bias
bias.assign(np.linspace(-1.0, 1.0, bias.shape[0]).astype("float32"))
model.save_task_weights(path)
"""

# Seconds before a hung `SAVE_TASK_WEIGHTS_SCRIPT` subprocess is killed.
SAVE_TASK_WEIGHTS_TIMEOUT = 300


class ModernBertMaskedLMTest(TestCase):
    """Tests for verifying the ModernBERT MaskedLM task model."""

    def setUp(self):
        vocab = [
            "[PAD]",
            "[MASK]",
            "[CLS]",
            "[SEP]",
            "[UNK]",
            "Ġ",
            "a",
            "t",
            "i",
            "b",
            "p",
            "l",
            "n",
            "e",
            "o",
            "r",
            "h",
            "ai",
            "pl",
            "po",
            "rt",
            "th",
            "air",
            "pla",
            "ne",
            "port",
            "plane",
            "Ġa",
            "Ġt",
            "Ġi",
            "Ġb",
            "Ġat",
            "Ġair",
        ]

        vocab = {token: index for index, token in enumerate(vocab)}

        merges = [
            "Ġ a",
            "Ġ t",
            "a i",
            "p l",
            "n e",
            "p o",
            "r t",
            "t h",
            "ai r",
            "pl a",
            "po rt",
            "pla ne",
        ]

        self.tokenizer = ModernBertTokenizer(
            vocabulary=vocab,
            merges=merges,
        )

        self.preprocessor = (
            modern_bert_masked_lm_preprocessor.ModernBertMaskedLMPreprocessor
        )(
            tokenizer=self.tokenizer,
            sequence_length=12,
            mask_selection_rate=0.0,
            mask_selection_length=2,
        )

        self.backbone = ModernBertBackbone(
            vocabulary_size=self.tokenizer.vocabulary_size(),
            num_layers=2,
            num_heads=2,
            hidden_dim=16,
            intermediate_dim=32,
            local_attention_window=128,
        )

        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": self.preprocessor,
        }

        self.input_data = [
            "airplane airport",
            "airplane",
        ]

        self.train_data = (
            self.input_data,
            None,
            None,
        )

        self.model = ModernBertMaskedLM(
            **self.init_kwargs,
        )

    def test_masked_lm_basics(self):
        """Validate training, output shape, and serialization."""
        self.run_task_test(
            cls=ModernBertMaskedLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            # (batch, mask_selection_length, vocabulary_size)
            expected_output_shape=(2, 2, 33),
        )

    @pytest.mark.large
    def test_saved_model(self):
        """Validate serialization lifecycle."""
        input_data = self.preprocessor(self.input_data)[0]

        self.run_model_saving_test(
            cls=ModernBertMaskedLM,
            init_kwargs=self.init_kwargs,
            input_data=input_data,
        )

    def test_decoder_bias_is_trained_and_saved(self):
        """The decoder bias must be trained and saved on every backend.

        Regression test for keras-team/keras-hub#3210. On torch the bias was
        untracked, so it was never trained or saved.
        """
        model = ModernBertMaskedLM(backbone=self.backbone, preprocessor=None)
        bias = model.mlm_head_decoder_bias
        self.assertTrue(
            any(weight is bias for weight in model.trainable_weights)
        )

        # Mask tokens in every sample so the loss has a gradient.
        preprocessor = (
            modern_bert_masked_lm_preprocessor.ModernBertMaskedLMPreprocessor
        )(
            tokenizer=self.tokenizer,
            sequence_length=12,
            mask_selection_rate=1.0,
            mask_selection_length=2,
        )
        x, y, sample_weight = preprocessor(self.input_data)

        # One `fit()` step only moves the zero-initialized bias if the bias
        # is wired into the logits.
        model.fit(x, y, sample_weight=sample_weight)
        self.assertNotAllClose(bias, np.zeros(bias.shape))

        # Reload the task weights on the same backbone and compare logits.
        path = os.path.join(self.get_temp_dir(), "task.weights.h5")
        model.save_task_weights(path)
        restored = ModernBertMaskedLM(backbone=self.backbone, preprocessor=None)
        restored.load_task_weights(path)
        self.assertAllClose(restored.predict(x), model.predict(x))

    @parameterized.named_parameters(
        ("saved_on_jax", "jax"),
        ("saved_on_tensorflow", "tensorflow"),
        ("saved_on_torch", "torch"),
    )
    @pytest.mark.large
    def test_task_weights_load_across_backends(self, saving_backend):
        """Task weights saved under another backend must load on this one.

        Regression test for keras-team/keras-hub#3210. Presets saved on jax
        failed to load on torch, which same-backend round trips cannot catch.
        """
        if saving_backend == keras.config.backend():
            self.skipTest("Requires a saving backend other than the current.")
        if importlib.util.find_spec(saving_backend) is None:
            self.skipTest(f"Requires the {saving_backend} backend.")

        config = self.backbone.get_config()
        path = os.path.join(self.get_temp_dir(), "task.weights.h5")
        # Make the subprocess import the source tree under test rather than
        # any installed release of `keras_hub`.
        repo_root = os.path.dirname(os.path.dirname(keras_hub.__file__))
        python_path = os.pathsep.join(
            filter(None, [repo_root, os.environ.get("PYTHONPATH")])
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                SAVE_TASK_WEIGHTS_SCRIPT,
                json.dumps(config),
                path,
            ],
            env={
                **os.environ,
                "KERAS_BACKEND": saving_backend,
                "PYTHONPATH": python_path,
            },
            check=True,
            timeout=SAVE_TASK_WEIGHTS_TIMEOUT,
        )

        model = ModernBertMaskedLM(
            backbone=ModernBertBackbone.from_config(config),
            preprocessor=None,
        )
        model.load_task_weights(path)
        bias = model.mlm_head_decoder_bias
        self.assertAllClose(bias, np.linspace(-1.0, 1.0, bias.shape[0]))
