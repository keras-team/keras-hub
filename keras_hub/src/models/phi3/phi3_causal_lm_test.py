import os

import pytest
from keras import ops

from keras_hub.src.models.phi3.phi3_backbone import Phi3Backbone
from keras_hub.src.models.phi3.phi3_causal_lm import Phi3CausalLM
from keras_hub.src.models.phi3.phi3_causal_lm_preprocessor import (
    Phi3CausalLMPreprocessor,
)
from keras_hub.src.models.phi3.phi3_tokenizer import Phi3Tokenizer
from keras_hub.src.tests.test_case import TestCase


class Phi3CausalLMTest(TestCase):
    def setUp(self):
        self.preprocessor = Phi3CausalLMPreprocessor(
            Phi3Tokenizer(
                # Generated using create_phi3_test_proto.py
                proto=os.path.join(
                    self.get_test_data_dir(), "phi3_test_vocab.spm"
                )
            ),
            sequence_length=12,
        )
        self.vocab_size = self.preprocessor.tokenizer.vocabulary_size()
        self.backbone = Phi3Backbone(
            vocabulary_size=self.vocab_size,
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }
        self.train_data = (["the quick brown fox", "the earth is round"],)
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_causal_lm_basics(self):
        self.run_task_test(
            cls=Phi3CausalLM,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 12, self.vocab_size),
        )

    def test_generate(self):
        self.run_generate_test(
            cls=Phi3CausalLM,
            init_kwargs=self.init_kwargs,
            prompt="the fox",
        )

    def test_early_stopping(self):
        self.run_early_stopping_test(
            cls=Phi3CausalLM,
            init_kwargs=self.init_kwargs,
            prompt=["the fox", "the earth"],
        )

    def test_generate_compilation(self):
        self.run_generate_compilation_test(
            cls=Phi3CausalLM,
            init_kwargs=self.init_kwargs,
            prompt="the fox",
        )

    # @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Phi3CausalLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        """Test LiteRT export for Phi3CausalLM with small test model."""
        model = Phi3CausalLM(**self.init_kwargs)

        # Convert boolean padding_mask to int32 for LiteRT compatibility
        input_data = self.input_data.copy()
        if "padding_mask" in input_data:
            input_data["padding_mask"] = ops.cast(
                input_data["padding_mask"], "int32"
            )

        expected_output_shape = (
            2,
            12,
            self.preprocessor.tokenizer.vocabulary_size(),
        )

        self.run_litert_export_test(
            model=model,
            input_data=input_data,
            expected_output_shape=expected_output_shape,
            comparison_mode="statistical",
            output_thresholds={"*": {"max": 1e-3, "mean": 1e-5}},
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Phi3CausalLM.presets:
            self.run_preset_test(
                cls=Phi3CausalLM,
                preset=preset,
                input_data=self.input_data,
            )

    def test_get_quantization_layer_structure(self):
        causal_lm = Phi3CausalLM(**self.init_kwargs)
        structure = causal_lm.get_quantization_layer_structure("gptq")
        self.assertIsInstance(structure, dict)
        self.assertIn("pre_block_layers", structure)
        self.assertIn("sequential_blocks", structure)
        self.assertEqual(
            structure["pre_block_layers"],
            [self.backbone.token_embedding],
        )
        self.assertEqual(
            structure["sequential_blocks"], self.backbone.transformer_layers
        )

        self.assertIsNone(causal_lm.get_quantization_layer_structure("int8"))
