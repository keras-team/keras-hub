import os

import numpy as np
from sentencepiece import SentencePieceTrainer
from transformers import GemmaForCausalLM as HFGemmaForCausalLM
from transformers import GemmaModel as HFGemmaModel
from transformers import GemmaTokenizer as HFGemmaTokenizer

from keras_hub.src.models.gemma.gemma_backbone import GemmaBackbone
from keras_hub.src.models.gemma.gemma_causal_lm import GemmaCausalLM
from keras_hub.src.models.gemma.gemma_causal_lm_preprocessor import (
    GemmaCausalLMPreprocessor,
)
from keras_hub.src.models.gemma.gemma_tokenizer import GemmaTokenizer
from keras_hub.src.tests.test_case import TestCase


class TestGemmaExport(TestCase):
    def test_export_to_hf(self):
        # Create a dummy tokenizer
        train_sentences = [
            "The quick brown fox jumped.",
            "I like pizza.",
            "This is a test.",
        ]
        # TODO:Consider using keras_hub/src/tests/test_data/gemma_test_vocab.spm
        # instead of retraining a new vocab here. Will be faster.
        proto_prefix = os.path.join(self.get_temp_dir(), "dummy_vocab")
        SentencePieceTrainer.train(
            sentence_iterator=iter(train_sentences),
            model_prefix=proto_prefix,
            vocab_size=290,
            model_type="unigram",
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
            byte_fallback=True,
            pad_piece="<pad>",
            bos_piece="<bos>",
            eos_piece="<eos>",
            unk_piece="<unk>",
            user_defined_symbols=["<start_of_turn>", "<end_of_turn>"],
        )
        tokenizer = GemmaTokenizer(proto=f"{proto_prefix}.model")

        # Create a small backbone
        backbone = GemmaBackbone(
            vocabulary_size=tokenizer.vocabulary_size(),
            num_layers=2,
            num_query_heads=4,
            num_key_value_heads=1,
            hidden_dim=512,
            intermediate_dim=1028,
            head_dim=128,
        )
        # Create preprocessor
        preprocessor = GemmaCausalLMPreprocessor(tokenizer=tokenizer)

        # Create the causal LM model
        keras_model = GemmaCausalLM(
            backbone=backbone, preprocessor=preprocessor
        )

        # Set all weights to random values
        rng = np.random.default_rng(42)
        weights = keras_model.get_weights()
        for i in range(len(weights)):
            weights[i] = rng.random(weights[i].shape).astype(weights[i].dtype)
        keras_model.set_weights(weights)

        # Export to Hugging Face format using the new methods
        export_path_backbone = os.path.join(
            self.get_temp_dir(), "export_backbone"
        )
        backbone.export_to_transformers(export_path_backbone)

        export_path_tokenizer = os.path.join(
            self.get_temp_dir(), "export_tokenizer"
        )
        preprocessor.tokenizer.export_to_transformers(export_path_tokenizer)

        export_path_task = os.path.join(self.get_temp_dir(), "export_task")
        keras_model.export_to_transformers(export_path_task)

        # Load Hugging Face models and tokenizer
        hf_backbone = HFGemmaModel.from_pretrained(export_path_backbone)
        hf_tokenizer = HFGemmaTokenizer.from_pretrained(export_path_tokenizer)
        hf_full_model = HFGemmaForCausalLM.from_pretrained(export_path_task)

        # Verify configuration
        hf_config = hf_backbone.config
        self.assertEqual(
            hf_config.vocab_size,
            backbone.vocabulary_size,
            "Vocabulary sizes do not match",
        )
        self.assertEqual(
            hf_config.num_hidden_layers,
            backbone.num_layers,
            "Number of layers do not match",
        )
        self.assertEqual(
            hf_config.num_attention_heads,
            backbone.num_query_heads,
            "Number of query heads do not match",
        )
        self.assertEqual(
            hf_config.num_key_value_heads,
            backbone.num_key_value_heads,
            "Number of key value heads do not match",
        )
        self.assertEqual(
            hf_config.hidden_size,
            backbone.hidden_dim,
            "Hidden dimensions do not match",
        )
        self.assertEqual(
            hf_config.intermediate_size,
            backbone.intermediate_dim // 2,
            "Intermediate sizes do not match",
        )
        self.assertEqual(
            hf_config.head_dim,
            backbone.head_dim,
            "Head dimensions do not match",
        )
        self.assertEqual(
            hf_config.max_position_embeddings,
            8192,
            "Max position embeddings do not match",
        )

        # Verify tokenizer compatibility
        self.assertEqual(
            hf_tokenizer.vocab_size,
            tokenizer.vocabulary_size(),
            "Tokenizer vocabulary sizes do not match",
        )

        # Compare generated outputs using full model
        prompt = "the quick"
        keras_output = keras_model.generate(prompt, max_length=20)
        input_ids = hf_tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = hf_full_model.generate(
                input_ids, max_length=20, do_sample=False
            )
        hf_output = hf_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        self.assertEqual(
            keras_output, hf_output, "Generated outputs do not match"
        )
