import unittest

import keras
import numpy as np

import keras_hub


class NoTensorflow(unittest.TestCase):
    def test_backbone_works(self):
        backbone = keras_hub.models.BertBackbone.from_preset(
            "bert_tiny_en_uncased",
        )
        backbone.predict(
            {
                "token_ids": np.ones((4, 128)),
                "padding_mask": np.ones((4, 128)),
                "segment_ids": np.ones((4, 128)),
            }
        )

    def test_tokenizer_works(self):
        # `WordPieceTokenizer` has a pure Python path, so `BertTokenizer` no
        # longer requires tensorflow-text. This previously asserted that
        # `from_preset` raised "pip install tensorflow-text".
        tokenizer = keras_hub.models.BertTokenizer.from_preset(
            "bert_tiny_en_uncased",
        )
        outputs = np.array(tokenizer("the quick brown fox"))
        self.assertEqual(outputs.ndim, 1)
        self.assertGreater(outputs.shape[0], 0)
        self.assertEqual(outputs.dtype, np.int32)
        # Round trip: detokenize and verify we recover the original text.
        decoded = tokenizer.detokenize(outputs)
        self.assertEqual(decoded, "the quick brown fox")

    def test_preprocessor_works(self):
        # Preprocessing layers default to the pure Python path, so model
        # preprocessors can be built and run without tensorflow-text.
        preprocessor = (
            keras_hub.models.BertTextClassifierPreprocessor.from_preset(
                "bert_tiny_en_uncased",
                sequence_length=8,
            )
        )
        x, y = preprocessor(["the quick brown fox"], [1])
        self.assertEqual(np.shape(x["token_ids"]), (1, 8))
        self.assertEqual(np.shape(x["padding_mask"]), (1, 8))
        self.assertEqual(np.shape(x["segment_ids"]), (1, 8))
        self.assertEqual(np.asarray(y).tolist(), [1])

    def test_tf_workflow_errors(self):
        # Explicitly requesting the TensorFlow path raises an informative
        # error on first use rather than at construction time.
        tokenizer = keras_hub.models.BertTokenizer.from_preset(
            "bert_tiny_en_uncased",
            _allow_python_workflow=False,
        )
        with self.assertRaisesRegex(ImportError, "pip install tensorflow-text"):
            tokenizer("the quick brown fox")

    def test_image_converter_works(self):
        # `Gemma4ImageConverter` has `keras.ops` branches for every TensorFlow
        # op, guarded by `in_tf_function()`, so it runs without TensorFlow.
        if keras.config.backend() == "openvino":
            self.skipTest("Image ops are not supported on OpenVINO.")
        converter = keras_hub.layers.Gemma4ImageConverter(
            patch_size=4, max_soft_tokens=1
        )
        outputs = converter(np.ones((1, 12, 12, 3), dtype="float32"))
        self.assertIn("pixel_values", outputs)
        self.assertIn("pixel_position_ids", outputs)

    def test_muse_glimmer_generate_preprocess_works(self):
        # The `generate()` path of the MuseGlimmer preprocessor builds the
        # vision inputs without TensorFlow.
        if keras.config.backend() == "openvino":
            self.skipTest("Image ops are not supported on OpenVINO.")
        merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        merges += ["Ġai r", "Ġa i", "pla ne"]
        vocab = []
        for merge in merges:
            a, b = merge.split(" ")
            vocab.extend([a, b, a + b])
        vocab += ["<|end_of_text|>", "<|begin_of_text|>", "<|eot|>"]
        vocab += ["<|finetune_right_pad|>", "<|image|>"]
        vocab += ["<|image_start|>", "<|image_end|>", "<|vid_start|>"]
        vocab += ["<|vid_end|>", "<|vid_frame_separator|>"]
        vocab = {token: i for i, token in enumerate(sorted(set(vocab)))}
        tokenizer = keras_hub.models.MuseGlimmerTokenizer(
            vocabulary=vocab,
            merges=merges,
            image_token_id=vocab["<|image|>"],
            unsplittable_tokens=["<|image|>"],
        )
        converter = keras_hub.layers.MuseGlimmerImageConverter(
            patch_size=4,
            patch_temporal=2,
            merge_size=2,
            max_image_tokens=64,
            image_mean=0.0,
            image_std=255.0,
        )
        preprocessor = keras_hub.models.MuseGlimmerCausalLMPreprocessor(
            tokenizer=tokenizer,
            image_converter=converter,
            sequence_length=8,
        )
        outputs = preprocessor.generate_preprocess(
            {
                "prompts": "<|image|> airplane",
                "images": np.ones((8, 8, 3), dtype="float32"),
            }
        )
        # An 8x8 image gives 4 patches of `2 * 3 * 4**2` values and one
        # merged vision token.
        self.assertEqual(np.shape(outputs["pixel_values"]), (4, 96))
        self.assertEqual(np.shape(outputs["image_grid_thw"]), (1, 3))
        self.assertEqual(np.shape(outputs["vision_indices"]), (1,))

    def test_muse_glimmer_video_generate_preprocess_works(self):
        # The video layout (frame sampling, timestamps and separators) is
        # also built without TensorFlow.
        if keras.config.backend() == "openvino":
            self.skipTest("Image ops are not supported on OpenVINO.")
        merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        merges += ["Ġai r", "Ġa i", "pla ne"]
        vocab = []
        for merge in merges:
            a, b = merge.split(" ")
            vocab.extend([a, b, a + b])
        vocab += list("Time:0123456789.s")
        vocab += ["<|end_of_text|>", "<|begin_of_text|>", "<|eot|>"]
        vocab += ["<|finetune_right_pad|>", "<|video|>"]
        vocab += ["<|image_start|>", "<|image_end|>", "<|vid_start|>"]
        vocab += ["<|vid_end|>", "<|vid_frame_separator|>"]
        vocab = {token: i for i, token in enumerate(sorted(set(vocab)))}
        tokenizer = keras_hub.models.MuseGlimmerTokenizer(
            vocabulary=vocab,
            merges=merges,
            video_token_id=vocab["<|video|>"],
            unsplittable_tokens=["<|video|>"],
        )
        converter = keras_hub.layers.MuseGlimmerVideoConverter(
            patch_size=4,
            patch_temporal=2,
            merge_size=2,
            fps=2.0,
            num_frames=4,
            max_video_frame_tokens=16,
            image_mean=0.0,
            image_std=255.0,
        )
        preprocessor = keras_hub.models.MuseGlimmerCausalLMPreprocessor(
            tokenizer=tokenizer,
            video_converter=converter,
            sequence_length=64,
        )
        outputs = preprocessor.generate_preprocess(
            {
                "prompts": "<|video|> airplane",
                "videos": np.ones((4, 8, 8, 3), dtype="float32"),
                "video_fps": 2.0,
            }
        )
        # 4 frames give 2 temporal groups with one merged token each.
        self.assertEqual(np.shape(outputs["pixel_values"]), (8, 96))
        self.assertEqual(np.shape(outputs["image_grid_thw"]), (1, 3))
        self.assertEqual(np.shape(outputs["vision_indices"]), (2,))

    def test_seq_2_seq_generate_preprocess_works(self):
        # Without `decoder_text`, the empty decoder prompt is built in Python.
        merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        merges += ["Ġai r", "Ġa i", "pla ne"]
        vocab = []
        for merge in merges:
            a, b = merge.split(" ")
            vocab.extend([a, b, a + b])
        vocab += ["<s>", "<pad>", "</s>", "<mask>"]
        vocab = {token: i for i, token in enumerate(sorted(set(vocab)))}
        tokenizer = keras_hub.models.BartTokenizer(
            vocabulary=vocab, merges=merges
        )
        preprocessor = keras_hub.models.BartSeq2SeqLMPreprocessor(
            tokenizer=tokenizer,
            encoder_sequence_length=5,
            decoder_sequence_length=8,
        )
        batched = preprocessor.generate_preprocess([" airplane at airport"])
        self.assertEqual(np.shape(batched["decoder_token_ids"]), (1, 8))
        unbatched = preprocessor.generate_preprocess(" airplane at airport")
        self.assertEqual(np.shape(unbatched["decoder_token_ids"]), (8,))

    def test_tf_only_layer_errors(self):
        # Layers that can only run on the TensorFlow path raise an informative
        # error at construction rather than failing on first call.
        def skip_fn(word):
            return word

        with self.assertRaisesRegex(ImportError, "requires `tensorflow`"):
            keras_hub.layers.RandomDeletion(
                rate=0.5, max_deletions=1, skip_fn=skip_fn
            )
