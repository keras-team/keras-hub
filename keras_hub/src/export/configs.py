"""Configuration classes for different Keras-Hub model types.

This module provides specific configurations for exporting different types
of Keras-Hub models, following the Optimum pattern.
"""

import keras
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.export.base import KerasHubExporterConfig


@keras_hub_export("keras_hub.export.CausalLMExporterConfig")
class CausalLMExporterConfig(KerasHubExporterConfig):
    """Exporter configuration for Causal Language Models (GPT, LLaMA, etc.)."""

    MODEL_TYPE = "causal_lm"
    EXPECTED_INPUTS = ["token_ids", "padding_mask"]
    DEFAULT_SEQUENCE_LENGTH = 128

    def _is_model_compatible(self):
        """Check if model is a causal language model.

        Returns:
            bool: True if compatible, False otherwise
        """
        try:
            from keras_hub.src.models.causal_lm import CausalLM

            return isinstance(self.model, CausalLM)
        except ImportError:
            # Fallback to class name checking
            return "CausalLM" in self.model.__class__.__name__

    def get_input_signature(self, sequence_length=None):
        """Get input signature for causal LM models.

        Args:
            sequence_length: Optional sequence length. If None, will be inferred
                from model.

        Returns:
            Dict[str, Any]: Dictionary mapping input names to their
            specifications
        """
        if sequence_length is None:
            sequence_length = self._get_sequence_length()

        return {
            "token_ids": keras.layers.InputSpec(
                shape=(None, sequence_length), dtype="int32", name="token_ids"
            ),
            "padding_mask": keras.layers.InputSpec(
                shape=(None, sequence_length), dtype="bool", name="padding_mask"
            ),
        }

    def _get_sequence_length(self):
        """Get sequence length from model or use default.

        Returns:
            int: The sequence length
        """
        if hasattr(self.model, "preprocessor") and self.model.preprocessor:
            return getattr(
                self.model.preprocessor,
                "sequence_length",
                self.DEFAULT_SEQUENCE_LENGTH,
            )
        return self.DEFAULT_SEQUENCE_LENGTH


@keras_hub_export("keras_hub.export.TextClassifierExporterConfig")
class TextClassifierExporterConfig(KerasHubExporterConfig):
    """Exporter configuration for Text Classification models."""

    MODEL_TYPE = "text_classifier"
    EXPECTED_INPUTS = ["token_ids", "padding_mask"]
    DEFAULT_SEQUENCE_LENGTH = 128

    def _is_model_compatible(self):
        """Check if model is a text classifier.

        Returns:
            bool: True if compatible, False otherwise
        """
        return "TextClassifier" in self.model.__class__.__name__

    def get_input_signature(self, sequence_length=None):
        """Get input signature for text classifier models.

        Args:
            sequence_length: Optional sequence length. If None, will be inferred
                from model.

        Returns:
            Dict[str, Any]: Dictionary mapping input names to their
            specifications
        """
        if sequence_length is None:
            sequence_length = self._get_sequence_length()

        return {
            "token_ids": keras.layers.InputSpec(
                shape=(None, sequence_length), dtype="int32", name="token_ids"
            ),
            "padding_mask": keras.layers.InputSpec(
                shape=(None, sequence_length), dtype="bool", name="padding_mask"
            ),
        }

    def _get_sequence_length(self):
        """Get sequence length from model or use default.

        Returns:
            int: The sequence length
        """
        if hasattr(self.model, "preprocessor") and self.model.preprocessor:
            return getattr(
                self.model.preprocessor,
                "sequence_length",
                self.DEFAULT_SEQUENCE_LENGTH,
            )
        return self.DEFAULT_SEQUENCE_LENGTH


@keras_hub_export("keras_hub.export.Seq2SeqLMExporterConfig")
class Seq2SeqLMExporterConfig(KerasHubExporterConfig):
    """Exporter configuration for Sequence-to-Sequence Language Models."""

    MODEL_TYPE = "seq2seq_lm"
    EXPECTED_INPUTS = [
        "encoder_token_ids",
        "encoder_padding_mask",
        "decoder_token_ids",
        "decoder_padding_mask",
    ]
    DEFAULT_SEQUENCE_LENGTH = 128

    def _is_model_compatible(self):
        """Check if model is a seq2seq language model.

        Returns:
            bool: True if compatible, False otherwise
        """
        try:
            from keras_hub.src.models.seq_2_seq_lm import Seq2SeqLM

            return isinstance(self.model, Seq2SeqLM)
        except ImportError:
            return "Seq2SeqLM" in self.model.__class__.__name__

    def get_input_signature(self, sequence_length=None):
        """Get input signature for seq2seq models.

        Args:
            sequence_length: Optional sequence length. If None, will be inferred
                from model.

        Returns:
            Dict[str, Any]: Dictionary mapping input names to their
            specifications
        """
        if sequence_length is None:
            sequence_length = self._get_sequence_length()

        return {
            "encoder_token_ids": keras.layers.InputSpec(
                shape=(None, sequence_length),
                dtype="int32",
                name="encoder_token_ids",
            ),
            "encoder_padding_mask": keras.layers.InputSpec(
                shape=(None, sequence_length),
                dtype="bool",
                name="encoder_padding_mask",
            ),
            "decoder_token_ids": keras.layers.InputSpec(
                shape=(None, sequence_length),
                dtype="int32",
                name="decoder_token_ids",
            ),
            "decoder_padding_mask": keras.layers.InputSpec(
                shape=(None, sequence_length),
                dtype="bool",
                name="decoder_padding_mask",
            ),
        }

    def _get_sequence_length(self):
        """Get sequence length from model or use default.

        Returns:
            int: The sequence length
        """
        if hasattr(self.model, "preprocessor") and self.model.preprocessor:
            return getattr(
                self.model.preprocessor,
                "sequence_length",
                self.DEFAULT_SEQUENCE_LENGTH,
            )
        return self.DEFAULT_SEQUENCE_LENGTH


@keras_hub_export("keras_hub.export.TextModelExporterConfig")
class TextModelExporterConfig(KerasHubExporterConfig):
    """Generic exporter configuration for text models."""

    MODEL_TYPE = "text_model"
    EXPECTED_INPUTS = ["token_ids", "padding_mask"]
    DEFAULT_SEQUENCE_LENGTH = 128

    def _is_model_compatible(self):
        """Check if model is a text model (fallback).

        Returns:
            bool: True if compatible, False otherwise
        """
        # This is a fallback config for text models that don't fit other
        # categories
        return (
            hasattr(self.model, "preprocessor")
            and self.model.preprocessor
            and hasattr(self.model.preprocessor, "tokenizer")
        )

    def get_input_signature(self, sequence_length=None):
        """Get input signature for generic text models.

        Args:
            sequence_length: Optional sequence length. If None, will be inferred
                from model.

        Returns:
            Dict[str, Any]: Dictionary mapping input names to their
            specifications
        """
        if sequence_length is None:
            sequence_length = self._get_sequence_length()

        return {
            "token_ids": keras.layers.InputSpec(
                shape=(None, sequence_length), dtype="int32", name="token_ids"
            ),
            "padding_mask": keras.layers.InputSpec(
                shape=(None, sequence_length), dtype="bool", name="padding_mask"
            ),
        }

    def _get_sequence_length(self):
        """Get sequence length from model or use default.

        Returns:
            int: The sequence length
        """
        if hasattr(self.model, "preprocessor") and self.model.preprocessor:
            return getattr(
                self.model.preprocessor,
                "sequence_length",
                self.DEFAULT_SEQUENCE_LENGTH,
            )
        return self.DEFAULT_SEQUENCE_LENGTH


@keras_hub_export("keras_hub.export.ImageClassifierExporterConfig")
class ImageClassifierExporterConfig(KerasHubExporterConfig):
    """Exporter configuration for Image Classification models."""

    MODEL_TYPE = "image_classifier"
    EXPECTED_INPUTS = ["images"]

    def _is_model_compatible(self):
        """Check if model is an image classifier.
        Returns:
            bool: True if compatible, False otherwise
        """
        return "ImageClassifier" in self.model.__class__.__name__

    def get_input_signature(self, image_size=None):
        """Get input signature for image classifier models.
        Args:
            image_size: Optional image size. If None, will be inferred
                from model.
        Returns:
            Dict[str, Any]: Dictionary mapping input names to their
            specifications
        """
        if image_size is None:
            image_size = self._get_image_size()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        return {
            "images": keras.layers.InputSpec(
                shape=(None, *image_size, 3), dtype=self._get_input_dtype(), name="images"
            ),
        }

    def _get_input_dtype(self):
        """Get input dtype from model.
        Returns:
            str: The input dtype (e.g., 'float32', 'float16')
        """
        if hasattr(self.model, "inputs") and self.model.inputs:
            return str(self.model.inputs[0].dtype)
        # Default fallback
        return "float32"

    def _get_image_size(self):
        """Get image size from model preprocessor.
        Returns:
            tuple: The image size (height, width)
        """
        if hasattr(self.model, "preprocessor") and self.model.preprocessor:
            if hasattr(self.model.preprocessor, "image_size"):
                return self.model.preprocessor.image_size

        # If no preprocessor image_size, try to infer from model inputs
        if hasattr(self.model, "inputs") and self.model.inputs:
            input_shape = self.model.inputs[0].shape
            if (
                len(input_shape) == 4
                and input_shape[1] is not None
                and input_shape[2] is not None
            ):
                # Shape is (batch, height, width, channels)
                return (input_shape[1], input_shape[2])

        # Last resort: raise an error instead of using hardcoded values
        raise ValueError(
            "Could not determine image size from model. "
            "Model should have a preprocessor with image_size attribute, "
            "or model inputs should have concrete shapes."
        )

    def get_dummy_inputs(self, image_size=None):
        """Generate dummy inputs for image classifier models.

        Args:
            image_size: Optional image size. If None, will be inferred from
                model.

        Returns:
            Dict[str, Any]: Dictionary of dummy inputs
        """
        if image_size is None:
            image_size = self._get_image_size()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        dummy_inputs = {}
        if "images" in self.EXPECTED_INPUTS:
            dummy_inputs["images"] = keras.ops.ones(
                (1, *image_size, 3), dtype="float32"
            )

        return dummy_inputs


@keras_hub_export("keras_hub.export.ObjectDetectorExporterConfig")
class ObjectDetectorExporterConfig(KerasHubExporterConfig):
    """Exporter configuration for Object Detection models."""

    MODEL_TYPE = "object_detector"
    EXPECTED_INPUTS = ["images", "image_shape"]

    def _is_model_compatible(self):
        """Check if model is an object detector.
        Returns:
            bool: True if compatible, False otherwise
        """
        return "ObjectDetector" in self.model.__class__.__name__

    def get_input_signature(self, image_size=None):
        """Get input signature for object detector models.
        Args:
            image_size: Optional image size. If None, will be inferred
                from model.
        Returns:
            Dict[str, Any]: Dictionary mapping input names to their
            specifications
        """
        if image_size is None:
            image_size = self._get_image_size()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        return {
            "images": keras.layers.InputSpec(
                shape=(None, *image_size, 3), dtype=self._get_input_dtype(), name="images"
            ),
            "image_shape": keras.layers.InputSpec(
                shape=(None, 2), dtype="int32", name="image_shape"
            ),
        }

    def _get_input_dtype(self):
        """Get input dtype from model.
        Returns:
            str: The input dtype (e.g., 'float32', 'float16')
        """
        if hasattr(self.model, "inputs") and self.model.inputs:
            return str(self.model.inputs[0].dtype)
        # Default fallback
        return "float32"

    def _get_image_size(self):
        """Get image size from model preprocessor.
        Returns:
            tuple: The image size (height, width)
        """
        if hasattr(self.model, "preprocessor") and self.model.preprocessor:
            if hasattr(self.model.preprocessor, "image_size"):
                return self.model.preprocessor.image_size

        # If no preprocessor image_size, try to infer from model inputs
        if hasattr(self.model, "inputs") and self.model.inputs:
            input_shape = self.model.inputs[0].shape
            if (
                len(input_shape) == 4
                and input_shape[1] is not None
                and input_shape[2] is not None
            ):
                # Shape is (batch, height, width, channels)
                return (input_shape[1], input_shape[2])

        # Last resort: raise an error instead of using hardcoded values
        raise ValueError(
            "Could not determine image size from model. "
            "Model should have a preprocessor with image_size attribute, "
            "or model inputs should have concrete shapes."
        )

    def get_dummy_inputs(self, image_size=None):
        """Generate dummy inputs for object detector models.

        Args:
            image_size: Optional image size. If None, will be inferred
                from model.

        Returns:
            Dict[str, Any]: Dictionary of dummy inputs
        """
        if image_size is None:
            image_size = self._get_image_size()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        dummy_inputs = {}

        # Create dummy image input
        dummy_inputs["images"] = keras.ops.random_uniform(
            (1, *image_size, 3), dtype="float32"
        )

        # Create dummy image shape input
        dummy_inputs["image_shape"] = keras.ops.constant(
            [[image_size[0], image_size[1]]], dtype="int32"
        )

        return dummy_inputs


@keras_hub_export("keras_hub.export.ImageSegmenterExporterConfig")
class ImageSegmenterExporterConfig(KerasHubExporterConfig):
    """Exporter configuration for Image Segmentation models."""

    MODEL_TYPE = "image_segmenter"
    EXPECTED_INPUTS = ["images"]

    def _is_model_compatible(self):
        """Check if model is an image segmenter.
        Returns:
            bool: True if compatible, False otherwise
        """
        return "ImageSegmenter" in self.model.__class__.__name__

    def get_input_signature(self, image_size=None):
        """Get input signature for image segmenter models.
        Args:
            image_size: Optional image size. If None, will be inferred
                from model.
        Returns:
            Dict[str, Any]: Dictionary mapping input names to their
            specifications
        """
        if image_size is None:
            image_size = self._get_image_size()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        return {
            "images": keras.layers.InputSpec(
                shape=(None, *image_size, 3), dtype=self._get_input_dtype(), name="images"
            ),
        }

    def _get_input_dtype(self):
        """Get input dtype from model.
        Returns:
            str: The input dtype (e.g., 'float32', 'float16')
        """
        if hasattr(self.model, "inputs") and self.model.inputs:
            return str(self.model.inputs[0].dtype)
        # Default fallback
        return "float32"

    def _get_image_size(self):
        """Get image size from model preprocessor.
        Returns:
            tuple: The image size (height, width)
        """
        if hasattr(self.model, "preprocessor") and self.model.preprocessor:
            if hasattr(self.model.preprocessor, "image_size"):
                return self.model.preprocessor.image_size

        # If no preprocessor image_size, try to infer from model inputs
        if hasattr(self.model, "inputs") and self.model.inputs:
            input_shape = self.model.inputs[0].shape
            if (
                len(input_shape) == 4
                and input_shape[1] is not None
                and input_shape[2] is not None
            ):
                # Shape is (batch, height, width, channels)
                return (input_shape[1], input_shape[2])

        # Last resort: raise an error instead of using hardcoded values
        raise ValueError(
            "Could not determine image size from model. "
            "Model should have a preprocessor with image_size attribute, "
            "or model inputs should have concrete shapes."
        )
