"""Configuration classes for different Keras-Hub model types.

This module provides specific configurations for exporting different types
of Keras-Hub models, following the Optimum pattern.
"""

import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.export.base import KerasHubExporterConfig
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.image_segmenter import ImageSegmenter
from keras_hub.src.models.object_detector import ObjectDetector
from keras_hub.src.models.seq_2_seq_lm import Seq2SeqLM
from keras_hub.src.models.text_classifier import TextClassifier


def _get_text_input_signature(model, sequence_length=128):
    """Get input signature for text models with token_ids and padding_mask.

    Args:
        model: The model instance.
        sequence_length: `int`. Sequence length (default: 128).

    Returns:
        `dict`. Dictionary mapping input names to their specifications
    """
    return {
        "token_ids": keras.layers.InputSpec(
            shape=(None, sequence_length), dtype="int32", name="token_ids"
        ),
        "padding_mask": keras.layers.InputSpec(
            shape=(None, sequence_length),
            dtype="int32",
            name="padding_mask",
        ),
    }


def _get_seq2seq_input_signature(model, sequence_length=128):
    """Get input signature for seq2seq models with encoder/decoder tokens.

    Args:
        model: The model instance.
        sequence_length: `int`. Sequence length (default: 128).

    Returns:
        `dict`. Dictionary mapping input names to their specifications
    """
    return {
        "encoder_token_ids": keras.layers.InputSpec(
            shape=(None, sequence_length),
            dtype="int32",
            name="encoder_token_ids",
        ),
        "encoder_padding_mask": keras.layers.InputSpec(
            shape=(None, sequence_length),
            dtype="int32",
            name="encoder_padding_mask",
        ),
        "decoder_token_ids": keras.layers.InputSpec(
            shape=(None, sequence_length),
            dtype="int32",
            name="decoder_token_ids",
        ),
        "decoder_padding_mask": keras.layers.InputSpec(
            shape=(None, sequence_length),
            dtype="int32",
            name="decoder_padding_mask",
        ),
    }


def _infer_image_size(model):
    """Infer image size from model preprocessor or inputs.

    Args:
        model: The model instance.

    Returns:
        `tuple`. Image size as (height, width).

    Raises:
        ValueError: If image_size cannot be determined.
    """
    image_size = None

    # Get from preprocessor
    if hasattr(model, "preprocessor") and model.preprocessor:
        if hasattr(model.preprocessor, "image_size"):
            image_size = model.preprocessor.image_size

    # Try to infer from model inputs
    if (
        image_size is None
        and hasattr(model, "inputs")
        and model.inputs
    ):
        input_shape = model.inputs[0].shape
        if (
            len(input_shape) == 4
            and input_shape[1] is not None
            and input_shape[2] is not None
        ):
            image_size = (input_shape[1], input_shape[2])

    if image_size is None:
        raise ValueError(
            "Could not determine image size from model. "
            "Model should have a preprocessor with image_size "
            "attribute, or model inputs should have concrete shapes."
        )

    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    return image_size


@keras_hub_export("keras_hub.export.CausalLMExporterConfig")
class CausalLMExporterConfig(KerasHubExporterConfig):
    """Exporter configuration for Causal Language Models (GPT, LLaMA, etc.)."""

    MODEL_TYPE = "causal_lm"
    EXPECTED_INPUTS = ["token_ids", "padding_mask"]
    DEFAULT_SEQUENCE_LENGTH = 128

    def _is_model_compatible(self):
        """Check if model is a causal language model.

        Returns:
            `bool`. True if compatible, False otherwise
        """
        return isinstance(self.model, CausalLM)

    def get_input_signature(self, sequence_length=None):
        """Get input signature for causal LM models.

        Args:
            sequence_length: `int` or `None`. Optional sequence length.

        Returns:
            `dict`. Dictionary mapping input names to their specifications
        """
        if sequence_length is None:
            if hasattr(self.model, "preprocessor") and self.model.preprocessor:
                sequence_length = getattr(
                    self.model.preprocessor,
                    "sequence_length",
                    self.DEFAULT_SEQUENCE_LENGTH,
                )
            else:
                sequence_length = self.DEFAULT_SEQUENCE_LENGTH

        return _get_text_input_signature(self.model, sequence_length)


@keras_hub_export("keras_hub.export.TextClassifierExporterConfig")
class TextClassifierExporterConfig(KerasHubExporterConfig):
    """Exporter configuration for Text Classification models."""

    MODEL_TYPE = "text_classifier"
    EXPECTED_INPUTS = ["token_ids", "padding_mask"]
    DEFAULT_SEQUENCE_LENGTH = 128

    def _is_model_compatible(self):
        """Check if model is an image classifier.

        Returns:
            `bool`. True if compatible, False otherwise
        """
        return isinstance(self.model, TextClassifier)

    def get_input_signature(self, sequence_length=None):
        """Get input signature for text classifier models.

        Args:
            sequence_length: `int` or `None`. Optional sequence length.

        Returns:
            `dict`. Dictionary mapping input names to their specifications
        """
        if sequence_length is None:
            if hasattr(self.model, "preprocessor") and self.model.preprocessor:
                sequence_length = getattr(
                    self.model.preprocessor,
                    "sequence_length",
                    self.DEFAULT_SEQUENCE_LENGTH,
                )
            else:
                sequence_length = self.DEFAULT_SEQUENCE_LENGTH

        return _get_text_input_signature(self.model, sequence_length)


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
            `bool`. True if compatible, False otherwise
        """
        return isinstance(self.model, Seq2SeqLM)

    def get_input_signature(self, sequence_length=None):
        """Get input signature for seq2seq models.

        Args:
            sequence_length: `int` or `None`. Optional sequence length.

        Returns:
            `dict`. Dictionary mapping input names to their specifications
        """
        if sequence_length is None:
            if hasattr(self.model, "preprocessor") and self.model.preprocessor:
                sequence_length = getattr(
                    self.model.preprocessor,
                    "sequence_length",
                    self.DEFAULT_SEQUENCE_LENGTH,
                )
            else:
                sequence_length = self.DEFAULT_SEQUENCE_LENGTH

        return _get_seq2seq_input_signature(self.model, sequence_length)


@keras_hub_export("keras_hub.export.TextModelExporterConfig")
class TextModelExporterConfig(KerasHubExporterConfig):
    """Generic exporter configuration for text models."""

    MODEL_TYPE = "text_model"
    EXPECTED_INPUTS = ["token_ids", "padding_mask"]
    DEFAULT_SEQUENCE_LENGTH = 128

    def _is_model_compatible(self):
        """Check if model is a text model (fallback).

        Returns:
            `bool`. True if compatible, False otherwise
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
            sequence_length: `int` or `None`. Optional sequence length.

        Returns:
            `dict`. Dictionary mapping input names to their specifications
        """
        if sequence_length is None:
            if hasattr(self.model, "preprocessor") and self.model.preprocessor:
                sequence_length = getattr(
                    self.model.preprocessor,
                    "sequence_length",
                    self.DEFAULT_SEQUENCE_LENGTH,
                )
            else:
                sequence_length = self.DEFAULT_SEQUENCE_LENGTH

        return _get_text_input_signature(self.model, sequence_length)


@keras_hub_export("keras_hub.export.ImageClassifierExporterConfig")
class ImageClassifierExporterConfig(KerasHubExporterConfig):
    """Exporter configuration for Image Classification models."""

    MODEL_TYPE = "image_classifier"
    EXPECTED_INPUTS = ["images"]

    def _is_model_compatible(self):
        """Check if model is an image classifier.
        Returns:
            `bool`. True if compatible, False otherwise
        """
        return isinstance(self.model, ImageClassifier)

    def get_input_signature(self, image_size=None):
        """Get input signature for image classifier models.
        Args:
            image_size: `int`, `tuple` or `None`. Optional image size.
        Returns:
            `dict`. Dictionary mapping input names to their specifications
        """
        if image_size is None:
            image_size = _infer_image_size(self.model)
        elif isinstance(image_size, int):
            image_size = (image_size, image_size)

        # Get input dtype
        dtype = "float32"
        if hasattr(self.model, "inputs") and self.model.inputs:
            model_dtype = self.model.inputs[0].dtype
            dtype = (
                model_dtype.name
                if hasattr(model_dtype, "name")
                else model_dtype
            )

        return {
            "images": keras.layers.InputSpec(
                shape=(None, *image_size, 3),
                dtype=dtype,
                name="images",
            ),
        }


@keras_hub_export("keras_hub.export.ObjectDetectorExporterConfig")
class ObjectDetectorExporterConfig(KerasHubExporterConfig):
    """Exporter configuration for Object Detection models."""

    MODEL_TYPE = "object_detector"
    EXPECTED_INPUTS = ["images", "image_shape"]

    def _is_model_compatible(self):
        """Check if model is an object detector.
        Returns:
            `bool`. True if compatible, False otherwise
        """
        return isinstance(self.model, ObjectDetector)

    def get_input_signature(self, image_size=None):
        """Get input signature for object detector models.
        Args:
            image_size: `int`, `tuple` or `None`. Optional image size.
        Returns:
            `dict`. Dictionary mapping input names to their specifications
        """
        if image_size is None:
            image_size = _infer_image_size(self.model)
        elif isinstance(image_size, int):
            image_size = (image_size, image_size)

        # Get input dtype
        dtype = "float32"
        if hasattr(self.model, "inputs") and self.model.inputs:
            model_dtype = self.model.inputs[0].dtype
            dtype = (
                model_dtype.name
                if hasattr(model_dtype, "name")
                else model_dtype
            )

        return {
            "images": keras.layers.InputSpec(
                shape=(None, *image_size, 3),
                dtype=dtype,
                name="images",
            ),
            "image_shape": keras.layers.InputSpec(
                shape=(None, 2), dtype="int32", name="image_shape"
            ),
        }


@keras_hub_export("keras_hub.export.ImageSegmenterExporterConfig")
class ImageSegmenterExporterConfig(KerasHubExporterConfig):
    """Exporter configuration for Image Segmentation models."""

    MODEL_TYPE = "image_segmenter"
    EXPECTED_INPUTS = ["images"]

    def _is_model_compatible(self):
        """Check if model is an image segmenter.
        Returns:
            `bool`. True if compatible, False otherwise
        """
        return isinstance(self.model, ImageSegmenter)

    def get_input_signature(self, image_size=None):
        """Get input signature for image segmenter models.
        Args:
            image_size: `int`, `tuple` or `None`. Optional image size.
        Returns:
            `dict`. Dictionary mapping input names to their specifications
        """
        if image_size is None:
            image_size = _infer_image_size(self.model)
        elif isinstance(image_size, int):
            image_size = (image_size, image_size)

        # Get input dtype
        dtype = "float32"
        if hasattr(self.model, "inputs") and self.model.inputs:
            model_dtype = self.model.inputs[0].dtype
            dtype = (
                model_dtype.name
                if hasattr(model_dtype, "name")
                else model_dtype
            )

        return {
            "images": keras.layers.InputSpec(
                shape=(None, *image_size, 3),
                dtype=dtype,
                name="images",
            ),
        }
