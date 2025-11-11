"""Configuration classes for different Keras-Hub model types.

This module provides specific configurations for exporting different types
of Keras-Hub models, following the Optimum pattern.
"""

import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.export.base import KerasHubExporterConfig
from keras_hub.src.models.audio_to_text import AudioToText
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.depth_estimator import DepthEstimator
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.image_segmenter import ImageSegmenter
from keras_hub.src.models.object_detector import ObjectDetector
from keras_hub.src.models.seq_2_seq_lm import Seq2SeqLM
from keras_hub.src.models.text_classifier import TextClassifier
from keras_hub.src.models.text_to_image import TextToImage


def _get_text_input_signature(model, sequence_length=None):
    """Get input signature for text models with token_ids and padding_mask.

    Args:
        model: The model instance.
        sequence_length: `int` or `None`. Sequence length. If None, uses
            dynamic shape to support variable-length inputs via
            resize_tensor_input at runtime.

    Returns:
        `dict`. Dictionary mapping input names to their specifications
    """
    return {
        "token_ids": keras.layers.InputSpec(
            dtype="int32", shape=(None, sequence_length)
        ),
        "padding_mask": keras.layers.InputSpec(
            dtype="int32", shape=(None, sequence_length)
        ),
    }


def _get_seq2seq_input_signature(model, sequence_length=None):
    """Get input signature for seq2seq models with encoder/decoder tokens.

    Args:
        model: The model instance.
        sequence_length: `int` or `None`. Sequence length. If None, uses
            dynamic shape to support variable-length inputs via
            resize_tensor_input at runtime.

    Returns:
        `dict`. Dictionary mapping input names to their specifications
    """
    return {
        "encoder_token_ids": keras.layers.InputSpec(
            dtype="int32", shape=(None, sequence_length)
        ),
        "encoder_padding_mask": keras.layers.InputSpec(
            dtype="int32", shape=(None, sequence_length)
        ),
        "decoder_token_ids": keras.layers.InputSpec(
            dtype="int32", shape=(None, sequence_length)
        ),
        "decoder_padding_mask": keras.layers.InputSpec(
            dtype="int32", shape=(None, sequence_length)
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
    if image_size is None and hasattr(model, "inputs") and model.inputs:
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


def _infer_image_dtype(model):
    """Infer image dtype from model inputs.

    Args:
        model: The model instance.

    Returns:
        `str`. Image dtype (defaults to "float32").
    """
    if hasattr(model, "inputs") and model.inputs:
        model_dtype = model.inputs[0].dtype
        return model_dtype.name if hasattr(model_dtype, "name") else model_dtype
    return "float32"


@keras_hub_export("keras_hub.export.CausalLMExporterConfig")
class CausalLMExporterConfig(KerasHubExporterConfig):
    """Exporter configuration for Causal Language Models (GPT, LLaMA, etc.)."""

    MODEL_TYPE = "causal_lm"

    def __init__(self, model):
        super().__init__(model)
        # Determine expected inputs based on whether model is multimodal
        # Check for Gemma3-style vision encoder
        if (
            hasattr(model, "backbone")
            and hasattr(model.backbone, "vision_encoder")
            and model.backbone.vision_encoder is not None
        ):
            self.EXPECTED_INPUTS = [
                "token_ids",
                "padding_mask",
                "images",
                "vision_mask",
                "vision_indices",
            ]
        # Check for PaliGemma-style multimodal (has image_encoder or
        # vit attributes)
        elif self._is_paligemma_style_multimodal(model):
            self.EXPECTED_INPUTS = [
                "token_ids",
                "padding_mask",
                "images",
                "response_mask",
            ]
        # Check for Parseq-style vision (has image_encoder in backbone)
        elif self._is_parseq_style_vision(model):
            self.EXPECTED_INPUTS = ["token_ids", "padding_mask", "images"]
        else:
            self.EXPECTED_INPUTS = ["token_ids", "padding_mask"]

    def _is_paligemma_style_multimodal(self, model):
        """Check if model is PaliGemma-style multimodal (vision + language)."""
        if hasattr(model, "backbone"):
            backbone = model.backbone
            # PaliGemma has vit parameters or image-related attributes
            if hasattr(backbone, "image_size") and (
                hasattr(backbone, "vit_num_layers")
                or hasattr(backbone, "vit_patch_size")
            ):
                return True
        return False

    def _is_parseq_style_vision(self, model):
        """Check if model is Parseq-style vision model (OCR causal LM)."""
        if hasattr(model, "backbone"):
            backbone = model.backbone
            # Parseq has an image_encoder attribute
            if hasattr(backbone, "image_encoder"):
                return True
        return False

    def _is_model_compatible(self):
        """Check if model is a causal language model.

        Returns:
            `bool`. True if compatible, False otherwise
        """
        return isinstance(self.model, CausalLM)

    def get_input_signature(self, sequence_length=None):
        """Get input signature for causal LM models.

        Args:
            sequence_length: `int`, `None`, or `dict`. Optional sequence length.
                If None, exports with dynamic shape for flexibility. If dict,
                should contain 'sequence_length' and 'image_size' for
                multimodal models.

        Returns:
            `dict`. Dictionary mapping input names to their specifications
        """
        # Use dynamic shape (None) by default for TFLite flexibility
        # Users can resize at runtime via interpreter.resize_tensor_input()

        # Handle dict param for multimodal models
        if isinstance(sequence_length, dict):
            seq_len = sequence_length.get("sequence_length", None)
        else:
            seq_len = sequence_length

        signature = _get_text_input_signature(self.model, seq_len)

        # Check if Gemma3-style multimodal (vision encoder)
        if (
            hasattr(self.model.backbone, "vision_encoder")
            and self.model.backbone.vision_encoder is not None
        ):
            # Add Gemma3 vision inputs
            if isinstance(sequence_length, dict):
                image_size = sequence_length.get("image_size", None)
                if image_size is not None and isinstance(image_size, tuple):
                    image_size = image_size[0]  # Use first dimension if tuple
            else:
                image_size = getattr(self.model.backbone, "image_size", 224)

            if image_size is None:
                image_size = getattr(self.model.backbone, "image_size", 224)

            signature.update(
                {
                    "images": keras.layers.InputSpec(
                        dtype="float32",
                        shape=(None, None, image_size, image_size, 3),
                    ),
                    "vision_mask": keras.layers.InputSpec(
                        dtype="int32",  # Use int32 instead of bool for
                        # TFLite compatibility
                        shape=(None, None),
                    ),
                    "vision_indices": keras.layers.InputSpec(
                        dtype="int32", shape=(None, None)
                    ),
                }
            )
        # Check if PaliGemma-style multimodal
        elif self._is_paligemma_style_multimodal(self.model):
            # Get image size from backbone
            image_size = getattr(self.model.backbone, "image_size", 224)
            if isinstance(sequence_length, dict):
                image_size = sequence_length.get("image_size", image_size)

            # Handle tuple image_size (height, width)
            if isinstance(image_size, tuple):
                image_height, image_width = image_size[0], image_size[1]
            else:
                image_height, image_width = image_size, image_size

            signature.update(
                {
                    "images": keras.layers.InputSpec(
                        dtype="float32",
                        shape=(None, image_height, image_width, 3),
                    ),
                    "response_mask": keras.layers.InputSpec(
                        dtype="int32", shape=(None, seq_len)
                    ),
                }
            )
        # Check if Parseq-style vision
        elif self._is_parseq_style_vision(self.model):
            # Get image size from backbone's image_encoder
            if hasattr(self.model.backbone, "image_encoder") and hasattr(
                self.model.backbone.image_encoder, "image_shape"
            ):
                image_shape = self.model.backbone.image_encoder.image_shape
                image_height, image_width = image_shape[0], image_shape[1]
            else:
                image_height, image_width = 32, 128  # Default for Parseq

            if isinstance(sequence_length, dict):
                image_height = sequence_length.get("image_height", image_height)
                image_width = sequence_length.get("image_width", image_width)

            signature.update(
                {
                    "images": keras.layers.InputSpec(
                        dtype="float32",
                        shape=(None, image_height, image_width, 3),
                    ),
                }
            )

        return signature


@keras_hub_export("keras_hub.export.TextClassifierExporterConfig")
class TextClassifierExporterConfig(KerasHubExporterConfig):
    """Exporter configuration for Text Classification models."""

    MODEL_TYPE = "text_classifier"

    def __init__(self, model):
        super().__init__(model)
        # Determine expected inputs based on model characteristics
        inputs = ["token_ids"]

        if self._model_uses_padding_mask():
            inputs.append("padding_mask")

        if self._model_uses_segment_ids():
            inputs.append("segment_ids")

        self.EXPECTED_INPUTS = inputs

    def _model_uses_segment_ids(self):
        """Check if the model expects segment_ids input.

        Returns:
            bool: True if model uses segment_ids, False otherwise
        """
        # Check if model has a backbone with num_segments attribute
        if hasattr(self.model, "backbone"):
            backbone = self.model.backbone
            # RoformerV2 and similar models have num_segments
            if hasattr(backbone, "num_segments"):
                return True
        return False

    def _model_uses_padding_mask(self):
        """Check if the model expects padding_mask input.

        Returns:
            bool: True if model uses padding_mask, False otherwise
        """
        # RoformerV2 doesn't use padding_mask in its preprocessor
        # Check the model's backbone type
        if hasattr(self.model, "backbone"):
            backbone_class_name = self.model.backbone.__class__.__name__
            # RoformerV2 doesn't use padding_mask
            if "RoformerV2" in backbone_class_name:
                return False
        return True

    def _is_model_compatible(self):
        """Check if model is a text classifier.

        Returns:
            `bool`. True if compatible, False otherwise
        """
        return isinstance(self.model, TextClassifier)

    def get_input_signature(self, sequence_length=None):
        """Get input signature for text classifier models.

        Args:
            sequence_length: `int` or `None`. Optional sequence length. If None,
                exports with dynamic shape for flexibility.

        Returns:
            `dict`. Dictionary mapping input names to their specifications
        """
        # Use dynamic shape (None) by default for TFLite flexibility
        # Users can resize at runtime via interpreter.resize_tensor_input()
        signature = {
            "token_ids": keras.layers.InputSpec(
                dtype="int32", shape=(None, sequence_length)
            )
        }

        # Add padding_mask if needed
        if self._model_uses_padding_mask():
            signature["padding_mask"] = keras.layers.InputSpec(
                dtype="int32", shape=(None, sequence_length)
            )

        # Add segment_ids if needed
        if self._model_uses_segment_ids():
            signature["segment_ids"] = keras.layers.InputSpec(
                dtype="int32", shape=(None, sequence_length)
            )

        return signature


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

    def _is_model_compatible(self):
        """Check if model is a seq2seq language model.

        Returns:
            `bool`. True if compatible, False otherwise
        """
        return isinstance(self.model, Seq2SeqLM)

    def get_input_signature(self, sequence_length=None):
        """Get input signature for seq2seq models.

        Args:
            sequence_length: `int` or `None`. Optional sequence length. If None,
                exports with dynamic shape for flexibility.

        Returns:
            `dict`. Dictionary mapping input names to their specifications
        """
        # Use dynamic shape (None) by default for TFLite flexibility
        # Users can resize at runtime via interpreter.resize_tensor_input()
        return _get_seq2seq_input_signature(self.model, sequence_length)


@keras_hub_export("keras_hub.export.AudioToTextExporterConfig")
class AudioToTextExporterConfig(KerasHubExporterConfig):
    """Exporter configuration for Audio-to-Text models.

    AudioToText models process audio input and generate text output,
    such as speech recognition or audio transcription models.
    """

    MODEL_TYPE = "audio_to_text"
    EXPECTED_INPUTS = [
        "encoder_input_values",  # Audio features
        "encoder_padding_mask",
        "decoder_token_ids",
        "decoder_padding_mask",
    ]

    def _is_model_compatible(self):
        """Check if model is an audio-to-text model.

        Returns:
            `bool`. True if compatible, False otherwise
        """
        return isinstance(self.model, AudioToText)

    def get_input_signature(self, sequence_length=None, audio_length=None):
        """Get input signature for audio-to-text models.

        Args:
            sequence_length: `int` or `None`. Optional text sequence length.
                If None, exports with dynamic shape for flexibility.
            audio_length: `int` or `None`. Optional audio sequence length.
                If None, exports with dynamic shape for flexibility.

        Returns:
            `dict`. Dictionary mapping input names to their specifications
        """
        # Audio features come from the audio encoder
        # Text tokens go to the decoder
        return {
            "encoder_input_values": keras.layers.InputSpec(
                dtype="float32", shape=(None, audio_length)
            ),
            "encoder_padding_mask": keras.layers.InputSpec(
                dtype="int32", shape=(None, audio_length)
            ),
            "decoder_token_ids": keras.layers.InputSpec(
                dtype="int32", shape=(None, sequence_length)
            ),
            "decoder_padding_mask": keras.layers.InputSpec(
                dtype="int32", shape=(None, sequence_length)
            ),
        }


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

        dtype = _infer_image_dtype(self.model)

        return {
            "images": keras.layers.InputSpec(
                dtype=dtype, shape=(None, *image_size, 3)
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
            # Try to infer from preprocessor, but fall back to dynamic shapes
            # for object detectors which support variable input sizes
            try:
                image_size = _infer_image_size(self.model)
            except ValueError:
                # If cannot infer, use dynamic shapes
                image_size = None
        elif isinstance(image_size, int):
            image_size = (image_size, image_size)

        dtype = _infer_image_dtype(self.model)

        if image_size is not None:
            # Use concrete shapes when image_size is available
            return {
                "images": keras.layers.InputSpec(
                    dtype=dtype, shape=(None, *image_size, 3)
                ),
                "image_shape": keras.layers.InputSpec(
                    dtype="int32", shape=(None, 2)
                ),
            }
        else:
            # Use dynamic shapes for variable input sizes
            return {
                "images": keras.layers.InputSpec(
                    dtype=dtype, shape=(None, None, None, 3)
                ),
                "image_shape": keras.layers.InputSpec(
                    dtype="int32", shape=(None, 2)
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

        dtype = _infer_image_dtype(self.model)

        return {
            "images": keras.layers.InputSpec(
                dtype=dtype, shape=(None, *image_size, 3)
            ),
        }


@keras_hub_export("keras_hub.export.SAMImageSegmenterExporterConfig")
class SAMImageSegmenterExporterConfig(KerasHubExporterConfig):
    """Exporter configuration for SAM (Segment Anything Model).

    SAM requires multiple prompt inputs (points, boxes, masks) in addition
    to images. For TFLite/LiteRT export, we use fixed shapes to avoid issues
    with 0-sized dimensions in the XNNPack delegate.

    Mobile SAM implementations typically use fixed shapes:
    - 1 point prompt (padded with zeros if not used)
    - 1 box prompt (padded with zeros if not used)
    - 1 mask prompt (zero-filled means "no mask")
    """

    MODEL_TYPE = "image_segmenter"
    EXPECTED_INPUTS = ["images", "points", "labels", "boxes", "masks"]

    def _is_model_compatible(self):
        """Check if model is a SAM image segmenter.
        Returns:
            `bool`. True if compatible, False otherwise
        """
        if not isinstance(self.model, ImageSegmenter):
            return False
        # Check if backbone is SAM - must have SAM in backbone class name
        if hasattr(self.model, "backbone"):
            backbone_class_name = self.model.backbone.__class__.__name__
            # Only SAM models should use this config
            if "SAM" in backbone_class_name.upper():
                return True
        return False

    def get_input_signature(self, image_size=None):
        """Get input signature for SAM models.
        Args:
            image_size: `int`, `tuple` or `None`. Optional image size.
        Returns:
            `dict`. Dictionary mapping input names to their specifications
        """
        if image_size is None:
            image_size = _infer_image_size(self.model)
        elif isinstance(image_size, int):
            image_size = (image_size, image_size)

        dtype = _infer_image_dtype(self.model)

        # For SAM, mask inputs should be at 4 * image_embedding_size resolution
        # image_embedding_size is typically image_size // 16 for patch_size=16
        image_embedding_size = (image_size[0] // 16, image_size[1] // 16)
        mask_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])

        return {
            "images": keras.layers.InputSpec(
                dtype=dtype, shape=(None, *image_size, 3)
            ),
            "points": keras.layers.InputSpec(
                dtype="float32",
                shape=(None, 1, 2),  # Fixed: 1 point
            ),
            "labels": keras.layers.InputSpec(
                dtype="float32",
                shape=(None, 1),  # Fixed: 1 label
            ),
            "boxes": keras.layers.InputSpec(
                dtype="float32",
                shape=(None, 1, 2, 2),  # Fixed: 1 box
            ),
            "masks": keras.layers.InputSpec(
                dtype="float32",
                shape=(
                    None,
                    1,
                    *mask_size,
                    1,
                ),  # Fixed: 1 mask at correct resolution
            ),
        }


@keras_hub_export("keras_hub.export.DepthEstimatorExporterConfig")
class DepthEstimatorExporterConfig(KerasHubExporterConfig):
    """Exporter configuration for Depth Estimation models."""

    MODEL_TYPE = "depth_estimator"
    EXPECTED_INPUTS = ["images"]

    def _is_model_compatible(self):
        """Check if model is a depth estimator.
        Returns:
            `bool`. True if compatible, False otherwise
        """
        return isinstance(self.model, DepthEstimator)

    def get_input_signature(self, image_size=None):
        """Get input signature for depth estimation models.
        Args:
            image_size: `int`, `tuple` or `None`. Optional image size.
        Returns:
            `dict`. Dictionary mapping input names to their specifications
        """
        if image_size is None:
            image_size = _infer_image_size(self.model)
        elif isinstance(image_size, int):
            image_size = (image_size, image_size)

        dtype = _infer_image_dtype(self.model)

        return {
            "images": keras.layers.InputSpec(
                dtype=dtype, shape=(None, *image_size, 3)
            ),
        }


@keras_hub_export("keras_hub.export.TextToImageExporterConfig")
class TextToImageExporterConfig(KerasHubExporterConfig):
    """Exporter configuration for Text-to-Image models.

    TextToImage models generate images from text prompts,
    such as Stable Diffusion, DALL-E, or similar generative models.
    """

    MODEL_TYPE = "text_to_image"
    EXPECTED_INPUTS = [
        "images",
        "latents",
        "clip_l_token_ids",
        "clip_l_negative_token_ids",
        "clip_g_token_ids",
        "clip_g_negative_token_ids",
        "num_steps",
        "guidance_scale",
    ]

    def _is_model_compatible(self):
        """Check if model is a text-to-image model.

        Returns:
            `bool`. True if compatible, False otherwise
        """
        return isinstance(self.model, TextToImage)

    def _is_stable_diffusion_3(self):
        """Check if model is Stable Diffusion 3.

        Returns:
            `bool`. True if model is SD3, False otherwise
        """
        return "StableDiffusion3" in self.model.__class__.__name__

    def get_input_signature(
        self, sequence_length=None, image_size=None, latent_shape=None
    ):
        """Get input signature for text-to-image models.

        Args:
            sequence_length: `int` or `None`. Optional text sequence length.
                If None, exports with dynamic shape for flexibility.
            image_size: `tuple`, `int` or `None`. Optional image size. If None,
                infers from model.
            latent_shape: `tuple` or `None`. Optional latent shape. If None,
                infers from model.

        Returns:
            `dict`. Dictionary mapping input names to their specifications
        """
        # Check if this is Stable Diffusion 3 which has dual CLIP encoders
        if self._is_stable_diffusion_3():
            # Get image size from backbone if available
            if image_size is None:
                if hasattr(self.model, "backbone") and hasattr(
                    self.model.backbone, "image_shape"
                ):
                    image_shape_tuple = self.model.backbone.image_shape
                    image_size = (image_shape_tuple[0], image_shape_tuple[1])
                else:
                    # Try to infer from inputs
                    if hasattr(self.model, "input") and isinstance(
                        self.model.input, dict
                    ):
                        if "images" in self.model.input:
                            img_shape = self.model.input["images"].shape
                            if (
                                img_shape[1] is not None
                                and img_shape[2] is not None
                            ):
                                image_size = (img_shape[1], img_shape[2])
                    if image_size is None:
                        raise ValueError(
                            "Could not determine image size for "
                            "StableDiffusion3. "
                            "Please provide image_size parameter."
                        )
            elif isinstance(image_size, int):
                image_size = (image_size, image_size)

            # Get latent shape from backbone if available
            if latent_shape is None:
                if hasattr(self.model, "backbone") and hasattr(
                    self.model.backbone, "latent_shape"
                ):
                    latent_shape_tuple = self.model.backbone.latent_shape
                    # latent_shape is (None, h, w, c), we need (h, w, c)
                    if latent_shape_tuple[0] is None:
                        latent_shape = latent_shape_tuple[1:]
                    else:
                        latent_shape = latent_shape_tuple
                else:
                    # Default latent shape for SD3 (typically 1/8 of image size
                    # with 16 channels)
                    latent_shape = (image_size[0] // 8, image_size[1] // 8, 16)

            return {
                "images": keras.layers.InputSpec(
                    dtype="float32", shape=(None, *image_size, 3)
                ),
                "latents": keras.layers.InputSpec(
                    dtype="float32", shape=(None, *latent_shape)
                ),
                "clip_l_token_ids": keras.layers.InputSpec(
                    dtype="int32", shape=(None, sequence_length)
                ),
                "clip_l_negative_token_ids": keras.layers.InputSpec(
                    dtype="int32", shape=(None, sequence_length)
                ),
                "clip_g_token_ids": keras.layers.InputSpec(
                    dtype="int32", shape=(None, sequence_length)
                ),
                "clip_g_negative_token_ids": keras.layers.InputSpec(
                    dtype="int32", shape=(None, sequence_length)
                ),
                "num_steps": keras.layers.InputSpec(
                    dtype="int32", shape=(None,)
                ),
                "guidance_scale": keras.layers.InputSpec(
                    dtype="float32", shape=(None,)
                ),
            }
        else:
            # For other text-to-image models, use simple text inputs
            return _get_text_input_signature(self.model, sequence_length)


def get_exporter_config(model):
    """Get the appropriate exporter configuration for a model instance.

    This function automatically detects the model type and returns the
    corresponding exporter configuration.

    Args:
        model: A Keras-Hub model instance (e.g., CausalLM, TextClassifier).

    Returns:
        An instance of the appropriate KerasHubExporterConfig subclass.

    Raises:
        ValueError: If the model type is not supported for export.
    """
    # Mapping of model classes to their config classes
    # NOTE: Order matters! More specific configs must be checked first:
    # - AudioToText before Seq2SeqLM (AudioToText is a subclass of Seq2SeqLM)
    # - Seq2SeqLM before CausalLM (Seq2SeqLM is a subclass of CausalLM)
    # - SAMImageSegmenterExporterConfig before ImageSegmenterExporterConfig
    _MODEL_TYPE_TO_CONFIG = [
        (AudioToText, AudioToTextExporterConfig),
        (Seq2SeqLM, Seq2SeqLMExporterConfig),
        (CausalLM, CausalLMExporterConfig),
        (TextClassifier, TextClassifierExporterConfig),
        (ImageClassifier, ImageClassifierExporterConfig),
        (ObjectDetector, ObjectDetectorExporterConfig),
        (ImageSegmenter, SAMImageSegmenterExporterConfig),  # Check SAM first
        (ImageSegmenter, ImageSegmenterExporterConfig),  # Then generic
        (DepthEstimator, DepthEstimatorExporterConfig),
        (TextToImage, TextToImageExporterConfig),
    ]

    # Find matching config class
    for model_class, config_class in _MODEL_TYPE_TO_CONFIG:
        if isinstance(model, model_class):
            # Try to create config and check compatibility
            try:
                config = config_class(model)
                return config
            except ValueError:
                # Model not compatible with this config, try next one
                continue

    # Model type not supported
    supported_types = ", ".join(
        set(cls.__name__ for cls, _ in _MODEL_TYPE_TO_CONFIG)
    )
    raise ValueError(
        f"Could not find exporter config for model type "
        f"'{model.__class__.__name__}'. "
        f"Supported types: {supported_types}"
    )
