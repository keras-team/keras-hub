"""LiteRT exporter for Keras-Hub models.

This module provides LiteRT export functionality specifically designed for
Keras-Hub models, handling their unique input structures and requirements.

The exporter supports dynamic shape inputs by default, leveraging TFLite's
native capability to resize input tensors at runtime. When applicable parameters
are not specified, models are exported with flexible dimensions that can be
resized via `interpreter.resize_tensor_input()` before inference.
"""

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.export.base import KerasHubExporter
from keras_hub.src.models.audio_to_text import AudioToText
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.depth_estimator import DepthEstimator
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.image_segmenter import ImageSegmenter
from keras_hub.src.models.object_detector import ObjectDetector
from keras_hub.src.models.seq_2_seq_lm import Seq2SeqLM
from keras_hub.src.models.text_classifier import TextClassifier
from keras_hub.src.models.text_to_image import TextToImage

try:
    from keras.src.export.litert import LiteRTExporter as KerasLitertExporter

    KERAS_LITE_RT_AVAILABLE = True
except ImportError:
    KERAS_LITE_RT_AVAILABLE = False
    KerasLitertExporter = None


@keras_hub_export("keras_hub.export.LiteRTExporter")
class LiteRTExporter(KerasHubExporter):
    """LiteRT exporter for Keras-Hub models.

    This exporter handles the conversion of Keras-Hub models to TensorFlow Lite
    format, properly managing the dictionary input structures that Keras-Hub
    models expect. By default, it exports models with dynamic shape support,
    allowing runtime flexibility via `interpreter.resize_tensor_input()`.

    For text-based models (CausalLM, TextClassifier, Seq2SeqLM), sequence
    dimensions are dynamic when max_sequence_length is not specified. For
    image-based models (ImageClassifier, ObjectDetector, ImageSegmenter),
    image dimensions are dynamic by default.

    Example usage with dynamic shapes:
        ```python
        # Export with dynamic shape support (default)
        model.export("model.tflite", format="litert")

        # At inference time, resize as needed:
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        input_details = interpreter.get_input_details()
        interpreter.resize_tensor_input(input_details[0]["index"], [1, 256])
        interpreter.allocate_tensors()
        ```
    """

    def __init__(
        self,
        config,
        max_sequence_length=None,
        aot_compile_targets=None,
        verbose=None,
        **kwargs,
    ):
        """Initialize the LiteRT exporter.

        Args:
            config: `KerasHubExporterConfig`. Exporter configuration.
            max_sequence_length: `int` or `None`. Maximum sequence length for
                text-based models (CausalLM, TextClassifier, Seq2SeqLM). If
                `None`, exports with dynamic sequence shapes, allowing runtime
                resizing via `interpreter.resize_tensor_input()`. Ignored for
                image-based models.
            aot_compile_targets: `list` or `None`. AOT compilation targets.
            verbose: `bool` or `None`. Whether to print progress. Defaults to
                `None`, which will use `True`.
            **kwargs: `dict`. Additional arguments passed to exporter.
        """
        super().__init__(config, **kwargs)

        if not KERAS_LITE_RT_AVAILABLE:
            raise ImportError(
                "Keras LiteRT exporter is not available. "
                "Make sure you have Keras with LiteRT support installed."
            )

        self.max_sequence_length = max_sequence_length
        self.aot_compile_targets = aot_compile_targets
        self.verbose = verbose if verbose is not None else True

    def _get_model_adapter_class(self):
        """Determine the appropriate adapter class for the model.

        Returns:
            `str`. The adapter type to use ("text", "image", or "multimodal").

        Raises:
            ValueError: If the model type is not supported for LiteRT export.
        """
        # Check if this is a multimodal model (has both vision and text inputs)
        model_to_check = self.model
        if hasattr(self.model, "backbone"):
            model_to_check = self.model.backbone

        # Check if model has multimodal inputs
        if hasattr(model_to_check, "input") and isinstance(
            model_to_check.input, dict
        ):
            input_names = set(model_to_check.input.keys())
            has_images = "images" in input_names
            has_text = any(
                name in input_names
                for name in ["token_ids", "encoder_token_ids"]
            )
            if has_images and has_text:
                return "multimodal"

        # Check for text-only models
        if isinstance(
            self.model,
            (CausalLM, TextClassifier, Seq2SeqLM, AudioToText, TextToImage),
        ):
            return "text"
        # Check for image-only models
        elif isinstance(
            self.model,
            (ImageClassifier, ObjectDetector, ImageSegmenter, DepthEstimator),
        ):
            return "image"
        else:
            # For other model types (audio, custom, etc.)
            raise ValueError(
                f"Model type {self.model.__class__.__name__} is not supported "
                "for LiteRT export. Currently supported model types are: "
                "CausalLM, TextClassifier, Seq2SeqLM, AudioToText, "
                "TextToImage, "
                "ImageClassifier, ObjectDetector, ImageSegmenter, "
                "DepthEstimator, and multimodal "
                "models (Gemma3CausalLM, PaliGemmaCausalLM, CLIPBackbone)."
            )

    def _get_export_param(self):
        """Get the appropriate parameter for export based on model type.

        Returns:
            The parameter to use for export (sequence_length for text models,
            image_size for image models, dict for multimodal, or None for
            other model types).
        """
        adapter_type = self._get_model_adapter_class()

        if adapter_type == "text":
            # For text models, use sequence_length
            return self.max_sequence_length
        elif adapter_type == "image":
            # For image models, get image_size from preprocessor
            if hasattr(self.model, "preprocessor") and hasattr(
                self.model.preprocessor, "image_size"
            ):
                return self.model.preprocessor.image_size
            else:
                return None  # Will use default in get_input_signature
        elif adapter_type == "multimodal":
            # For multimodal models, return dict with both params
            model_to_check = self.model
            if hasattr(self.model, "backbone"):
                model_to_check = self.model.backbone

            # Try to infer image size from vision encoder
            image_size = None
            for attr in ["vision_encoder", "vit", "image_encoder"]:
                if hasattr(model_to_check, attr):
                    encoder = getattr(model_to_check, attr)
                    if hasattr(encoder, "image_shape"):
                        image_shape = encoder.image_shape
                        if image_shape:
                            image_size = image_shape[:2]
                            break
                    elif hasattr(encoder, "image_size"):
                        size = encoder.image_size
                        image_size = (
                            (size, size) if isinstance(size, int) else size
                        )
                        break

            # Check model's image_size attribute
            if image_size is None and hasattr(model_to_check, "image_size"):
                size = model_to_check.image_size
                image_size = (size, size) if isinstance(size, int) else size

            return {
                "image_size": image_size,
                "sequence_length": self.max_sequence_length,
            }
        else:
            # For other model types
            return None

    def export(self, filepath):
        """Export the Keras-Hub model to LiteRT format.

        This method now delegates to Keras Core's LiteRT exporter, which
        automatically handles dictionary inputs. The domain-specific input
        signature (with sequence_length, image_size, etc.) is still built
        using Keras-Hub's config system.

        Args:
            filepath: `str`. Path where to save the model. If it doesn't end
                with '.tflite', the extension will be added automatically.
        """
        from keras.src.export.litert import export_litert
        from keras.src.utils import io_utils

        # Ensure filepath ends with .tflite
        if not filepath.endswith(".tflite"):
            filepath = filepath + ".tflite"

        if self.verbose:
            io_utils.print_msg(
                f"Starting LiteRT export for {self.model.__class__.__name__}"
            )

        # Get export parameter based on model type
        # (e.g., sequence_length, image_size)
        param = self._get_export_param()

        # Get input signature from config (domain-specific knowledge)
        # Keras Core's export_litert will handle model building
        input_signature = self.config.get_input_signature(param)

        try:
            # Use Keras Core's export - it handles dict inputs automatically!
            export_litert(
                self.model,
                filepath,
                input_signature=input_signature,
                aot_compile_targets=self.aot_compile_targets,
                verbose=self.verbose,
                **self.export_kwargs,
            )

            if self.verbose:
                io_utils.print_msg(
                    f"Export completed successfully to: {filepath}"
                )

        except Exception as e:
            raise RuntimeError(f"LiteRT export failed: {e}") from e


# Convenience function for direct export
def export_litert(model, filepath, **kwargs):
    """Export a Keras-Hub model to Litert format.

    This is a convenience function that automatically detects the model type
    and exports it using the appropriate configuration.

    Args:
        model: `keras.Model`. The Keras-Hub model to export.
        filepath: `str`. Path where to save the model (without extension).
        **kwargs: `dict`. Additional arguments passed to exporter.
    """
    from keras_hub.src.export.configs import get_exporter_config

    # Get the appropriate configuration for this model
    config = get_exporter_config(model)

    # Create and use the LiteRT exporter
    exporter = LiteRTExporter(config, **kwargs)
    exporter.export(filepath)
