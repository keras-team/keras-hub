"""LiteRT exporter for Keras-Hub models.

This module provides LiteRT export functionality specifically designed for
Keras-Hub models, handling their unique input structures and requirements.

The exporter supports dynamic shape inputs by default, leveraging TFLite's
native capability to resize input tensors at runtime. When applicable parameters
are not specified, models are exported with flexible dimensions that can be
resized via `interpreter.resize_tensor_input()` before inference.
"""

import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.export.base import KerasHubExporter
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.image_segmenter import ImageSegmenter
from keras_hub.src.models.object_detector import ObjectDetector
from keras_hub.src.models.seq_2_seq_lm import Seq2SeqLM
from keras_hub.src.models.text_classifier import TextClassifier

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
        if isinstance(self.model, (CausalLM, TextClassifier, Seq2SeqLM)):
            return "text"
        # Check for image-only models
        elif isinstance(
            self.model, (ImageClassifier, ObjectDetector, ImageSegmenter)
        ):
            return "image"
        else:
            # For other model types (audio, custom, etc.)
            raise ValueError(
                f"Model type {self.model.__class__.__name__} is not supported "
                "for LiteRT export. Currently supported model types are: "
                "CausalLM, TextClassifier, Seq2SeqLM, ImageClassifier, "
                "ObjectDetector, ImageSegmenter, and multimodal models "
                "(Gemma3CausalLM, PaliGemmaCausalLM, CLIPBackbone)."
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

        Args:
            filepath: `str`. Path where to save the model. If it doesn't end
                with '.tflite', the extension will be added automatically.
        """
        from keras.src.utils import io_utils

        # Ensure filepath ends with .tflite
        if not filepath.endswith(".tflite"):
            filepath = filepath + ".tflite"

        if self.verbose:
            io_utils.print_msg(
                f"Starting LiteRT export for {self.model.__class__.__name__}"
            )

        # Get export parameter based on model type
        param = self._get_export_param()

        # Ensure model is built
        self._ensure_model_built(param)

        # Get input signature
        input_signature = self.config.get_input_signature(param)

        # Get adapter class type for this model
        adapter_type = self._get_model_adapter_class()

        # Create a wrapper that adapts the Keras-Hub model to work with Keras
        # LiteRT exporter
        wrapped_model = self._create_export_wrapper(param, adapter_type)

        # Convert dict input signature to list format for all models
        # The adapter's call() method will handle converting back to dict
        if isinstance(input_signature, dict):
            # Extract specs in the order expected by the model
            signature_list = []
            for input_name in self.config.EXPECTED_INPUTS:
                if input_name in input_signature:
                    signature_list.append(input_signature[input_name])
            input_signature = signature_list

        # Create the Keras LiteRT exporter with the wrapped model
        keras_exporter = KerasLitertExporter(
            wrapped_model,
            input_signature=input_signature,
            aot_compile_targets=self.aot_compile_targets,
            verbose=self.verbose,
            **self.export_kwargs,
        )

        try:
            # Export using the Keras exporter
            keras_exporter.export(filepath)

            if self.verbose:
                io_utils.print_msg(
                    f"Export completed successfully to: {filepath}"
                )

        except Exception as e:
            raise RuntimeError(f"LiteRT export failed: {e}") from e

    def _create_export_wrapper(self, param, adapter_type):
        """Create a wrapper model that handles the input structure conversion.

        This creates a type-specific adapter that converts between the
        list-based inputs that Keras LiteRT exporter provides and the
        dictionary format expected by Keras-Hub models. Note: This adapter
        is independent of dynamic shape support - it only handles input
        format conversion.

        Args:
            param: The parameter for input signature (sequence_length for
                text models, image_size for image models, or None for
                dynamic shapes).
            adapter_type: `str`. The type of adapter to use - "text",
                "image", "multimodal", or "base".
        """

        class BaseModelAdapter(keras.Model):
            """Base adapter for Keras-Hub models."""

            def __init__(
                self,
                keras_hub_model,
                expected_inputs,
                input_signature,
                is_multimodal=False,
            ):
                super().__init__()
                self.keras_hub_model = keras_hub_model
                self.expected_inputs = expected_inputs
                self.input_signature = input_signature
                self.is_multimodal = is_multimodal

                # Create Input layers based on the input signature
                self._input_layers = []
                for input_name in expected_inputs:
                    if input_name in input_signature:
                        spec = input_signature[input_name]
                        input_layer = keras.layers.Input(
                            shape=spec.shape[1:],  # Remove batch dimension
                            dtype=spec.dtype,
                            name=input_name,
                        )
                        self._input_layers.append(input_layer)

                # Store references to the original model's variables
                self._variables = keras_hub_model.variables
                self._trainable_variables = keras_hub_model.trainable_variables
                self._non_trainable_variables = (
                    keras_hub_model.non_trainable_variables
                )

            @property
            def variables(self):
                return self._variables

            @property
            def trainable_variables(self):
                return self._trainable_variables

            @property
            def non_trainable_variables(self):
                return self._non_trainable_variables

            def get_config(self):
                """Return the configuration of the wrapped model."""
                return self.keras_hub_model.get_config()

        class ModelAdapter(BaseModelAdapter):
            """Universal adapter for all Keras-Hub models.

            Handles conversion between list-based inputs (from TFLite) and
            dictionary format expected by Keras-Hub models. Supports text,
            image, and multimodal models.
            """

            def call(self, inputs, training=None, mask=None):
                """Convert list inputs to Keras-Hub model format."""
                if isinstance(inputs, dict):
                    return self.keras_hub_model(inputs, training=training)

                # Convert to list if needed
                if not isinstance(inputs, (list, tuple)):
                    inputs = [inputs]

                # Single input image models can receive tensor directly
                if len(self.expected_inputs) == 1 and not self.is_multimodal:
                    return self.keras_hub_model(inputs[0], training=training)

                # Multi-input models need dictionary format
                input_dict = {}
                for i, input_name in enumerate(self.expected_inputs):
                    if i < len(inputs):
                        input_dict[input_name] = inputs[i]

                return self.keras_hub_model(input_dict, training=training)

        # Create adapter with multimodal flag if needed
        is_multimodal = adapter_type == "multimodal"
        adapter = ModelAdapter(
            self.model,
            self.config.EXPECTED_INPUTS,
            self.config.get_input_signature(param),
            is_multimodal=is_multimodal,
        )

        # Build the adapter as a Functional model by calling it with the
        # inputs. Pass the input layers as a list - the adapter's call()
        # will convert to dict format as needed.
        outputs = adapter(adapter._input_layers)
        functional_model = keras.Model(
            inputs=adapter._input_layers, outputs=outputs
        )

        # Copy over the variables from the original model
        functional_model._variables = adapter._variables
        functional_model._trainable_variables = adapter._trainable_variables
        functional_model._non_trainable_variables = (
            adapter._non_trainable_variables
        )

        return functional_model


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
