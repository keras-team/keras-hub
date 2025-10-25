"""LiteRT exporter for Keras-Hub models.

This module provides LiteRT export functionality specifically designed for
Keras-Hub models, handling their unique input structures and requirements.
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
    models expect.
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
            max_sequence_length: `int` or `None`. Maximum sequence length.
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

        # Determine the parameter to pass based on model type using isinstance
        is_text_model = isinstance(
            self.model, (CausalLM, TextClassifier, Seq2SeqLM)
        )
        is_image_model = isinstance(
            self.model, (ImageClassifier, ObjectDetector, ImageSegmenter)
        )

        # For text models, use sequence_length; for image models, get image_size
        # from preprocessor
        if is_text_model:
            param = self.max_sequence_length
        elif is_image_model:
            # Get image_size from model's preprocessor
            if hasattr(self.model, "preprocessor") and hasattr(
                self.model.preprocessor, "image_size"
            ):
                param = self.model.preprocessor.image_size
            else:
                param = None  # Will use default in get_input_signature
        else:
            param = None

        # Ensure model is built
        self._ensure_model_built(param)

        # Get input signature
        input_signature = self.config.get_input_signature(param)

        # Create a wrapper that adapts the Keras-Hub model to work with Keras
        # LiteRT exporter
        wrapped_model = self._create_export_wrapper()

        # Convert input signature to list format expected by Keras exporter
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

    def _create_export_wrapper(self):
        """Create a wrapper model that handles the input structure conversion.

        This creates a type-specific adapter that converts between the
        list-based inputs that Keras LiteRT exporter provides and the format
        expected by Keras-Hub models.
        """

        class BaseModelAdapter(keras.Model):
            """Base adapter for Keras-Hub models."""

            def __init__(
                self, keras_hub_model, expected_inputs, input_signature
            ):
                super().__init__()
                self.keras_hub_model = keras_hub_model
                self.expected_inputs = expected_inputs
                self.input_signature = input_signature

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

            @property
            def inputs(self):
                """Return the input layers for the Keras exporter to use."""
                return self._input_layers

            def get_config(self):
                """Return the configuration of the wrapped model."""
                return self.keras_hub_model.get_config()

        class TextModelAdapter(BaseModelAdapter):
            """Adapter for text models (CausalLM, TextClassifier, Seq2SeqLM).

            Text models expect dictionary inputs with keys like 'token_ids'
            and 'padding_mask'.
            """

            def call(self, inputs, training=None, mask=None):
                """Convert list inputs to dictionary format for text models."""
                if isinstance(inputs, dict):
                    return self.keras_hub_model(inputs, training=training)

                # Convert to list if needed
                if not isinstance(inputs, (list, tuple)):
                    inputs = [inputs]

                # Map inputs to expected dictionary keys
                input_dict = {}
                for i, input_name in enumerate(self.expected_inputs):
                    if i < len(inputs):
                        input_dict[input_name] = inputs[i]

                return self.keras_hub_model(input_dict, training=training)

        class ImageModelAdapter(BaseModelAdapter):
            """Adapter for image models (ImageClassifier, ObjectDetector,
            ImageSegmenter).

            Image models typically expect a single tensor input but may also
            accept dictionary format with 'images' key.
            """

            def call(self, inputs, training=None, mask=None):
                """Convert list inputs to format expected by image models."""
                if isinstance(inputs, dict):
                    return self.keras_hub_model(inputs, training=training)

                # Convert to list if needed
                if not isinstance(inputs, (list, tuple)):
                    inputs = [inputs]

                # Most image models expect a single tensor input
                if len(self.expected_inputs) == 1:
                    return self.keras_hub_model(inputs[0], training=training)

                # If multiple inputs, use dictionary format
                input_dict = {}
                for i, input_name in enumerate(self.expected_inputs):
                    if i < len(inputs):
                        input_dict[input_name] = inputs[i]

                return self.keras_hub_model(input_dict, training=training)

        # Determine the parameter to pass based on model type
        is_text_model = isinstance(
            self.model, (CausalLM, TextClassifier, Seq2SeqLM)
        )
        is_image_model = isinstance(
            self.model, (ImageClassifier, ObjectDetector, ImageSegmenter)
        )

        # Get the appropriate parameter for input signature
        if is_text_model:
            param = self.max_sequence_length
        elif is_image_model:
            # Get image_size from model's preprocessor
            if hasattr(self.model, "preprocessor") and hasattr(
                self.model.preprocessor, "image_size"
            ):
                param = self.model.preprocessor.image_size
            else:
                param = None  # Will use default in get_input_signature
        else:
            param = None

        # Select the appropriate adapter based on model type
        if is_text_model:
            adapter_class = TextModelAdapter
        elif is_image_model:
            adapter_class = ImageModelAdapter
        else:
            # Fallback to base adapter for unknown types
            adapter_class = BaseModelAdapter

        return adapter_class(
            self.model,
            self.config.EXPECTED_INPUTS,
            self.config.get_input_signature(param),
        )


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
    from keras_hub.src.export.base import ExporterRegistry

    # Get the appropriate configuration for this model
    config = ExporterRegistry.get_config_for_model(model)

    # Create and use the LiteRT exporter
    exporter = LiteRTExporter(config, **kwargs)
    exporter.export(filepath)
