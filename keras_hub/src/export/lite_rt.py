"""LiteRT exporter for Keras-Hub models.

This module provides LiteRT export functionality specifically designed for
Keras-Hub models, handling their unique input structures and requirements.
"""

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.export.base import KerasHubExporter

try:
    from keras.src.export.lite_rt_exporter import (
        LiteRTExporter as KerasLiteRTExporter,
    )

    KERAS_LITE_RT_AVAILABLE = True
except ImportError:
    KERAS_LITE_RT_AVAILABLE = False
    KerasLiteRTExporter = None


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
        verbose=False,
        **kwargs,
    ):
        """Initialize the LiteRT exporter.

        Args:
            config: Exporter configuration for the model
            max_sequence_length: Maximum sequence length for conversion
            aot_compile_targets: List of AOT compilation targets
            verbose: Enable verbose logging
            **kwargs: Additional arguments passed to the underlying exporter
        """
        super().__init__(config, **kwargs)

        if not KERAS_LITE_RT_AVAILABLE:
            raise ImportError(
                "Keras LiteRT exporter is not available. "
                "Make sure you have Keras with LiteRT support installed."
            )

        self.max_sequence_length = max_sequence_length
        self.aot_compile_targets = aot_compile_targets
        self.verbose = verbose

        # Get sequence length from model if not provided
        if self.max_sequence_length is None:
            if hasattr(self.model, "preprocessor") and self.model.preprocessor:
                self.max_sequence_length = getattr(
                    self.model.preprocessor,
                    "sequence_length",
                    self.config.DEFAULT_SEQUENCE_LENGTH,
                )
            else:
                self.max_sequence_length = self.config.DEFAULT_SEQUENCE_LENGTH

    def export(self, filepath):
        """Export the Keras-Hub model to LiteRT format.

        Args:
            filepath: Path where to save the exported model (without extension)
        """
        if self.verbose:
            print(f"Starting LiteRT export for {self.config.MODEL_TYPE} model")

        # Ensure model is built with correct input structure
        # For text models, use sequence length; for image models, use None to auto-detect
        if self.config.MODEL_TYPE in ["causal_lm", "text_classifier", "seq2seq_lm"]:
            build_param = self.max_sequence_length
        else:
            build_param = None  # Let image models auto-detect from preprocessor
            
        self._ensure_model_built(build_param)

        # Get the proper input signature for this model type
        # For text models, pass sequence length; for image models, pass None to auto-detect
        if self.config.MODEL_TYPE in ["causal_lm", "text_classifier", "seq2seq_lm"]:
            signature_param = self.max_sequence_length
        else:
            signature_param = None  # Let image models auto-detect from preprocessor
            
        input_signature = self.config.get_input_signature(signature_param)

        # Create a wrapper that adapts the Keras-Hub model to work with Keras
        # LiteRT exporter
        wrapped_model = self._create_export_wrapper()

        # Create the Keras LiteRT exporter with the wrapped model
        keras_exporter = KerasLiteRTExporter(
            wrapped_model,
            input_signature=input_signature,
            aot_compile_targets=self.aot_compile_targets,
            verbose=1 if self.verbose else 0,
            **self.export_kwargs,
        )

        try:
            # Export using the Keras exporter
            keras_exporter.export(filepath)

            if self.verbose:
                print(f"Export completed successfully to: {filepath}.tflite")

        except Exception as e:
            raise RuntimeError(f"LiteRT export failed: {e}") from e
            keras_exporter.export(filepath)

            if self.verbose:
                print("✅ Export completed successfully!")
                print(f"📁 Model saved to: {filepath}.tflite")

        except Exception as e:
            if self.verbose:
                print(f"❌ Export failed: {e}")
            raise

    def _create_export_wrapper(self):
        """Create a wrapper model that handles the input structure conversion.

        This wrapper converts between the list-based inputs that Keras LiteRT
        exporter provides and the dictionary-based inputs that Keras-Hub models
        expect.
        """
        import keras

        class KerasHubModelWrapper(keras.Model):
            """Wrapper that adapts Keras-Hub models for export."""

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
                        # Ensure we preserve the correct dtype
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

            def call(self, inputs, training=None, mask=None):
                """Convert list inputs to dictionary format and call the
                original model."""
                if isinstance(inputs, dict):
                    # Already in dictionary format
                    return self.keras_hub_model(
                        inputs, training=training, mask=mask
                    )

                # Convert list inputs to dictionary format
                if not isinstance(inputs, (list, tuple)):
                    inputs = [inputs]

                # For image classifiers, try the direct tensor approach first
                # since most Keras-Hub vision models expect single tensor inputs
                if len(self.expected_inputs) == 1 and self.expected_inputs[0] == "images":
                    try:
                        return self.keras_hub_model(
                            inputs[0], training=training, mask=mask
                        )
                    except Exception:
                        # Fall back to dictionary approach if that fails
                        pass

                # For LiteRT export, we need to handle the fact that different
                # Keras Hub models expect inputs in different formats. Some
                # expect dictionaries, others expect single tensors.
                try:
                    # First, try mapping to the expected input names (dictionary format)
                    input_dict = {}
                    if len(self.expected_inputs) == 1:
                        input_dict[self.expected_inputs[0]] = inputs[0]
                    else:
                        for i, input_name in enumerate(self.expected_inputs):
                            input_dict[input_name] = inputs[i]
                    
                    return self.keras_hub_model(
                        input_dict, training=training, mask=mask
                    )
                except ValueError as e:
                    error_msg = str(e)
                    # If that fails, try direct tensor input (positional format)
                    if ("doesn't match the expected structure" in error_msg and 
                        "Expected: keras_tensor" in error_msg):
                        # The model expects a single tensor, not a dictionary
                        if len(inputs) == 1:
                            return self.keras_hub_model(
                                inputs[0], training=training, mask=mask
                            )
                        else:
                            # Multiple inputs - try as positional arguments
                            return self.keras_hub_model(
                                *inputs, training=training, mask=mask
                            )
                    elif "Missing data for input" in error_msg:
                        # Extract the actual expected input names from the error
                        if "Expected the following keys:" in error_msg:
                            # Parse the expected keys from error message
                            start = error_msg.find("Expected the following keys: [")
                            if start != -1:
                                start += len("Expected the following keys: [")
                                end = error_msg.find("]", start)
                                if end != -1:
                                    keys_str = error_msg[start:end]
                                    actual_input_names = [k.strip().strip("'\"") for k in keys_str.split(",")]
                                    
                                    # Map inputs to actual expected names
                                    input_dict = {}
                                    for i, actual_name in enumerate(actual_input_names):
                                        if i < len(inputs):
                                            input_dict[actual_name] = inputs[i]
                                    
                                    return self.keras_hub_model(
                                        input_dict, training=training, mask=mask
                                    )
                    
                    # If we still can't figure it out, re-raise the original error
                    raise

            def get_config(self):
                """Return the configuration of the wrapped model."""
                return self.keras_hub_model.get_config()

        # Pass the correct parameter based on model type
        if self.config.MODEL_TYPE in ["causal_lm", "text_classifier", "seq2seq_lm"]:
            signature_param = self.max_sequence_length
        else:
            signature_param = None  # Let image models auto-detect from preprocessor
            
        return KerasHubModelWrapper(
            self.model,
            self.config.EXPECTED_INPUTS,
            self.config.get_input_signature(signature_param),
        )


# Convenience function for direct export
def export_lite_rt(model, filepath, **kwargs):
    """Export a Keras-Hub model to LiteRT format.

    This is a convenience function that automatically detects the model type
    and exports it using the appropriate configuration.

    Args:
        model: The Keras-Hub model to export
        filepath: Path where to save the exported model (without extension)
        **kwargs: Additional arguments passed to the exporter
    """
    from keras_hub.src.export.base import ExporterRegistry

    # Get the appropriate configuration for this model
    config = ExporterRegistry.get_config_for_model(model)

    # Create and use the LiteRT exporter
    exporter = LiteRTExporter(config, **kwargs)
    exporter.export(filepath)
