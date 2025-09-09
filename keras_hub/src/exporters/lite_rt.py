"""LiteRT exporter for Keras-Hub models.

This module provides LiteRT export functionality specifically designed for Keras-Hub models,
handling their unique input structures and requirements.
"""

import sys
from typing import Optional

# Add the keras path to import from local repo
sys.path.insert(0, '/Users/hellorahul/Projects/keras')

from keras_hub.src.exporters.base import KerasHubExporter, KerasHubExporterConfig
from keras_hub.src.api_export import keras_hub_export

try:
    from keras.src.export.lite_rt_exporter import LiteRTExporter as KerasLiteRTExporter
    KERAS_LITE_RT_AVAILABLE = True
except ImportError:
    KERAS_LITE_RT_AVAILABLE = False
    KerasLiteRTExporter = None


@keras_hub_export("keras_hub.exporters.LiteRTExporter")
class LiteRTExporter(KerasHubExporter):
    """LiteRT exporter for Keras-Hub models.
    
    This exporter handles the conversion of Keras-Hub models to TensorFlow Lite format,
    properly managing the dictionary input structures that Keras-Hub models expect.
    """
    
    def __init__(self, config: KerasHubExporterConfig, 
                 max_sequence_length: Optional[int] = None,
                 aot_compile_targets: Optional[list] = None,
                 verbose: Optional[int] = None,
                 **kwargs):
        """Initialize the LiteRT exporter.
        
        Args:
            config: Exporter configuration for the model
            max_sequence_length: Maximum sequence length for conversion
            aot_compile_targets: List of AOT compilation targets
            verbose: Verbosity level
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
        self.verbose = verbose or 0
        
        # Get sequence length from model if not provided
        if self.max_sequence_length is None:
            if hasattr(self.model, 'preprocessor') and self.model.preprocessor:
                self.max_sequence_length = getattr(
                    self.model.preprocessor, 
                    'sequence_length', 
                    self.config.DEFAULT_SEQUENCE_LENGTH
                )
            else:
                self.max_sequence_length = self.config.DEFAULT_SEQUENCE_LENGTH
    
    def export(self, filepath: str) -> None:
        """Export the Keras-Hub model to LiteRT format.
        
        Args:
            filepath: Path where to save the exported model (without extension)
        """
        if self.verbose:
            print(f"🚀 Starting LiteRT export for {self.config.MODEL_TYPE} model...")
            print(f"   Model: {type(self.model).__name__}")
            print(f"   Expected inputs: {self.config.EXPECTED_INPUTS}")
            print(f"   Sequence length: {self.max_sequence_length}")
        
        # Ensure model is built with correct input structure
        self._ensure_model_built(self.max_sequence_length)
        
        # Get the proper input signature for this model type
        input_signature = self.config.get_input_signature(self.max_sequence_length)
        
        if self.verbose:
            print(f"   Input signature: {list(input_signature.keys())}")
        
        # Create a wrapper that adapts the Keras-Hub model to work with Keras LiteRT exporter
        wrapped_model = self._create_export_wrapper()
        
        # Create the Keras LiteRT exporter with the wrapped model
        keras_exporter = KerasLiteRTExporter(
            wrapped_model,
            input_signature=input_signature,
            max_sequence_length=self.max_sequence_length,
            aot_compile_targets=self.aot_compile_targets,
            verbose=self.verbose,
            **self.export_kwargs
        )
        
        try:
            # Export using the Keras exporter
            keras_exporter.export(filepath)
            
            if self.verbose:
                print(f"✅ Export completed successfully!")
                print(f"📁 Model saved to: {filepath}.tflite")
                
        except Exception as e:
            if self.verbose:
                print(f"❌ Export failed: {e}")
            raise
    
    def _create_export_wrapper(self):
        """Create a wrapper model that handles the input structure conversion.
        
        This wrapper converts between the list-based inputs that Keras LiteRT exporter
        provides and the dictionary-based inputs that Keras-Hub models expect.
        """
        import keras
        
        class KerasHubModelWrapper(keras.Model):
            """Wrapper that adapts Keras-Hub models for export."""
            
            def __init__(self, keras_hub_model, expected_inputs, input_signature, verbose=False):
                super().__init__()
                self.keras_hub_model = keras_hub_model
                self.expected_inputs = expected_inputs
                self.input_signature = input_signature
                self.verbose = verbose
                
                # Create Input layers based on the input signature
                self._input_layers = []
                for input_name in expected_inputs:
                    if input_name in input_signature:
                        spec = input_signature[input_name]
                        # Ensure we preserve the correct dtype
                        input_layer = keras.layers.Input(
                            shape=spec.shape[1:],  # Remove batch dimension
                            dtype=spec.dtype,
                            name=input_name
                        )
                        self._input_layers.append(input_layer)
                        
                        if self.verbose:
                            print(f"Created input layer: {input_name} - shape={spec.shape} dtype={spec.dtype}")
                
                # Store references to the original model's variables
                self._variables = keras_hub_model.variables
                self._trainable_variables = keras_hub_model.trainable_variables
                self._non_trainable_variables = keras_hub_model.non_trainable_variables
                
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
                """Convert list inputs to dictionary format and call the original model."""
                if isinstance(inputs, dict):
                    # Already in dictionary format
                    return self.keras_hub_model(inputs, training=training, mask=mask)
                
                # Convert list inputs to dictionary format
                if not isinstance(inputs, (list, tuple)):
                    inputs = [inputs]
                
                # Map inputs to expected dictionary structure
                input_dict = {}
                for i, input_name in enumerate(self.expected_inputs):
                    if i < len(inputs):
                        input_dict[input_name] = inputs[i]
                    else:
                        # Handle missing inputs - this shouldn't happen but let's be safe
                        print(f"⚠️  Missing input for {input_name}")
                        
                return self.keras_hub_model(input_dict, training=training, mask=mask)
            
            def get_config(self):
                """Return the configuration of the wrapped model."""
                return self.keras_hub_model.get_config()
        
        return KerasHubModelWrapper(
            self.model, 
            self.config.EXPECTED_INPUTS, 
            self.config.get_input_signature(self.max_sequence_length),
            verbose=self.verbose
        )


# Convenience function for direct export
@keras_hub_export("keras_hub.exporters.export_lite_rt")
def export_lite_rt(model, filepath: str, **kwargs) -> None:
    """Export a Keras-Hub model to LiteRT format.
    
    This is a convenience function that automatically detects the model type
    and exports it using the appropriate configuration.
    
    Args:
        model: The Keras-Hub model to export
        filepath: Path where to save the exported model (without extension)
        **kwargs: Additional arguments passed to the exporter
    """
    from keras_hub.src.exporters.base import ExporterRegistry
    
    # Get the appropriate configuration for this model
    config = ExporterRegistry.get_config_for_model(model)
    
    # Create and use the LiteRT exporter
    exporter = LiteRTExporter(config, **kwargs)
    exporter.export(filepath)
