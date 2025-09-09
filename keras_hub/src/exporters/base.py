"""Base classes for Keras-Hub model exporters.

This module provides the foundation for exporting Keras-Hub models to various formats.
It follows the Optimum pattern of having different exporters for different model types and formats.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
import sys

# Add the keras path to import from local repo  
sys.path.insert(0, '/Users/hellorahul/Projects/keras')

try:
    import keras
    from keras.src.export.export_utils import get_input_signature
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    keras = None


class KerasHubExporterConfig(ABC):
    """Base configuration class for Keras-Hub model exporters.
    
    This class defines the interface for exporter configurations that specify
    how different types of Keras-Hub models should be exported.
    """
    
    # Model type this exporter handles (e.g., "causal_lm", "text_classifier")
    MODEL_TYPE: str = None
    
    # Expected input structure for this model type
    EXPECTED_INPUTS: List[str] = []
    
    # Default sequence length if not specified
    DEFAULT_SEQUENCE_LENGTH: int = 128
    
    def __init__(self, model, **kwargs):
        """Initialize the exporter configuration.
        
        Args:
            model: The Keras-Hub model to export
            **kwargs: Additional configuration parameters
        """
        self.model = model
        self.config = kwargs
        self._validate_model()
        
    def _validate_model(self):
        """Validate that the model is compatible with this exporter."""
        if not self._is_model_compatible():
            raise ValueError(
                f"Model {type(self.model)} is not compatible with "
                f"{self.__class__.__name__} (expected {self.MODEL_TYPE})"
            )
    
    @abstractmethod
    def _is_model_compatible(self) -> bool:
        """Check if the model is compatible with this exporter."""
        pass
    
    @abstractmethod
    def get_input_signature(self, sequence_length: Optional[int] = None) -> Dict[str, Any]:
        """Get the input signature for the model.
        
        Args:
            sequence_length: Optional sequence length override
            
        Returns:
            Dictionary mapping input names to their specifications
        """
        pass
    
    def get_dummy_inputs(self, sequence_length: Optional[int] = None) -> Dict[str, Any]:
        """Generate dummy inputs for model tracing.
        
        Args:
            sequence_length: Optional sequence length override
            
        Returns:
            Dictionary of dummy inputs for the model
        """
        if sequence_length is None:
            if hasattr(self.model, 'preprocessor') and self.model.preprocessor:
                sequence_length = getattr(self.model.preprocessor, 'sequence_length', self.DEFAULT_SEQUENCE_LENGTH)
            else:
                sequence_length = self.DEFAULT_SEQUENCE_LENGTH
        
        dummy_inputs = {}
        
        if "token_ids" in self.EXPECTED_INPUTS:
            dummy_inputs["token_ids"] = keras.ops.ones((1, sequence_length), dtype='int32')
        if "padding_mask" in self.EXPECTED_INPUTS:
            dummy_inputs["padding_mask"] = keras.ops.ones((1, sequence_length), dtype='bool')
        if "encoder_token_ids" in self.EXPECTED_INPUTS:
            dummy_inputs["encoder_token_ids"] = keras.ops.ones((1, sequence_length), dtype='int32')
        if "encoder_padding_mask" in self.EXPECTED_INPUTS:
            dummy_inputs["encoder_padding_mask"] = keras.ops.ones((1, sequence_length), dtype='bool')
        if "decoder_token_ids" in self.EXPECTED_INPUTS:
            dummy_inputs["decoder_token_ids"] = keras.ops.ones((1, sequence_length), dtype='int32')
        if "decoder_padding_mask" in self.EXPECTED_INPUTS:
            dummy_inputs["decoder_padding_mask"] = keras.ops.ones((1, sequence_length), dtype='bool')
            
        return dummy_inputs


class KerasHubExporter(ABC):
    """Base class for Keras-Hub model exporters.
    
    This class provides the common interface for exporting Keras-Hub models
    to different formats (LiteRT, ONNX, etc.).
    """
    
    def __init__(self, config: KerasHubExporterConfig, **kwargs):
        """Initialize the exporter.
        
        Args:
            config: Exporter configuration specifying model type and parameters
            **kwargs: Additional exporter-specific parameters
        """
        self.config = config
        self.model = config.model
        self.export_kwargs = kwargs
        
    @abstractmethod
    def export(self, filepath: str) -> None:
        """Export the model to the specified filepath.
        
        Args:
            filepath: Path where to save the exported model
        """
        pass
    
    def _ensure_model_built(self, sequence_length: Optional[int] = None):
        """Ensure the model is properly built with correct input structure.
        
        Args:
            sequence_length: Optional sequence length for dummy inputs
        """
        if not self.model.built:
            print("🔧 Building model with sample inputs...")
            
            dummy_inputs = self.config.get_dummy_inputs(sequence_length)
            
            try:
                # Build the model with the correct input structure
                _ = self.model(dummy_inputs, training=False)
                print("✅ Model built successfully")
            except Exception as e:
                print(f"⚠️  Model building failed: {e}")
                # Try alternative approach
                try:
                    input_shapes = {key: tensor.shape for key, tensor in dummy_inputs.items()}
                    self.model.build(input_shape=input_shapes)
                    print("✅ Model built using .build() method")
                except Exception as e2:
                    print(f"❌ Alternative building method also failed: {e2}")
                    raise


class ExporterRegistry:
    """Registry for mapping model types to their appropriate exporters."""
    
    _configs = {}
    _exporters = {}
    
    @classmethod
    def register_config(cls, model_type: str, config_class: type):
        """Register an exporter configuration for a model type.
        
        Args:
            model_type: The model type identifier (e.g., "causal_lm")
            config_class: The configuration class for this model type
        """
        cls._configs[model_type] = config_class
        
    @classmethod
    def register_exporter(cls, format_name: str, exporter_class: type):
        """Register an exporter for a specific format.
        
        Args:
            format_name: The export format identifier (e.g., "lite_rt")
            exporter_class: The exporter class for this format
        """
        cls._exporters[format_name] = exporter_class
    
    @classmethod
    def get_config_for_model(cls, model) -> KerasHubExporterConfig:
        """Get the appropriate configuration for a model.
        
        Args:
            model: The Keras-Hub model
            
        Returns:
            An appropriate exporter configuration
            
        Raises:
            ValueError: If no suitable configuration is found
        """
        # Try to detect model type
        model_type = cls._detect_model_type(model)
        
        if model_type not in cls._configs:
            raise ValueError(f"No exporter configuration found for model type: {model_type}")
            
        config_class = cls._configs[model_type]
        return config_class(model)
    
    @classmethod
    def get_exporter(cls, format_name: str, config: KerasHubExporterConfig, **kwargs):
        """Get an exporter for the specified format.
        
        Args:
            format_name: The export format
            config: The exporter configuration
            **kwargs: Additional parameters for the exporter
            
        Returns:
            An appropriate exporter instance
        """
        if format_name not in cls._exporters:
            raise ValueError(f"No exporter found for format: {format_name}")
            
        exporter_class = cls._exporters[format_name]
        return exporter_class(config, **kwargs)
    
    @classmethod
    def _detect_model_type(cls, model) -> str:
        """Detect the model type from the model instance.
        
        Args:
            model: The Keras-Hub model
            
        Returns:
            The detected model type
        """
        # Import here to avoid circular imports
        try:
            from keras_hub.src.models.causal_lm import CausalLM
            from keras_hub.src.models.seq_2_seq_lm import Seq2SeqLM
        except ImportError:
            CausalLM = None
            Seq2SeqLM = None
        
        model_class_name = model.__class__.__name__
        
        if CausalLM and isinstance(model, CausalLM):
            return "causal_lm"
        elif 'TextClassifier' in model_class_name:
            return "text_classifier"
        elif Seq2SeqLM and isinstance(model, Seq2SeqLM):
            return "seq2seq_lm"
        elif 'ImageClassifier' in model_class_name:
            return "image_classifier"
        else:
            # Fallback to text model if it has a preprocessor with tokenizer
            if hasattr(model, 'preprocessor') and model.preprocessor:
                if hasattr(model.preprocessor, 'tokenizer'):
                    return "text_model"
            
            return "unknown"
