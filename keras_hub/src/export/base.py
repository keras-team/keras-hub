"""Base classes for Keras-Hub model exporters.

This module provides the foundation for exporting Keras-Hub models to various formats.
It follows the Optimum pattern of having different exporters for different model types and formats.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type

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
        self.config_kwargs = kwargs
        self._validate_model()
        
    def _validate_model(self):
        """Validate that the model is compatible with this exporter."""
        if not self._is_model_compatible():
            raise ValueError(
                f"Model {self.model.__class__.__name__} is not compatible "
                f"with {self.__class__.__name__}"
            )
    
    @abstractmethod
    def _is_model_compatible(self) -> bool:
        """Check if the model is compatible with this exporter."""
        pass
    
    @abstractmethod
    def get_input_signature(self, sequence_length: Optional[int] = None) -> Dict[str, Any]:
        """Get the input signature for this model type.
        
        Args:
            sequence_length: Optional sequence length for input tensors
            
        Returns:
            Dictionary mapping input names to their signatures
        """
        pass
    
    def get_dummy_inputs(self, sequence_length: Optional[int] = None) -> Dict[str, Any]:
        """Generate dummy inputs for model building and testing.
        
        Args:
            sequence_length: Optional sequence length for dummy inputs
            
        Returns:
            Dictionary of dummy inputs
        """
        if sequence_length is None:
            sequence_length = self.DEFAULT_SEQUENCE_LENGTH
            
        dummy_inputs = {}
        
        # Common inputs for most Keras-Hub models
        if "token_ids" in self.EXPECTED_INPUTS:
            dummy_inputs["token_ids"] = keras.ops.ones((1, sequence_length), dtype='int32')
        if "padding_mask" in self.EXPECTED_INPUTS:
            dummy_inputs["padding_mask"] = keras.ops.ones((1, sequence_length), dtype='bool')
        
        # Encoder-decoder specific inputs
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
    
    def _ensure_model_built(self, sequence_length: Optional[int] = None) -> None:
        """Ensure the model is properly built with correct input structure.
        
        Args:
            sequence_length: Optional sequence length for dummy inputs
        """
        if not self.model.built:
            dummy_inputs = self.config.get_dummy_inputs(sequence_length)
            
            try:
                # Build the model with the correct input structure
                _ = self.model(dummy_inputs, training=False)
            except Exception as e:
                # Try alternative approach using build() method
                try:
                    input_shapes = {key: tensor.shape for key, tensor in dummy_inputs.items()}
                    self.model.build(input_shape=input_shapes)
                except Exception:
                    raise ValueError(
                        f"Failed to build model: {e}. Please ensure the model is properly constructed."
                    )


class ExporterRegistry:
    """Registry for mapping model types to their appropriate exporters."""
    
    _configs = {}
    _exporters = {}
    
    @classmethod
    def register_config(cls, model_type: str, config_class: Type[KerasHubExporterConfig]) -> None:
        """Register a configuration class for a model type.
        
        Args:
            model_type: The model type (e.g., "causal_lm")
            config_class: The configuration class
        """
        cls._configs[model_type] = config_class
    
    @classmethod
    def register_exporter(cls, format_name: str, exporter_class: Type[KerasHubExporter]) -> None:
        """Register an exporter class for a format.
        
        Args:
            format_name: The export format (e.g., "lite_rt")
            exporter_class: The exporter class
        """
        cls._exporters[format_name] = exporter_class
    
    @classmethod
    def get_config_for_model(cls, model) -> KerasHubExporterConfig:
        """Get the appropriate configuration for a model.
        
        Args:
            model: The Keras-Hub model
            
        Returns:
            An appropriate exporter configuration instance
            
        Raises:
            ValueError: If no configuration is found for the model type
        """
        model_type = cls._detect_model_type(model)
        
        if model_type not in cls._configs:
            raise ValueError(f"No configuration found for model type: {model_type}")
            
        config_class = cls._configs[model_type]
        return config_class(model)
    
    @classmethod
    def get_exporter(cls, format_name: str, config: KerasHubExporterConfig, **kwargs) -> KerasHubExporter:
        """Get an exporter for the specified format.
        
        Args:
            format_name: The export format
            config: The exporter configuration
            **kwargs: Additional parameters for the exporter
            
        Returns:
            An appropriate exporter instance
            
        Raises:
            ValueError: If no exporter is found for the format
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
            # Default to text model for generic Keras-Hub models
            return "text_model"
