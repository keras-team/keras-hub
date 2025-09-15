"""Configuration classes for different Keras-Hub model types.

This module provides specific configurations for exporting different types
of Keras-Hub models, following the Optimum pattern.
"""

from typing import Dict, Any, Optional
from keras_hub.src.export.base import KerasHubExporterConfig
from keras_hub.src.api_export import keras_hub_export


@keras_hub_export("keras_hub.export.CausalLMExporterConfig")
class CausalLMExporterConfig(KerasHubExporterConfig):
    """Exporter configuration for Causal Language Models (GPT, LLaMA, etc.)."""
    
    MODEL_TYPE = "causal_lm"
    EXPECTED_INPUTS = ["token_ids", "padding_mask"]
    DEFAULT_SEQUENCE_LENGTH = 128
    
    def _is_model_compatible(self) -> bool:
        """Check if model is a causal language model."""
        try:
            from keras_hub.src.models.causal_lm import CausalLM
            return isinstance(self.model, CausalLM)
        except ImportError:
            # Fallback to class name checking
            return 'CausalLM' in self.model.__class__.__name__
    
    def get_input_signature(self, sequence_length: Optional[int] = None) -> Dict[str, Any]:
        """Get input signature for causal LM models.
        
        Args:
            sequence_length: Optional sequence length. If None, will be inferred from model.
            
        Returns:
            Dictionary mapping input names to their specifications
        """
        if sequence_length is None:
            sequence_length = self._get_sequence_length()
        
        import keras
        return {
            "token_ids": keras.layers.InputSpec(
                shape=(None, sequence_length), 
                dtype='int32',
                name='token_ids'
            ),
            "padding_mask": keras.layers.InputSpec(
                shape=(None, sequence_length), 
                dtype='bool',
                name='padding_mask'
            )
        }
    
    def _get_sequence_length(self) -> int:
        """Get sequence length from model or use default."""
        if hasattr(self.model, 'preprocessor') and self.model.preprocessor:
            return getattr(self.model.preprocessor, 'sequence_length', self.DEFAULT_SEQUENCE_LENGTH)
        return self.DEFAULT_SEQUENCE_LENGTH


@keras_hub_export("keras_hub.export.TextClassifierExporterConfig")
class TextClassifierExporterConfig(KerasHubExporterConfig):
    """Exporter configuration for Text Classification models."""
    
    MODEL_TYPE = "text_classifier"
    EXPECTED_INPUTS = ["token_ids", "padding_mask"]
    DEFAULT_SEQUENCE_LENGTH = 128
    
    def _is_model_compatible(self) -> bool:
        """Check if model is a text classifier."""
        return 'TextClassifier' in self.model.__class__.__name__
    
    def get_input_signature(self, sequence_length: Optional[int] = None) -> Dict[str, Any]:
        """Get input signature for text classifier models.
        
        Args:
            sequence_length: Optional sequence length. If None, will be inferred from model.
            
        Returns:
            Dictionary mapping input names to their specifications
        """
        if sequence_length is None:
            sequence_length = self._get_sequence_length()
        
        import keras
        return {
            "token_ids": keras.layers.InputSpec(
                shape=(None, sequence_length), 
                dtype='int32',
                name='token_ids'
            ),
            "padding_mask": keras.layers.InputSpec(
                shape=(None, sequence_length), 
                dtype='bool',
                name='padding_mask'
            )
        }
    
    def _get_sequence_length(self) -> int:
        """Get sequence length from model or use default."""
        if hasattr(self.model, 'preprocessor') and self.model.preprocessor:
            return getattr(self.model.preprocessor, 'sequence_length', self.DEFAULT_SEQUENCE_LENGTH)
        return self.DEFAULT_SEQUENCE_LENGTH


@keras_hub_export("keras_hub.export.Seq2SeqLMExporterConfig")
class Seq2SeqLMExporterConfig(KerasHubExporterConfig):
    """Exporter configuration for Sequence-to-Sequence Language Models."""
    
    MODEL_TYPE = "seq2seq_lm"
    EXPECTED_INPUTS = ["encoder_token_ids", "encoder_padding_mask", "decoder_token_ids", "decoder_padding_mask"]
    DEFAULT_SEQUENCE_LENGTH = 128
    
    def _is_model_compatible(self) -> bool:
        """Check if model is a seq2seq language model."""
        try:
            from keras_hub.src.models.seq_2_seq_lm import Seq2SeqLM
            return isinstance(self.model, Seq2SeqLM)
        except ImportError:
            return 'Seq2SeqLM' in self.model.__class__.__name__
    
    def get_input_signature(self, sequence_length: Optional[int] = None) -> Dict[str, Any]:
        """Get input signature for seq2seq models.
        
        Args:
            sequence_length: Optional sequence length. If None, will be inferred from model.
            
        Returns:
            Dictionary mapping input names to their specifications
        """
        if sequence_length is None:
            sequence_length = self._get_sequence_length()
        
        import keras
        return {
            "encoder_token_ids": keras.layers.InputSpec(
                shape=(None, sequence_length), 
                dtype='int32',
                name='encoder_token_ids'
            ),
            "encoder_padding_mask": keras.layers.InputSpec(
                shape=(None, sequence_length), 
                dtype='bool',
                name='encoder_padding_mask'
            ),
            "decoder_token_ids": keras.layers.InputSpec(
                shape=(None, sequence_length), 
                dtype='int32',
                name='decoder_token_ids'
            ),
            "decoder_padding_mask": keras.layers.InputSpec(
                shape=(None, sequence_length), 
                dtype='bool',
                name='decoder_padding_mask'
            )
        }
    
    def _get_sequence_length(self) -> int:
        """Get sequence length from model or use default."""
        if hasattr(self.model, 'preprocessor') and self.model.preprocessor:
            return getattr(self.model.preprocessor, 'sequence_length', self.DEFAULT_SEQUENCE_LENGTH)
        return self.DEFAULT_SEQUENCE_LENGTH


@keras_hub_export("keras_hub.export.TextModelExporterConfig")
class TextModelExporterConfig(KerasHubExporterConfig):
    """Generic exporter configuration for text models."""
    
    MODEL_TYPE = "text_model"
    EXPECTED_INPUTS = ["token_ids", "padding_mask"]
    DEFAULT_SEQUENCE_LENGTH = 128
    
    def _is_model_compatible(self) -> bool:
        """Check if model is a text model (fallback)."""
        # This is a fallback config for text models that don't fit other categories
        return hasattr(self.model, 'preprocessor') and self.model.preprocessor and hasattr(self.model.preprocessor, 'tokenizer')
    
    def get_input_signature(self, sequence_length: Optional[int] = None) -> Dict[str, Any]:
        """Get input signature for generic text models.
        
        Args:
            sequence_length: Optional sequence length. If None, will be inferred from model.
            
        Returns:
            Dictionary mapping input names to their specifications
        """
        if sequence_length is None:
            sequence_length = self._get_sequence_length()
        
        import keras
        return {
            "token_ids": keras.layers.InputSpec(
                shape=(None, sequence_length), 
                dtype='int32',
                name='token_ids'
            ),
            "padding_mask": keras.layers.InputSpec(
                shape=(None, sequence_length), 
                dtype='bool',
                name='padding_mask'
            )
        }
    
    def _get_sequence_length(self) -> int:
        """Get sequence length from model or use default."""
        if hasattr(self.model, 'preprocessor') and self.model.preprocessor:
            return getattr(self.model.preprocessor, 'sequence_length', self.DEFAULT_SEQUENCE_LENGTH)
        return self.DEFAULT_SEQUENCE_LENGTH
