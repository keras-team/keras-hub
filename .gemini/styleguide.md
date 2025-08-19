## Key Principles

- **Modularity**: Models are broken down into distinct, reusable components: Backbone, Converter (Tokenizer, ImageConverter, etc.), Preprocessor, and Task.
- **Consistency**: Strict adherence to naming and file structure conventions is crucial for predictability and maintainability.
- **Validation**: Every component must be rigorously validated against the original model's implementation. Numerical equivalence is a primary requirement, demonstrated via Colab notebooks.
- **Reusability**: Prioritize using existing layers from `keras.layers` and `keras_hub.layers` before implementing custom logic.
- **Backend Agnostic**: All code must be backend-agnostic, supporting TensorFlow, JAX, and PyTorch backends.

## Directory and File Structure

All new model contributions must follow a standardized directory and file structure within `keras_hub/src/models/`. For a model named `MyModel`, the structure must be:

```
keras_hub/
└── src/
    └── models/
        └── my_model/
            ├── __init__.py
            ├── my_model_backbone.py
            ├── my_model_backbone_test.py
            ├── my_model_tokenizer.py             # For NLP models
            ├── my_model_tokenizer_test.py        # For NLP models
            ├── my_model_image_converter.py       # For Vision models
            ├── my_model_image_converter_test.py  # For Vision models
            ├── my_model_audio_converter.py       # For Audio models
            ├── my_model_audio_converter_test.py  # For Audio models
            ├── my_model_classifier.py            # Example task
            ├── my_model_classifier_test.py       # Example task test
            ├── my_model_preprocessor.py          # Preprocessor for all tasks
            ├── my_model_preprocessor_test.py
            └── my_model_presets.py
```

Checkpoint conversion scripts have their own location:
```
tools/
└── checkpoint_conversion/
    └── convert_my_model_checkpoints.py
```

For models being ported from HuggingFace, converters should be added to:
```
keras_hub/src/utils/transformers/
├── convert_my_model.py
└── convert_my_model_test.py
```

## Naming Conventions

### Files
- **Format**: All filenames must be lowercase with underscores (snake_case).
- **Pattern**: Follow the pattern `<model_name>_<component_type>.py`.
- **Examples**: `distil_bert_backbone.py`, `distil_bert_tokenizer.py`, `distil_bert_classifier_test.py`.

### Classes
- **Format**: All class names must use CapWords (PascalCase).
- **Pattern**: Follow the pattern `<ModelName><ComponentType>`.
- **Examples**: `DistilBertBackbone`, `DistilBertTokenizer`, `DistilBertClassifier`, `DistilBertPreprocessor`.

### Functions and Methods
- **Format**: Use lowercase with underscores (snake_case).
- **Examples**: `from_preset()`, `call()`, `predict()`.

### Model Inputs
Use standardized names for model input arguments to ensure interoperability:
- **Text Models**: `token_ids`, `padding_mask`
- **Image Models**: `pixel_values`
- **Audio Models**: `audio_features`

## Code Implementation Style

### Backbone Models (`<model_name>_backbone.py`)

**Structure**: The backbone model must be a class that inherits from `keras.Model`.

**Implementation**: Use the Keras Functional API to define the model graph inside the class `__init__` method.

**Reusability**: Prefer using layers from `keras.layers` and `keras_hub.layers`. Custom layers should only be implemented for significant architectural deviations not covered by existing Keras components.

**Example Structure**:
```python
@keras_hub_export("keras_hub.models.MyModelBackbone")
class MyModelBackbone(Backbone):
    """MyModel core network with hyperparameters.
    
    This backbone implements the base architecture for MyModel.
    
    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        hidden_dim: int. The size of the transformer hidden state.
        intermediate_dim: int. The output dimension of the first Dense layer.
        dropout: float. Dropout probability for the Transformer encoder.
        max_sequence_length: int. The maximum sequence length that this encoder
            can consume.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights.
        
    Example:
    ```python
    input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
    }
    
    # Pretrained MyModel backbone.
    model = keras_hub.models.MyModelBackbone.from_preset("my_model_base")
    model(input_data)
    ```
    """
    
    def __init__(
        self,
        vocabulary_size,
        num_layers,
        # ... other parameters
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        
        # === Layers ===
        self.token_embedding = keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            name="token_embedding",
        )
        # ... other layers
        
        # === Functional Model ===
        token_ids = keras.Input(shape=(None,), dtype="int32", name="token_ids")
        padding_mask = keras.Input(shape=(None,), dtype="int32", name="padding_mask")
        
        # ... model graph definition
        
        super().__init__(
            inputs={
                "token_ids": token_ids,
                "padding_mask": padding_mask,
            },
            outputs=outputs,
            **kwargs,
        )
```

### Data Converters (`<model_name>_tokenizer.py`, etc.)

**Purpose**: Converters transform raw data (text, images, audio) into a numerical format. They handle tasks like vocabulary mapping, resizing, or feature extraction.

**Structure**:
- Text models use a `Tokenizer` class (e.g., `MyModelTokenizer`).
- Image models use an `ImageConverter` class (e.g., `MyModelImageConverter`).
- Audio models use an `AudioConverter` class (e.g., `MyModelAudioConverter`).

**Inheritance**: Subclass from the appropriate base class in KerasHub where available.

**Example Tokenizer**:
```python
@keras_hub_export(
    [
        "keras_hub.tokenizers.MyModelTokenizer",
        "keras_hub.models.MyModelTokenizer",
    ]
)
class MyModelTokenizer(WordPieceTokenizer):
    """A MyModel tokenizer using WordPiece subword segmentation.
    
    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_hub.tokenizers.WordPieceTokenizer`.
    
    Args:
        vocabulary: list of strings or str. A list of strings or a string filename path.
        lowercase: bool. If `True`, the input text will be first lowered before tokenization.
        
    Examples:
    ```python
    # Unbatched input.
    tokenizer = keras_hub.models.MyModelTokenizer.from_preset("my_model_base")
    tokenizer("The quick brown fox jumped.")
    ```
    """
    
    backbone_cls = MyModelBackbone
    
    def __init__(
        self,
        vocabulary=None,
        lowercase=False,
        **kwargs,
    ):
        super().__init__(vocabulary=vocabulary, lowercase=lowercase, **kwargs)
        
        # Add special tokens
        self.start_token_id = self.token_to_id("[CLS]")
        self.end_token_id = self.token_to_id("[SEP]")
        self.pad_token_id = self.token_to_id("[PAD]")
        self.mask_token_id = self.token_to_id("[MASK]")
```

### Preprocessors (`<model_name>_preprocessor.py`)

**Purpose**: A preprocessor is a `keras.layers.Layer` that orchestrates the entire preprocessing pipeline, turning raw user input into model-ready tensors.

**Structure**: It internally instantiates and uses the model's specific Converter (e.g., `MyModelTokenizer`).

**Functionality**: It handles padding, truncation, generating attention masks, and formatting the output into a dictionary of tensors that match the backbone's input signature (e.g., `{"token_ids": ..., "padding_mask": ...}`).

**Example Preprocessor**:
```python
@keras_hub_export("keras_hub.models.MyModelPreprocessor")
class MyModelPreprocessor(TextClassifierPreprocessor):
    """MyModel preprocessing for text classification.
    
    This preprocessing layer will prepare inputs for text classification.
    
    Args:
        tokenizer: `keras_hub.models.MyModelTokenizer`. A tokenizer instance.
        sequence_length: int. The length of the packed inputs.
        
    Examples:
    ```python
    preprocessor = keras_hub.models.MyModelPreprocessor.from_preset("my_model_base")
    preprocessor("The quick brown fox jumped.")
    ```
    """
    
    backbone_cls = MyModelBackbone
    tokenizer_cls = MyModelTokenizer
    
    def __init__(
        self,
        tokenizer,
        sequence_length=512,
        **kwargs,
    ):
        super().__init__(tokenizer=tokenizer, sequence_length=sequence_length, **kwargs)
```

### Task Models (`<model_name>_<task>.py`)

**Purpose**: A task model combines a Backbone, a Preprocessor, and a task-specific head (e.g., a classification or generation head).

**Structure**: It should be a class that inherits from `keras.Model`.

**API**: Provide a simple, high-level API for end-users, such as `predict()`, `fit()`, and `generate()`.

**Example Task Model**:
```python
@keras_hub_export("keras_hub.models.MyModelTextClassifier")
class MyModelTextClassifier(TextClassifier):
    """MyModel text classification model.
    
    This model combines a MyModel backbone with a classification head.
    
    Args:
        backbone: `keras_hub.models.MyModelBackbone`. A backbone instance.
        preprocessor: `keras_hub.models.MyModelPreprocessor`. A preprocessor instance.
        num_classes: int. Number of classes to predict.
        
    Examples:
    ```python
    classifier = keras_hub.models.MyModelTextClassifier.from_preset(
        "my_model_base",
        num_classes=2,
    )
    classifier.predict(["What an amazing movie!", "A total waste of my time."])
    ```
    """
    
    backbone_cls = MyModelBackbone
    preprocessor_cls = MyModelPreprocessor
    
    def __init__(
        self,
        backbone,
        preprocessor=None,
        num_classes=2,
        activation="softmax",
        **kwargs,
    ):
        # ... implementation
```

### Presets (`<model_name>_presets.py`)

**Purpose**: This file defines a dictionary of preset configurations for the model.

**Content**: Each entry includes the configuration arguments for the model (`config`), a description, and the URL to the pre-trained weights hosted on Kaggle (`weights_url`).

**Example Presets**:
```python
"""MyModel preset configurations."""

backbone_presets = {
    "my_model_base_en": {
        "metadata": {
            "description": "Base MyModel model trained on English text.",
            "params": 110000000,
            "path": "my_model",
        },
        "kaggle_handle": "kaggle://keras/my_model/keras/my_model_base_en/1",
    },
    "my_model_large_en": {
        "metadata": {
            "description": "Large MyModel model trained on English text.",
            "params": 340000000,
            "path": "my_model",
        },
        "kaggle_handle": "kaggle://keras/my_model/keras/my_model_large_en/1",
    },
}
```

## Docstrings and Type Hinting

### Docstrings
- Use Google-style docstrings for all public classes, methods, and functions.
- The first line should be a concise summary.
- Include comprehensive examples showing usage patterns.
- Document all parameters, return values, and exceptions.

### Type Hints
- KerasHub does not use type hints in function signatures or `__init__` methods.
- Type information is provided in the docstring Args section using the format `arg_name: type. description`.
- Focus on clear, descriptive parameter names and comprehensive docstrings.

**Example of good docstring with type hints in Args section**:
```python
def load_vocabulary(vocab_path):
    """Loads a vocabulary file into a dictionary.

    Args:
        vocab_path: str. The path to the vocabulary file. Each line in the
            file should contain a single token.

    Returns:
        A dictionary mapping tokens to their integer IDs.
        
    Raises:
        FileNotFoundError: If the vocabulary file does not exist.
    """
    vocab = {}
    with open(vocab_path, "r") as reader:
        for index, token in enumerate(reader):
            token = token.strip()
            vocab[token] = index
    return vocab
```

## Testing and Validation

### Testing Requirements
Testing is a non-negotiable part of every contribution. Beyond the existence of test files, the tests themselves must follow standardized routines to ensure all core functionality is covered.

### Unit Tests
**Requirement**: Every `.py` file containing logic (backbone, tokenizer, task, etc.) must have a corresponding `_test.py` file.

### Standardized Test Routines
KerasHub provides helper methods in the `TestCase` class that handle the standardized test routines. Users should use these methods instead of writing tests from scratch:

#### 1. Basic Usage and Shape Inference
**Method**: Use `self.run_backbone_test()` for backbone models or `self.run_layer_test()` for layers.
**Purpose**: Verifies that the model can be instantiated and called with valid inputs, checks output shapes, and runs additional validation.

#### 2. Variable Input Shapes
**Method**: Handled automatically by `self.run_backbone_test()` and `self.run_layer_test()`.
**Purpose**: Ensures the model works with dynamic input shapes (e.g., variable batch size or sequence length).

#### 3. from_preset() Functionality
**Method**: Use `self.run_preset_test()` for testing preset loading.
**Purpose**: Confirms that all model presets can be loaded correctly and produce expected outputs.

#### 4. Serialization (save() and load_model())
**Method**: Use `self.run_model_saving_test()` for testing model serialization.
**Purpose**: Guarantees that the model can be saved and reloaded without losing its state.

#### 5. Attached Preprocessor (for Task Models)
**Method**: Use `self.run_task_test()` for testing task models with preprocessors.
**Purpose**: Verifies the end-to-end functionality of a task model with raw inputs.

#### Available Test Helper Methods:
- `self.run_backbone_test()` - For backbone models
- `self.run_vision_backbone_test()` - For vision backbone models  
- `self.run_layer_test()` - For individual layers
- `self.run_preprocessor_test()` - For preprocessors
- `self.run_task_test()` - For task models
- `self.run_preset_test()` - For testing preset loading
- `self.run_model_saving_test()` - For testing serialization

### Example Test Structure
```python
import pytest
from keras import ops

from keras_hub.src.models.my_model.my_model_backbone import MyModelBackbone
from keras_hub.src.tests.test_case import TestCase


class MyModelBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "hidden_dim": 16,
            "intermediate_dim": 32,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=MyModelBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 16),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MyModelBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=MyModelBackbone,
            preset="my_model_base",
            input_data=self.input_data,
            expected_output_shape=(2, 5, 16),
            expected_partial_output=ops.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in MyModelBackbone.presets:
            self.run_preset_test(
                cls=MyModelBackbone,
                preset=preset,
                input_data=self.input_data,
            )
```

## Validation Colab Notebooks

### Requirement
Each pull request must include links to Colab notebooks that demonstrate numerical equivalence with the original model's implementation.

### Structure
Provide separate Colabs for each major component:

1. **Backbone Validation**: Load original weights into your KerasHub backbone and show that for the same input tensor, the output tensor is numerically identical (or within a very small tolerance).

2. **Converter/Preprocessor Validation**: Show that your preprocessor pipeline produces the same token IDs, padding masks, or pixel values as the original library's preprocessing functions.

3. **End-to-End Validation**: Use `MyModelTask.from_preset()` to load your pre-trained model and run a full task (e.g., classification). The final output (e.g., logits, probabilities) must match the original model.

## Import Conventions

### Keras Imports
Prefer importing `keras` as a top-level object:
```python
import keras
from keras import ops
from keras import layers
```

❌ `tf.keras.activations.X`  
✅ `keras.activations.X`

❌ `layers.X`  
✅ `keras.layers.X` or `keras_hub.layers.X`

### KerasHub Imports
```python
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.tests.test_case import TestCase
```

## Layer Implementation Guidelines

### Ideal Layer Style
When writing a new KerasHub layer (or tokenizer or metric), follow these guidelines:

1. **Accept `**kwargs`** in `__init__` and forward this to the super class.
2. **Keep Python attributes** on the layer for each `__init__` argument to the layer. The name and value should match the passed value.
3. **Write a `get_config()`** which chains to super.
4. **Document thoroughly** including call behavior through a class level docstring.
5. **Include usage examples** using the full symbol location in `keras_hub`.
6. **Include reference citations** if applicable.

### Example Layer Implementation
```python
@keras_hub_export("keras_hub.layers.MyCustomLayer")
class MyCustomLayer(keras.layers.Layer):
    """A custom layer for specific functionality.

    This layer implements [specific functionality] as described in
    [reference paper]. It accepts [input description] and outputs
    [output description].

    Args:
        param1: int. Description of parameter 1.
        param2: str. Description of parameter 2.

    Example:
    ```python
    layer = keras_hub.layers.MyCustomLayer(param1=10, param2=20)
    output = layer(input_tensor)
    ```

    Reference:
     - [Author et al., Year](https://arxiv.org/abs/paper_id)
    """

    def __init__(self, param1, param2, **kwargs):
        super().__init__(**kwargs)
        self.param1 = param1
        self.param2 = param2

    def build(self, input_shape):
        # Layer building logic
        super().build(input_shape)

    def call(self, inputs):
        # Layer computation logic
        return processed_output

    def get_config(self):
        config = super().get_config()
        config.update({
            "param1": self.param1,
            "param2": self.param2,
        })
        return config
```

## Checkpoint Conversion

### Script Location
All checkpoint conversion scripts should be placed in `tools/checkpoint_conversion/`.

### Script Requirements
- Must be reusable and well-documented
- Should handle all presets for the model
- Must demonstrate numerical equivalence with original implementation
- Should include proper error handling and validation

### Example Conversion Script Structure
```python
"""Convert MyModel checkpoints from original format to KerasHub."""

import argparse
import json
import os

import keras
import numpy as np

from keras_hub.src.models.my_model.my_model_backbone import MyModelBackbone


def convert_checkpoint(checkpoint_path, output_dir, preset_name):
    """Convert a MyModel checkpoint to KerasHub format."""
    # Load original checkpoint
    # Convert weights to KerasHub format
    # Save in KerasHub format
    # Validate numerical equivalence
    pass


def main():
    parser = argparse.ArgumentParser(description="Convert MyModel checkpoints")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--preset_name", required=True)
    
    args = parser.parse_args()
    convert_checkpoint(args.checkpoint_path, args.output_dir, args.preset_name)


if __name__ == "__main__":
    main()
```

## HuggingFace Converters

### When to Add
If the model is being ported from HuggingFace, a converter must be added to `keras_hub/src/utils/transformers/`.

### Converter Structure
Each HuggingFace converter should include:

1. **Configuration conversion**: `convert_backbone_config()` function that maps HuggingFace config to KerasHub config
2. **Weight conversion**: `convert_weights()` function that maps HuggingFace weights to KerasHub weights
3. **Backbone class reference**: `backbone_cls` variable pointing to the KerasHub backbone class

### Example HuggingFace Converter
```python
"""Convert MyModel from HuggingFace format to KerasHub."""

import numpy as np

from keras_hub.src.models.my_model.my_model_backbone import MyModelBackbone
from keras_hub.src.utils.preset_utils import get_file

backbone_cls = MyModelBackbone


def convert_backbone_config(transformers_config):
    """Convert HuggingFace config to KerasHub config."""
    return {
        "vocabulary_size": transformers_config["vocab_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "num_heads": transformers_config["num_attention_heads"],
        "hidden_dim": transformers_config["hidden_size"],
        "intermediate_dim": transformers_config["intermediate_size"],
    }


def convert_weights(backbone, loader, transformers_config):
    """Convert HuggingFace weights to KerasHub weights."""
    # Embedding layer
    loader.port_weight(
        keras_variable=backbone.get_layer("token_embedding").embeddings,
        hf_weight_key="model.embed_tokens.weight",
    )
    
    # Transformer layers
    for i in range(backbone.num_layers):
        layer = backbone.get_layer(f"transformer_layer_{i}")
        hf_prefix = f"model.layers.{i}"
        
        # Attention weights
        loader.port_weight(
            keras_variable=layer.attention.query_dense.kernel,
            hf_weight_key=f"{hf_prefix}.self_attn.q_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.transpose(
                np.reshape(hf_tensor, (keras_shape[0], keras_shape[2], keras_shape[1])),
                axes=(0, 2, 1),
            ),
        )
        # ... additional weight mappings
```

### Converter Test Requirements
- Test that the converter can load HuggingFace models correctly
- Verify class detection works for both backbone and task models
- Test with `load_weights=False` to ensure config conversion works
- Include numerical equivalence tests when possible

### Example Converter Test
```python
"""Tests for MyModel HuggingFace converter."""

import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.my_model.my_model_backbone import MyModelBackbone
from keras_hub.src.models.my_model.my_model_text_classifier import MyModelTextClassifier
from keras_hub.src.models.text_classifier import TextClassifier
from keras_hub.src.tests.test_case import TestCase


class TestMyModelConverter(TestCase):
    @pytest.mark.large
    def test_convert_preset(self):
        model = MyModelTextClassifier.from_preset(
            "hf://huggingface/my-model-base", num_classes=2
        )
        prompt = "This is a test sentence."
        model.predict([prompt])

    @pytest.mark.large
    def test_class_detection(self):
        model = TextClassifier.from_preset(
            "hf://huggingface/my-model-base",
            num_classes=2,
            load_weights=False,
        )
        self.assertIsInstance(model, MyModelTextClassifier)
        
        model = Backbone.from_preset(
            "hf://huggingface/my-model-base",
            load_weights=False,
        )
        self.assertIsInstance(model, MyModelBackbone)
```

## Code Quality Standards

### Formatting
- Use `black` for code formatting
- Follow PEP 8 guidelines
- Use consistent indentation (4 spaces)

### Error Handling
- Provide meaningful error messages
- Use appropriate exception types
- Include context in error messages

### Performance
- Support XLA compilation where applicable
- Use efficient data structures and algorithms

### Documentation
- All public APIs must be documented
- Include comprehensive examples
- Document edge cases and limitations
- Keep documentation up-to-date with code changes
