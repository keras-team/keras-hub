# LiteRT Model Export Design Document

**Feature:** Unified LiteRT Export for Keras and Keras-Hub  
**PRs:** [keras#21674](https://github.com/keras-team/keras/pull/21674), [keras-hub#2405](https://github.com/keras-team/keras-hub/pull/2405)  
**Status:** Implemented  
**Last Updated:** October 2025

---

## Quick Reference

**What is LiteRT?** LiteRT (formerly TensorFlow Lite) is TensorFlow's framework for deploying models on mobile, embedded, and edge devices with optimized inference.

**Minimal Export Example:**
```python
import keras
import keras_hub
import tensorflow as tf

# Keras Core model - must have at least one layer
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,))
])
model.export("model.tflite", format="litert")

# Keras-Hub model - from_preset() includes preprocessor
model = keras_hub.models.GemmaCausalLM.from_preset("gemma_2b")
model.export("model.tflite", max_sequence_length=128)

# With quantization (recommended for production)
model.export(
    "model_quantized.tflite",
    format="litert",
    litert_kwargs={
        "optimizations": [tf.lite.Optimize.DEFAULT]
    }
)
```

**When to Use:** Export Keras models to `.tflite` format for deployment on Android, iOS, or embedded devices. See Section 9 FAQ for deployment links.

---

## Glossary

| Term | Definition |
|------|------------|
| **LiteRT** | TensorFlow's lightweight runtime (formerly TensorFlow Lite) for mobile/edge inference |
| **Registry Pattern** | Design pattern that maps model types to their configuration handlers |
| **Adapter Pattern** | Wrapper that converts one interface (dict) to another (list) without changing the original |
| **AOT Compilation** | Ahead-Of-Time compilation optimizing `.tflite` models for specific hardware targets (arm64, x86_64, etc.) |
| **Functional Model** | Keras model created with `keras.Model(inputs, outputs)` - has static graph |
| **Sequential Model** | Keras model with linear layer stack: `keras.Sequential([layer1, layer2])` |
| **Subclassed Model** | Keras model with custom `call()` method - has dynamic behavior |
| **Input Signature** | Type specification defining tensor shapes and dtypes for model inputs |
| **Preprocessor** | Keras-Hub component that transforms raw data (text/images) into model inputs |
| **TF Select Ops** | TensorFlow operators not natively supported in TFLite - included as fallback for compatibility |
| **Quantization** | Process of reducing model precision (e.g., float32 → int8) to reduce size and improve performance |
| **Dynamic Range Quantization** | Post-training quantization converting weights to int8 while keeping activations in float (~75% size reduction) |
| **Full Integer Quantization** | Quantization converting both weights and activations to int8 (requires representative dataset) |
| **Representative Dataset** | Sample data used to calibrate quantization ranges for better accuracy |
| **litert_kwargs** | Dictionary parameter for passing TFLite converter options (optimizations, quantization, etc.) |

---

## Table of Contents

1. [Objective](#1-objective)
2. [Background](#2-background)
3. [Goals](#3-goals)
4. [Detailed Design](#4-detailed-design)
5. [Usage Examples](#5-usage-examples)
6. [Alternatives Considered](#6-alternatives-considered)
7. [Testing Strategy](#7-testing-strategy)
8. [Known Limitations](#8-known-limitations)
9. [FAQ](#9-faq)
10. [References](#10-references)

---

## 1. Objective

### 1.1 What

Enable seamless export of Keras and Keras-Hub models to LiteRT (TensorFlow Lite) format through a unified `model.export()` API, supporting deployment to mobile, embedded, and edge devices.

**Quick Example:**
```python
import keras
import keras_hub

# Keras model export
model = keras.Sequential([keras.layers.Dense(10, input_shape=(784,))])
model.export("model.tflite", format="litert")

# Keras-Hub model export
model = keras_hub.models.GemmaCausalLM.from_preset("gemma_2b")
model.export("model.tflite", max_sequence_length=128)
```

### 1.2 Why

**Problem Statement:**

**Problem Statement:**

Keras 3.x introduced multi-backend support (TensorFlow, JAX, PyTorch), breaking the existing TFLite export workflow from Keras 2.x. Additionally:
- Manual export required 5+ steps with TensorFlow Lite Converter
- Keras-Hub models use dictionary inputs incompatible with TFLite's list-based interface
- No unified API across Keras Core and Keras-Hub
- Error-prone manual configuration of converter settings

**Impact:**

Without this feature, users must manually handle SavedModel conversion, input signature wrapping, and adapter pattern implementation - a complex process requiring deep TensorFlow knowledge.

### 1.3 Target Audience

- **ML Engineers:** Deploying trained models to production
- **Mobile Developers:** Integrating `.tflite` models into apps
- **Backend Engineers:** Building automated export pipelines

**Prerequisites:** Basic familiarity with Keras model types and model deployment concepts.

---

## 2. Background

### 2.1 LiteRT (TensorFlow Lite) Overview

**What is LiteRT?** LiteRT (formerly TensorFlow Lite) is TensorFlow's framework for deploying ML models on mobile, embedded, and edge devices with optimized inference.

**Key Characteristics:**
- Optimized for on-device inference (low latency, small binary size)
- Supports Android, iOS, embedded Linux, microcontrollers
- Uses flatbuffer format (`.tflite` files)
- Requires positional (list-based) input arguments, not dictionary inputs

### 2.2 The Problem: Broken Export in Keras 3.x

**Before these PRs:**
```python
# Old way: Manual 5-step process (Keras 2.x or Keras 3.x)
import tensorflow as tf

# 1. Save model as SavedModel
model.save("temp_saved_model/", save_format="tf")

# 2. Load converter
converter = tf.lite.TFLiteConverter.from_saved_model("temp_saved_model/")

# 3. Configure converter (ops, optimization, etc.)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

# 4. Convert to TFLite bytes
tflite_model = converter.convert()

# 5. Write to file
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
```

**Issues with manual approach:**
- No native LiteRT export in Keras 3.x (SavedModel API changed)
- Keras-Hub models with dict inputs couldn't export (TFLite expects lists)
- Requires understanding TFLite converter internals
- No unified API across Keras Core and Keras-Hub

**After these PRs:**
```python
# New way: Single line
model.export("model.tflite", format="litert")
```

### 2.3 Key Challenges

1. **Dictionary Input Problem:** Keras-Hub models expect dictionary inputs like `{"token_ids": [...], "padding_mask": [...]}`, but TFLite requires positional list inputs
2. **Multi-Backend Compatibility:** Models trained with JAX or PyTorch backends need TensorFlow conversion for TFLite
3. **Input Signature Inference:** Different model types (Functional, Sequential, Subclassed) have different ways to introspect input shapes
4. **Code Organization:** Avoid duplication between Keras Core and Keras-Hub implementations

---

## 3. Goals

### 3.1 Primary Goals

1. **Unified API:** Single `model.export(filepath, format="litert")` works across all Keras and Keras-Hub models
2. **Zero Manual Configuration:** Automatic input signature inference, format detection, and converter setup
3. **Dict-to-List Conversion:** Transparent handling of Keras-Hub's dictionary inputs
4. **Backend Agnostic:** Export models trained with any backend (TensorFlow, JAX, PyTorch)

### 3.2 Non-Goals

- ONNX export (separate feature)
- Post-training quantization (use TFLite APIs directly)
- Custom operator registration (requires TFLite tooling)
- Runtime optimization tuning (TFLite's responsibility)

### 3.3 Success Metrics

- ✅ All Keras model types (Functional, Sequential, Subclassed) export successfully
- ✅ All Keras-Hub model types (text and vision tasks) export successfully
- ✅ Models trained with JAX/PyTorch export without manual TensorFlow conversion
- ✅ Zero-config export for 95%+ use cases (only edge cases need explicit configuration)

---

## 4. Detailed Design

### 4.1 System Architecture

The export system follows a **two-layer architecture**:

```
┌─────────────────────────────────────────────────────────┐
│                   User API Layer                        │
│  model.export(filepath, format="litert", **kwargs)      │
└───────────────────────┬─────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
┌───────▼──────────┐          ┌─────────▼──────────┐
│  Keras Core      │          │  Keras-Hub         │
│  LiteRTExporter  │          │  LiteRTExporter    │
└───────┬──────────┘          └─────────┬──────────┘
        │                               │
        │  Direct conversion            │  Wraps with adapter
        │                               │
        └───────────────┬───────────────┘
                        │
              ┌─────────▼──────────┐
              │  TFLite Converter  │
              │  (TensorFlow)      │
              └────────────────────┘
```

**Which Path Does My Model Take?**

| Your Model | Export Path | Reason |
|------------|-------------|--------|
| `keras.Model(...)` or `keras.Sequential(...)` | Keras Core → Direct | Standard Keras models with list/single inputs |
| Custom `class MyModel(keras.Model)` | Keras Core → Direct | Custom Keras model (non-Keras-Hub) |
| `keras_hub.models.GemmaCausalLM(...)` | Keras-Hub → Adapter → Core | Keras-Hub model with dict inputs |
| Keras-Hub Subclassed model | Keras-Hub → Adapter → Core | Inherits from Keras-Hub task classes |

**Key Principles:**

1. **Separation of Concerns:** Keras Core handles basic model types; Keras-Hub handles dict input conversion
2. **Adapter Pattern:** Keras-Hub wraps models to convert dictionary inputs to list inputs
3. **Composition:** Keras-Hub's exporter reuses Keras Core's exporter (no code duplication)
4. **Registry Pattern:** Automatic exporter selection based on `isinstance()` checks

**Important Notes:**

⚠️ **Adapter Overhead:** The adapter wrapper only exists during export. The generated `.tflite` file contains the original model weights - no runtime overhead.

⚠️ **Backend Compatibility:** Models can be trained with any backend (JAX, PyTorch, TensorFlow) and saved to `.keras` format. However, for LiteRT export, the model **must be loaded with TensorFlow backend** during conversion. The exporter handles tensor conversion transparently, but TensorFlow backend is required for TFLite compatibility. If your model uses operations not available in TensorFlow, you'll get a conversion error.

⚠️ **Op Compatibility:** Check if your layers use [TFLite-supported operations](https://www.tensorflow.org/lite/guide/ops_compatibility). Unsupported ops will cause conversion errors. Enable `verbose=True` during export to see which ops are problematic.

### 4.2 Keras Core Implementation

**Location:** `keras/src/export/litert.py`

**Responsibilities:**
- Export Functional, Sequential, and Subclassed Keras models
- Infer input signatures from model structure  
- Convert to TFLite using TensorFlow Lite Converter
- Support AOT compilation for hardware optimization

**Export Pipeline:**

```
┌─────────────┐
│   Model     │
│  (any type) │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ 1. Build Check      │ Ensure model has variables
│    model.built?     │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ 2. Input Signature  │ Infer or validate signature
│    get_signature()  │ • Functional: [nested_struct]
└──────┬──────────────┘ • Sequential: flat_inputs
       │                • Subclassed: recorded_shapes
       ▼
┌─────────────────────┐
│ 3. TFLite Convert   │ Model → bytes
│    Strategy:        │
│    ├─ Direct (try)  │
│    └─ Wrapper (fallback)
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ 4. Save File        │ Write .tflite
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ 5. AOT Compile      │ Optional hardware optimization
│    (optional)       │
└─────────────────────┘
```

### 4.3 Input Signature Strategy by Model Type

> **⚠️ CRITICAL: Functional Model Signature Wrapping**
> 
> Functional models with dictionary inputs require special handling: the signature must be wrapped in a single-element list `[input_signature_dict]` rather than passed directly as a dict. This is because Functional models' `call()` signature expects one positional argument containing the full nested structure, not multiple positional arguments.
> 
> **This is handled automatically** by the exporter - you don't need to do anything. This note explains why you might see `[{...}]` instead of `{...}` in logs or error messages.

**Design Decision:** Different model types have different call signatures, requiring type-specific handling.

| Model Type | Signature Format | Reason | Auto-Inference? |
|------------|-----------------|--------|-----------------|
| **Functional** | Single-element list `[nested_inputs]` | `call()` expects one positional arg with full structure | ✅ Yes (from `model.inputs`) |
| **Sequential** | Flat list `[input1, input2, ...]` | `call()` maps over inputs directly | ✅ Yes (from `model.inputs`) |
| **Subclassed** | Inferred from first call | Dynamic `call()` signature not statically known | ⚠️ Only if model built |

**When Auto-Inference Fails:**

Subclassed models that haven't been called cannot infer signature automatically. You'll see:
```
ValueError: Model must be built before export. Call model(inputs) or provide input_signature.
```

**Solution:** Build model first or provide explicit signature:
```python
# Option 1: Build by calling
model = MyCustomModel()
model(dummy_input)  # Now model.built == True
model.export("model.tflite")

# Option 2: Provide signature explicitly
model.export("model.tflite", input_signature=[InputSpec(shape=(None, 10))])
```

**Critical Insight (from PR review):**
> Functional models need single-element list wrapping because their `call()` signature is `call(inputs)` where `inputs` is the complete nested structure, not `call(*inputs)`.

### 4.4 Conversion Strategy Decision Tree

```
Model (any type)
  │
  ├─ STEP 1: Try Direct Conversion (all models)
  │  │
  │  ├─ TFLiteConverter.from_keras_model(model)
  │  ├─ Set supported ops (TFLite + TF Select)
  │  └─ converter.convert() → Success? Return bytes ✅
  │
  └─ STEP 2: If Direct Fails → Wrapper-based Conversion (fallback)
     │
     ├─ Wrap model in tf.Module
     ├─ Add @tf.function signature
     ├─ Handle backend tensor conversion
     └─ TFLiteConverter.from_concrete_functions()
```

**Important:** The code tries direct conversion first for ALL model types (Functional, Sequential, AND Subclassed). Wrapper-based conversion is only used as a fallback if direct conversion fails.

**Why Two Strategies?**

1. **Direct Conversion (attempted first):**
   - Simpler and faster path
   - Works for most well-formed models
   - TFLite converter directly inspects Keras model structure

2. **Wrapper-based (fallback when direct fails):**
   - Required when direct conversion encounters errors
   - Provides explicit concrete function with @tf.function
   - Handles edge cases and complex model structures
   - Multiple retry strategies for better compatibility

### 4.5 Backend Tensor Conversion

**Challenge:** Keras 3.x supports multiple backends (TensorFlow, JAX, PyTorch), but TFLite only accepts TensorFlow tensors.

**Solution Flow:**

```
Keras Backend Tensor
       │
       ▼
ops.convert_to_tensor()  ← Standardize to Keras tensor
       │
       ▼
Model Call
       │
       ▼
ops.convert_to_numpy()   ← Convert to numpy (universal)
       │
       ▼
tf.convert_to_tensor()   ← Convert to TensorFlow
       │
       ▼
TFLite Converter
```

This three-step conversion ensures compatibility across all Keras backends.

---

### 4.6 Keras-Hub Implementation

**Location:** `keras_hub/src/export/`

**Challenge:** Keras-Hub models use dictionary inputs, but TFLite expects positional list inputs.

**Solution:** Adapter Pattern + Registry Pattern

#### 4.6.1 Registry Pattern

```
┌──────────────────────────────────────────────┐
│           ExporterRegistry                   │
├──────────────────────────────────────────────┤
│                                              │
│  Model Classes → Config Classes              │
│  ├─ CausalLM → CausalLMExporterConfig        │
│  ├─ TextClassifier → TextClassifierConfig    │
│  ├─ ImageClassifier → ImageClassifierConfig  │
│  └─ ...                                      │
│                                              │
│  Formats → Exporter Classes                  │
│  └─ "litert" → LiteRTExporter                │
│                                              │
└──────────────────────────────────────────────┘

Usage:
  model = keras_hub.models.GemmaCausalLM(...)
       │
       ├─ Registry.get_config(model)
       │  └─ Returns: CausalLMExporterConfig
       │
       ├─ Registry.get_exporter("litert", config)
       │  └─ Returns: LiteRTExporter instance
       │
       └─ exporter.export("model.tflite")
```

**Why Registry?**
- ✅ Extensible: Add new model types without modifying core logic
- ✅ Maintainable: Config logic separated by model type
- ✅ Type-safe: Each model type has dedicated configuration

#### 4.6.2 Model Type Configurations

Each model type has a config class defining:
1. **EXPECTED_INPUTS**: Which inputs the model needs
2. **get_input_signature()**: How to create input specs
3. **Type-specific defaults**: e.g., sequence_length for text, image_size for vision

**What is a Preprocessor?**

A Keras-Hub preprocessor is a component that transforms raw data into model-ready tensors:
- **Text preprocessors**: Tokenize text → `token_ids` + `padding_mask`
- **Vision preprocessors**: Resize/normalize images → image tensors

Preprocessors store metadata (e.g., `sequence_length`, `image_size`) that export uses for signature inference.

**Configuration Matrix:**

| Model Type | Input Keys | Parameter | Default/Source | How to Set |
|------------|-----------|-----------|----------------|------------|
| **CausalLM** | `token_ids`, `padding_mask` | `sequence_length` | 128 or from preprocessor | `max_sequence_length=512` in export |
| **TextClassifier** | `token_ids`, `padding_mask` | `sequence_length` | 128 or from preprocessor | `max_sequence_length=512` in export |
| **Seq2SeqLM** | `encoder_*`, `decoder_*` (4 inputs) | `sequence_length` | 128 or from preprocessor | `max_sequence_length=512` in export |
| **ImageClassifier** | `images` | `image_size` | From preprocessor (required) | Auto-detected, cannot override |
| **ObjectDetector** | `images`, `image_shape` | `image_size` | From preprocessor (required) | Auto-detected, cannot override |
| **ImageSegmenter** | `images` | `image_size` | From preprocessor (required) | Auto-detected, cannot override |

**Sequence Length Priority (Text Models):**
1. User-specified `max_sequence_length` parameter (highest priority)
2. Preprocessor's `sequence_length` attribute (if available)
3. `DEFAULT_SEQUENCE_LENGTH = 128` (fallback)

**Example:**
```python
# Case 1: Inferred from preprocessor
model = keras_hub.models.GemmaCausalLM.from_preset("gemma_2b")  
# model.preprocessor.sequence_length = 8192
model.export("model.tflite")  # Uses 8192 ✅

# Case 2: Override with parameter
model.export("model.tflite", max_sequence_length=512)  # Uses 512 ✅

# Case 3: No preprocessor, no parameter
model_without_preprocessor.export("model.tflite")  # Uses 128 (default) ⚠️
```

**Design Note:** Text models have `DEFAULT_SEQUENCE_LENGTH` class constant; vision models infer from preprocessor.

#### 4.6.3 Adapter Pattern: Input Structure Conversion

**Core Innovation:** Wrap Keras-Hub model to change input interface without modifying model code.

```
┌─────────────────────────────────────────────────────────┐
│                  TextModelAdapter                       │
│  (Keras Model subclass)                                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  inputs (property):                                     │
│    └─ [Input("token_ids"), Input("padding_mask")]       │
│        ↑                                                │
│        │ Keras exporter sees list of Input layers       │
│        │                                                │
│  call(inputs: list):                                    │
│    ├─ Convert: [t1, t2] → {"token_ids": t1,             │
│    │                        "padding_mask": t2}         │
│    ├─ Call: keras_hub_model(inputs_dict)                │
│    └─ Return: output                                    │
│                                                         │
│  variables (property):                                  │
│    └─ keras_hub_model.variables  (direct reference)     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Why It Works:**
1. Keras Core exporter calls `adapter.inputs` → gets list of Input layers
2. TFLite converter creates list-based signature
3. **At export time**: Adapter is compiled into the `.tflite` file as the model's interface
4. **At inference time** (on mobile device): The `.tflite` model expects list inputs (no dict conversion needed - it's baked in)
5. No model code changes needed!

**Important Clarification:**
- **During export**: The adapter wraps the model temporarily to convert interfaces
- **In .tflite file**: The conversion is "compiled in" - the file's interface is list-based
- **During inference**: Your mobile app passes a list (no adapter exists at runtime)

#### 4.6.4 Export Flow Integration

```
User Code: model.export("model.tflite")
     │
     ▼
┌─────────────────────────────────────────┐
│ Keras-Hub Task.export()                 │
│  └─ calls export_model(model, filepath) │
└─────────┬───────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│ Registry: Get Config for Model          │
│  ├─ model is CausalLM                   │
│  └─ return CausalLMExporterConfig       │
└─────────┬───────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│ Config: Build Input Signature           │
│  ├─ Infer sequence_length from          │
│  │   preprocessor (if available)        │
│  └─ Create InputSpec for each input     │
└─────────┬───────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│ Create Adapter Wrapper                  │
│  ├─ TextModelAdapter                    │
│  ├─ Wrap original model                 │
│  └─ Convert dict → list interface       │
└─────────┬───────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│ Call Keras Core Exporter                │
│  └─ Pass wrapped model + list signature │
└─────────┬───────────────────────────────┘
          │
          ▼
     .tflite file
```

#### 4.6.5 Key Design Decisions

**1. Subclass Registration Order**

**Problem:** Seq2SeqLM inherits from CausalLM. How to select right config?

**Solution:** Register subclasses first
```python
# CORRECT order (subclass first)
ExporterRegistry.register_config(Seq2SeqLM, Seq2SeqLMExporterConfig)
ExporterRegistry.register_config(CausalLM, CausalLMExporterConfig)

# Registry checks isinstance() in order → returns first match
```

**2. Model Building Strategy**

**Problem:** Need model variables before export, but don't want to allocate memory for dummy data.

**Solution:** Use `model.build(input_shapes)` - creates variables without data allocation.

**3. Parameter Type Specialization**

**Design Choice:** Keep param types in specific configs, not base class.

```
Base Class (KerasHubExporterConfig)
  ├─ No param defaults ← model-agnostic
  │
  ├─ Text Configs (CausalLM, TextClassifier, Seq2SeqLM)
  │  └─ DEFAULT_SEQUENCE_LENGTH = 128
  │
  └─ Vision Configs (ImageClassifier, ObjectDetector, etc.)
     └─ No defaults (infer from preprocessor)
```

This keeps each model type self-contained and prevents inappropriate defaults.

---

### 4.7 Cross-Component Integration

**How Keras-Hub reuses Keras Core:**

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                        │
│                                                             │
│  User Code:                                                 │
│    model = keras_hub.models.GemmaCausalLM(...)              │
│    model.export("model.tflite")                             │
│                                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   KERAS-HUB LAYER                           │
│  (Handles complex models with dict inputs)                  │
│                                                             │
│  Registry Pattern:                                          │
│    ├─ Model type detection (CausalLM, TextClassifier, etc.) │
│    ├─ Config selection (input specs, defaults)              │
│    └─ Adapter creation (dict → list conversion)             │
│                                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ Delegates to:
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   KERAS CORE LAYER                          │
│  (Handles basic models with list/single inputs)             │
│                                                             │
│  Export Strategy:                                           │
│    ├─ Signature inference (Functional/Sequential)           │
│    ├─ Conversion logic (Direct vs Wrapper)                  │
│    └─ TFLite generation (tf.lite.TFLiteConverter)           │
│                                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
                  .tflite file
```

**Design Rationale:**
- **Separation of Concerns**: Keras Core handles basic export; Keras-Hub adds NLP/Vision preprocessing
- **Extensibility**: New model types added to Keras-Hub without modifying Core
- **Reusability**: Core exporter used by both layers

### 4.8 Critical Integration Points

**Integration Point 1: Input Signature Transformation**

```
Keras-Hub Creates:
  input_signature = {
    "token_ids": InputSpec(shape=(None, 128), dtype="int32"),
    "padding_mask": InputSpec(shape=(None, 128), dtype="int32")
  }

Adapter Transforms:
  keras_hub_model.inputs → TextModelAdapter.inputs
  └─ [Input("token_ids"), Input("padding_mask")]
     ↑
     List of Input layers (Keras Core expects this)

Keras Core Converts:
  [InputSpec, InputSpec] → tf.TensorSpec list
  └─ Used by TFLite converter
```

**Integration Point 2: Model Variable Sharing**

```python
# Keras-Hub creates adapter
adapter = TextModelAdapter(
    keras_hub_model,      # Original model
    expected_inputs,      # ["token_ids", "padding_mask"]
    input_signature       # InputSpec dict
)

# Critical: adapter.variables references original model
adapter.variables = keras_hub_model.variables
#                   ↑
#                   Same memory location - no copy!

# Keras Core exporter uses adapter.variables
keras_exporter = KerasLitertExporter(adapter, ...)
#                                     ↑
#                             Sees same variables as original
```

**Why This Matters:**
- ✅ No weight duplication in memory
- ✅ TFLite file contains correct trained weights
- ✅ Adapter is just interface wrapper, not a copy

### 4.9 Advanced Design Considerations

**Functional Model Signature Handling**

Functional models require special signature wrapping due to their call semantics. The signature must be wrapped in a single-element list `[input_signature]` because Functional models' `call()` method expects one positional argument containing the complete nested structure, not multiple positional arguments.

```python
# Correct signature for Functional model with dict inputs
signature = [{
    "input_a": tf.TensorSpec(shape=(None, 10), dtype=tf.float32),
    "input_b": tf.TensorSpec(shape=(None, 20), dtype=tf.float32)
}]

# This ensures TFLite converter receives the correct call structure
```

**Registry-Based Configuration Selection**

The implementation uses a registry pattern for mapping model types to their configuration classes, providing O(1) lookup performance and clean extensibility. New model types can be added by simply registering a new config class without modifying core export logic.

```python
# Registry lookup example
config = ExporterRegistry.get_config(model)
# Returns appropriate config class based on model type

# Adding new model type:
ExporterRegistry.register_config(NewModelType, NewModelTypeConfig)
```

**Inheritance-Aware Model Type Detection**

For model hierarchies with inheritance (e.g., Seq2SeqLM extends CausalLM), the registry maintains registration order to ensure subclasses are matched before parent classes. This prevents incorrect configuration selection when a model inherits from a more general base class.

```python
# Registration order matters for inheritance
ExporterRegistry.register_config(Seq2SeqLM, Seq2SeqLMExporterConfig)  # Subclass first
ExporterRegistry.register_config(CausalLM, CausalLMExporterConfig)    # Parent class second

# isinstance() check returns first match, ensuring specificity
```

**Memory-Efficient Model Building**

Models must be built before export to ensure variables exist, but using `model.build(input_shape)` instead of `model(dummy_data)` avoids unnecessary memory allocation for actual tensor data.

```python
# Memory-efficient approach
input_shape = {
    "token_ids": (None, 128),
    "padding_mask": (None, 128)
}
model.build(input_shape)  # Creates variables without allocating tensor data
```

### 4.10 Error Handling Design

**Error Categories:**

| Error Type | Example | Handled By | User Action |
|-----------|---------|------------|-------------|
| **Model not built** | Subclassed model never called | Keras Core | Call model or provide signature |
| **Unsupported type** | AudioClassifier export | Keras-Hub Registry | Check supported models |
| **Wrong extension** | `export("model.pb")` | Both layers | Use `.tflite` extension |
| **Missing preprocessor** | Vision model without image_size | Keras-Hub Config | Add preprocessor or set param |
| **Backend mismatch** | JAX model → TFLite | Keras Core | Convert to TF backend first |

**Error Flow Example:**

```
User: model.export("model.pb")
  │
  ├─ Keras-Hub checks: format="litert" → filename must end with .tflite
  │  └─ AssertionError: "filepath must end with '.tflite'" ❌
  │
  └─ (If passed) Keras Core validates model built
     └─ ValueError: "Model not built" ❌
```

### 4.11 Complete Export Pipeline

```
┌───────────────────────────────────────────────────────────┐
│ STEP 1: User Invokes Export                               │
│  model.export("model.tflite", format="litert",            │
│               max_sequence_length=128)                    │
└─────────────┬─────────────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────────────────────────────┐
│ STEP 2: Keras-Hub Registry Lookup                          │
│  ├─ Detect model type: isinstance(model, CausalLM)         │
│  ├─ Get config: CausalLMExporterConfig                     │
│  └─ Get exporter: LiteRTExporter                           │
└─────────────┬──────────────────────────────────────────────┘
              │
              ▼
┌───────────────────────────────────────────────────────────┐
│ STEP 3: Build Model & Get Signature                       │
│  ├─ Infer sequence_length from preprocessor (if None)     │
│  │  └─ Or use max_sequence_length=128 param               │
│  ├─ Build model: model.build({                            │
│  │    "token_ids": (None, 128),                           │
│  │    "padding_mask": (None, 128)                         │
│  │  })                                                    │
│  └─ Get signature: config.get_input_signature(128)        │
└─────────────┬─────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────┐
│ STEP 4: Create Adapter Wrapper                           │
│  adapter = TextModelAdapter(                             │
│    keras_hub_model=model,                                │
│    expected_inputs=["token_ids", "padding_mask"],        │
│    input_signature={...}                                 │
│  )                                                       │
│  ├─ adapter.inputs = [Input("token_ids"),                │
│  │                     Input("padding_mask")]            │
│  └─ adapter.variables = model.variables  (shared!)       │
└─────────────┬────────────────────────────────────────────┘
              │
              ▼
┌───────────────────────────────────────────────────────────┐
│ STEP 5: Delegate to Keras Core                            │
│  keras_exporter = KerasLitertExporter(                    │
│    model=adapter,                                         │
│    input_signature=[InputSpec, InputSpec]  (list!)        │
│  )                                                        │
│  keras_exporter.export("model.tflite")                    │
└─────────────┬─────────────────────────────────────────────┘
              │
              ▼
┌───────────────────────────────────────────────────────────┐
│ STEP 6: TFLite Conversion (Keras Core)                    │
│  ├─ Create tf.function(adapter.call)                      │
│  ├─ Build concrete function with signature                │
│  ├─ Convert to SavedModel (temp)                          │
│  ├─ Run TFLiteConverter                                   │
│  └─ Write model.tflite                                    │
└─────────────┬─────────────────────────────────────────────┘
              │
              ▼
          .tflite file
          ├─ Contains: adapter weights (= original model)
          ├─ Signature: [token_ids, padding_mask] (list)
          └─ Ready for inference on device
```

---

## 5. Usage Examples

### 5.1 Basic Export API

**Unified Interface:**

```python
model.export(filepath, format="litert", **options)
```

**Common Options:**

| Option | Type | Purpose | Example |
|--------|------|---------|---------|
| `filepath` | str | Output path (must end in `.tflite`) | `"model.tflite"` |
| `format` | str | Export format | `"litert"` |
| `input_signature` | list | Override signature | `[InputSpec(...)]` |
| `verbose` | bool | Show progress | `True` |
| `litert_kwargs` | dict | TFLite converter options | `{"optimizations": [tf.lite.Optimize.DEFAULT]}` |

**Available `litert_kwargs` Options:**

| Key | Type | Purpose | Example |
|-----|------|---------|---------|
| `optimizations` | list | Quantization/optimization strategy | `[tf.lite.Optimize.DEFAULT]` |
| `representative_dataset` | callable | Dataset for full int quantization | `representative_dataset_fn` |
| `experimental_new_quantizer` | bool | Use experimental quantizer | `True` |
| `aot_compile_targets` | list | Hardware-specific compilation | `["arm64", "x86_64"]` |
| `target_spec` | dict | Advanced TFLite converter settings | `{"supported_ops": [...]}` |

**Note:** `litert_kwargs` are passed directly to `tf.lite.TFLiteConverter`. See [TFLite Converter documentation](https://www.tensorflow.org/lite/api_docs/python/tf/lite/TFLiteConverter) for all available options.

### 5.2 Model Type Examples

**Keras Core (Simple Models):**

```python
# Functional
inputs = keras.Input(shape=(224, 224, 3))
outputs = keras.layers.Dense(10)(...)
model = keras.Model(inputs, outputs)
model.export("model.tflite", format="litert")

# Sequential
model = keras.Sequential([Dense(64), Dense(10)])
model.export("model.tflite", format="litert")

# Subclassed (must build first)
model = MyCustomModel()
model(dummy_input)  # Build by calling
model.export("model.tflite", format="litert")
```

**Keras-Hub (Complex Models):**

```python
# Text models (specify sequence_length)
model = keras_hub.models.GemmaCausalLM.from_preset("gemma_2b")
model.export("gemma.tflite", max_sequence_length=128)

# Vision models (auto-infer from preprocessor)
model = keras_hub.models.ResNetImageClassifier.from_preset("resnet50")
model.export("resnet.tflite")  # image_size inferred
```

### 5.3 Common Patterns

**Pattern 1: Export with Explicit Parameters**

```python
# When you want specific input shape
model.export(
    "model.tflite",
    format="litert",
    max_sequence_length=256  # Override default
)
```

**Pattern 2: Quantized Export (Recommended for Production)**

```python
import tensorflow as tf

# Simple dynamic range quantization (~75% size reduction)
model.export(
    "model_quantized.tflite",
    format="litert",
    litert_kwargs={
        "optimizations": [tf.lite.Optimize.DEFAULT]
    }
)

# Full integer quantization (best performance)
def representative_dataset():
    for i in range(100):
        # Use real training data samples for best results
        yield [training_data[i]]

model.export(
    "model_int8.tflite",
    format="litert",
    litert_kwargs={
        "optimizations": [tf.lite.Optimize.DEFAULT],
        "representative_dataset": representative_dataset
    }
)
```

**Pattern 3: Hardware-Optimized Export**

```python
# AOT compilation for specific targets (reduces inference latency)
model.export(
    "model.tflite",
    format="litert",
    litert_kwargs={
        "aot_compile_targets": ["arm64", "x86_64"]  # Common targets
    }
)

# Valid targets: "arm64", "x86_64", "arm", "riscv64"
# Note: AOT compilation increases file size but improves runtime performance
```

**Pattern 4: Debug Mode**

```python
# See detailed conversion logs
model.export("model.tflite", format="litert", verbose=True)
```

**Pattern 5: Advanced TFLite Converter Options**

```python
import tensorflow as tf

# Combine multiple converter options
model.export(
    "model_advanced.tflite",
    format="litert",
    litert_kwargs={
        "optimizations": [
            tf.lite.Optimize.DEFAULT,
            tf.lite.Optimize.EXPERIMENTAL_SPARSITY
        ],
        "representative_dataset": representative_dataset,
        "experimental_new_quantizer": True,
        "target_spec": {
            "supported_ops": [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
        }
    }
)
```

**Pattern 6: Override Signature (Advanced)**

```python
# Use when: (1) Subclassed model not built, (2) Custom input shapes needed
custom_sig = [keras.layers.InputSpec(shape=(None, 128), dtype="int32")]
model.export("model.tflite", input_signature=custom_sig)
```

### 5.4 Quantization and Optimization

Quantization reduces model size (~75% reduction) and improves inference speed by converting weights from float32 to int8. Use the `litert_kwargs` parameter to enable optimizations.

#### Basic Quantization

```python
import tensorflow as tf

# Dynamic range quantization (simplest - no dataset needed)
model.export(
    "model_quantized.tflite",
    format="litert",
    litert_kwargs={
        "optimizations": [tf.lite.Optimize.DEFAULT]
    }
)

# Full integer quantization (best performance - requires dataset)
def representative_dataset():
    for i in range(100):
        yield [training_data[i].astype(np.float32)]

model.export(
    "model_int8.tflite",
    format="litert",
    litert_kwargs={
        "optimizations": [tf.lite.Optimize.DEFAULT],
        "representative_dataset": representative_dataset
    }
)
```

#### Available Optimization Flags

| Flag | Purpose | Requires Dataset? |
|------|---------|-------------------|
| `tf.lite.Optimize.DEFAULT` | Quantization (weights → int8) | No |
| `tf.lite.Optimize.DEFAULT` + dataset | Full int8 quantization | Yes |
| `tf.lite.Optimize.OPTIMIZE_FOR_SIZE` | Size optimization | No |
| `tf.lite.Optimize.OPTIMIZE_FOR_LATENCY` | Latency optimization | No |
| `tf.lite.Optimize.EXPERIMENTAL_SPARSITY` | Sparsity optimization | No |

**Combining optimizations:**
```python
model.export(
    "model.tflite",
    format="litert",
    litert_kwargs={
        "optimizations": [
            tf.lite.Optimize.DEFAULT,
            tf.lite.Optimize.EXPERIMENTAL_SPARSITY
        ]
    }
)
```

**See also:** [TFLite Quantization Guide](https://www.tensorflow.org/lite/performance/post_training_quantization) for advanced techniques including quantization-aware training.

### 5.5 Troubleshooting

**Common Errors and Solutions:**

| Error Message | Cause | Solution |
|--------------|-------|----------|
| `ValueError: Model must be built` | Subclassed model never called | Call `model(dummy_input)` or provide `input_signature` |
| `AssertionError: filepath must end with '.tflite'` | Wrong file extension | Use `.tflite` extension: `model.export("model.tflite")` |
| `ValueError: X model type is not supported for export` | Unsupported Keras-Hub model | Check supported models in Section 1.3 |
| `RuntimeError: Some ops are not supported by TFLite` | TF ops not in TFLite | Check TFLite op compatibility or use TF Select ops |
| `ValueError: Cannot infer sequence_length` | Text model without preprocessor | Specify `max_sequence_length=N` in export call |
| `ValueError: Cannot infer image_size` | Vision model without preprocessor | Add preprocessor or specify image size |

**Debug Checklist:**

1. ✅ Is model built? (Check `model.built == True`)
2. ✅ Does filepath end with `.tflite`?
3. ✅ For Keras-Hub models, is preprocessor attached or parameters specified?
4. ✅ Are all layers/ops supported by TFLite? (Run with `verbose=True`)
5. ✅ For large models (>2GB), do you have sufficient memory?

**Performance Considerations:**

- **Export Time:** Proportional to model size. Typical models (100M-1B parameters): ~5-30 seconds. Large models (5B+ parameters): several minutes.
- **File Size:** `.tflite` file ≈ model parameter count × 4 bytes (float32). Use quantization to reduce.
- **Memory:** Export has high memory requirements, especially for large models. This is a known limitation of TFLite converter:
  - **Small models** (<1GB): ~3-5x model size in RAM
  - **Large models** (5GB+): Can require 10x or more peak memory (e.g., 5GB model may need 45GB+ RAM)
  - This varies significantly by architecture and is a known TFLite/LiteRT limitation without current fix
  - For large models: Use high-memory machines (cloud VMs) or apply quantization during training to reduce model size first

### 5.6 Decision Tree: When to Use What

```
Do you have a Keras-Hub model?
  ├─ YES → Use task.export()
  │   │
  │   ├─ Text model? → Specify max_sequence_length
  │   └─ Vision model? → Preprocessor handles image_size
  │
  └─ NO → Keras Core model
      │
      ├─ Functional/Sequential? → Direct export
      └─ Subclassed? → Build first, then export
```

---

## 6. Alternatives Considered

*This section documents alternative approaches considered during design and why they were rejected.*

### 6.1 Adapter Pattern Rationale

**Problem:** Keras-Hub models use dictionary inputs, but TFLite expects list inputs.

**Chosen Solution:** Adapter Pattern (as implemented)

**Alternatives Considered:**
- **Direct model modification**: Modify model's `call()` signature to accept list inputs
  - ❌ Rejected: Would break existing user code
- **Fork TFLite Converter**: Modify TFLite to support dict inputs
  - ❌ Rejected: Too invasive, maintenance burden

---

## 7. Testing Strategy

### 7.1 Test Pyramid

```
                  ┌──────────────┐
                  │  Integration │  ← End-to-end: model.export() → .tflite
                  │     Tests    │     Keras-Hub + Keras Core
                  └──────┬───────┘
                        ╱ ╲
                       ╱   ╲
                      ╱     ╲
                     ╱       ╲
           ┌────────┴─────────┴────────┐
           │   Component Tests          │  ← Registry, Adapters, Configs
           │  (Keras-Hub specific)      │     Input signature generation
           └────────────┬───────────────┘
                       ╱ ╲
                      ╱   ╲
                     ╱     ╲
                    ╱       ╲
          ┌────────┴─────────┴─────────┐
          │     Unit Tests              │  ← Signature inference, conversion
          │   (Keras Core)              │     Direct vs wrapper strategies
          └─────────────────────────────┘
```

### 7.2 Test Coverage Matrix

| Layer | Component | Test Type | Example |
|-------|-----------|-----------|---------|
| **Keras Core** | Functional model | Unit | Single input → .tflite |
| **Keras Core** | Functional model | Unit | Dict inputs → .tflite |
| **Keras Core** | Sequential model | Unit | Standard layers → .tflite |
| **Keras Core** | Subclassed model | Unit | Custom call() → .tflite |
| **Keras Core** | Signature inference | Unit | Auto-detect from `model.inputs` |
| **Keras Core** | Conversion strategy | Unit | Direct vs Wrapper selection |
| **Keras Core** | Quantization | Unit | DEFAULT optimization |
| **Keras Core** | Quantization | Unit | OPTIMIZE_FOR_SIZE |
| **Keras Core** | Quantization | Unit | OPTIMIZE_FOR_LATENCY |
| **Keras Core** | Quantization | Unit | EXPERIMENTAL_SPARSITY |
| **Keras Core** | Quantization | Unit | Multiple optimizations combined |
| **Keras Core** | Quantization | Unit | Representative dataset |
| **Keras Core** | Quantization | Unit | File size verification (~75% reduction) |
| **Keras-Hub** | CausalLM | Integration | Gemma → .tflite with text inputs |
| **Keras-Hub** | TextClassifier | Integration | BERT → .tflite with classification |
| **Keras-Hub** | Seq2SeqLM | Integration | T5 → .tflite with 4 inputs |
| **Keras-Hub** | ImageClassifier | Integration | ResNet → .tflite with images |
| **Keras-Hub** | Registry | Component | Model type → Config mapping |
| **Keras-Hub** | Adapter | Component | Dict → List conversion |
| **Keras-Hub** | Config | Component | Input signature generation |
| **Cross-layer** | litert_kwargs | Integration | Custom converter options |

### 7.3 Key Test Scenarios

**Scenario 1: Sequence Length Inference**

```python
# Test: Auto-infer from preprocessor
model = keras_hub.models.GemmaCausalLM.from_preset(
    "gemma_1.1_instruct_2b_en"
    # preprocessor has sequence_length=512
)
model.export("model.tflite")  # Should use 512, not default 128

# Verify:
interpreter = tf.lite.Interpreter("model.tflite")
input_shape = interpreter.get_input_details()[0]['shape']
assert input_shape[1] == 512  ← Inferred correctly ✅
```

**Scenario 2: Adapter Variable Sharing**

```python
# Test: Adapter shares variables (no copy)
model = create_causal_lm()
adapter = TextModelAdapter(model, ...)

# Modify adapter variables
adapter.variables[0].assign(new_value)

# Check: Original model sees same change
assert np.array_equal(model.variables[0], adapter.variables[0])  ✅
```

**Scenario 3: Registry Subclass Ordering**

```python
# Test: Seq2SeqLM gets correct config (not CausalLM)
model = keras_hub.models.T5(...)  # T5 is Seq2SeqLM
config = ExporterRegistry.get_config(model)

assert isinstance(config, Seq2SeqLMExporterConfig)  ✅
assert config.EXPECTED_INPUTS == [
    "encoder_token_ids",
    "encoder_padding_mask",
    "decoder_token_ids",
    "decoder_padding_mask"
]
```

**Scenario 4: Quantization with litert_kwargs**

```python
import tensorflow as tf
import os

# Test: Dynamic range quantization reduces file size
model = create_conv_model()  # Large model for size comparison

# Export without quantization
model.export("model_float32.tflite")
size_float32 = os.path.getsize("model_float32.tflite")

# Export with quantization
model.export(
    "model_quantized.tflite",
    format="litert",
    litert_kwargs={
        "optimizations": [tf.lite.Optimize.DEFAULT]
    }
)
size_quantized = os.path.getsize("model_quantized.tflite")

# Verify ~75% size reduction
reduction = size_quantized / size_float32
assert reduction < 0.3  # Should be ~25% of original size ✅

# Verify quantized model still runs
interpreter = tf.lite.Interpreter("model_quantized.tflite")
interpreter.allocate_tensors()
# Check for int8 tensors
tensor_details = interpreter.get_tensor_details()
int8_count = sum(1 for t in tensor_details if t['dtype'] == np.int8)
assert int8_count > 0  # Should have quantized tensors ✅
```

**Scenario 5: Error Handling**

```python
# Test: Unsupported model type
model = AudioClassifier(...)  # Not in registry
with pytest.raises(ValueError, match="not supported"):
    model.export("model.tflite")

# Test: Wrong file extension
model = keras_hub.models.GemmaCausalLM(...)
with pytest.raises(AssertionError, match="must end with '.tflite'"):
    model.export("model.pb", format="litert")
```

---

---

## 8. Known Limitations

### 8.1 Memory Requirements During Conversion

**Issue:** TFLite conversion requires **10x or more RAM** than model size.

**Example:** A 5GB model may need 45GB+ of RAM during conversion.

**Root Cause:** TensorFlow Lite Converter builds multiple intermediate graph representations in memory.

**Workarounds:**
- Use a machine with sufficient RAM (cloud instance for large models)
- The generated `.tflite` file will be normal size (no bloat)
- Consider model quantization to reduce model size before export

**Status:** This is a TFLite Converter limitation, not fixable in Keras export code.

### 8.2 Hardcoded Input Name Assumptions

**Issue:** Keras-Hub model configs assume standard input names:
- Text models: `["token_ids", "padding_mask"]`
- Image models: `["images"]`
- Seq2Seq models: `["encoder_token_ids", "encoder_padding_mask", "decoder_token_ids", "decoder_padding_mask"]`

**Impact:** Custom Keras-Hub models with non-standard input names will fail export.

**Workaround:** Subclass the config and override `EXPECTED_INPUTS`:
```python
from keras_hub.src.export.configs import CausalLMExporterConfig

class CustomConfig(CausalLMExporterConfig):
    EXPECTED_INPUTS = ["my_input_ids", "my_mask"]  # Your names
```

---

### Private API Dependency

**Issue:** Uses TensorFlow internal `_DictWrapper` class for layer unwrapping.

**Risk:** Could break if TensorFlow changes internal structure (unlikely).

**Impact:** Only affects Keras-Hub models, not Keras Core models.

---

## 9. FAQ (Frequently Asked Questions)

**Q: Can I export models trained with JAX or PyTorch backends?**  
A: Yes! Export works from any Keras 3.x backend. The exporter automatically converts backend tensors to TensorFlow format during export. However, if your model uses operations not supported by TensorFlow, you'll get a conversion error.

**Q: Does the adapter wrapper add runtime overhead on mobile devices?**  
A: No. The adapter only exists during export to convert interfaces. The final `.tflite` file contains your original model weights with no wrapper overhead.

**Q: Can I quantize models during export?**  
A: **Yes!** Quantization is fully supported through the `litert_kwargs` parameter. You can apply dynamic range quantization (~75% size reduction), full integer quantization, and various optimization strategies. See **[Section 5.4: Quantization and Optimization](#54-quantization-and-optimization)** for comprehensive examples and best practices.

**Q: What if my model uses custom layers or operations?**  
A: Custom Keras layers that use standard TensorFlow ops will work. If you have truly custom TFLite ops, you'll need to register them separately using TFLite's custom op mechanism (out of scope for this export API).

**Q: Can I export multiple models into one `.tflite` file?**  
A: No. Each `.tflite` file contains one model. For multi-model deployment, export separately and load multiple interpreters on the device.

**Q: How do I load the exported model on Android/iOS?**  
A: Use TensorFlow Lite's platform-specific APIs:
- **Android**: [TFLite Java/Kotlin API](https://www.tensorflow.org/lite/android)
- **iOS**: [TFLite Swift/Obj-C API](https://www.tensorflow.org/lite/ios)

**Q: My model is 5GB. Will export work?**  
A: Export has very high memory requirements for large models. Based on real-world data:

**Memory Requirements (Known Issue):**
- **Gemma3 1B / Llama3 1B models** (~5GB float32): Require **45GB+ peak RAM**
- This is a **known limitation** of TFLite/LiteRT converter with no current fix
- Memory usage scales unpredictably with model size and architecture
- Not a simple 3x multiplier - can be 10x or more for large models

**If you have insufficient RAM:**
- ✅ Use high-memory cloud VMs (e.g., AWS r6i.4xlarge with 128GB RAM)
- ✅ Apply quantization **during training** to reduce model size first
- ✅ Consider model pruning or distillation to create smaller variants
- ❌ No streaming/chunked export mode currently available

**Why so much memory?**
The TFLite converter creates multiple intermediate representations (SavedModel, concrete functions, TFLite graph) during conversion, all of which must fit in memory simultaneously. This is a known limitation of the current TFLite architecture.

**Q: Can I resume an interrupted export?**  
A: No. Export is atomic - if interrupted, you must restart. The process typically takes seconds to minutes, so interruptions are rare.

**Q: Why does my exported model have different accuracy than in Keras?**  
A: Common causes:
1. **Quantization**: If you applied post-training quantization
2. **Op differences**: Some TF ops behave slightly differently in TFLite
3. **Numerical precision**: TFLite may use different precision settings

**How to debug:**
```python
import numpy as np
import tensorflow as tf

# 1. Get test input
test_input = np.random.randn(1, 224, 224, 3).astype(np.float32)

# 2. Keras prediction
keras_output = model.predict(test_input)

# 3. TFLite prediction
interpreter = tf.lite.Interpreter("model.tflite")
interpreter.allocate_tensors()
interpreter.set_tensor(interpreter.get_input_details()[0]['index'], test_input)
interpreter.invoke()
tflite_output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

# 4. Compare
diff = np.abs(keras_output - tflite_output).max()
print(f"Max difference: {diff}")  # Should be < 1e-5 for float32
```

**Q: Is there a size limit for `.tflite` files?**  
A: No hard limit in the format itself, but practical limits exist:
- Mobile apps: Google Play has 150MB APK size limit (use download manager for large models)
- Embedded devices: Limited by device storage and RAM

**Q: Can I export Keras 2.x models?**  
A: This export API is for Keras 3.x only. For Keras 2.x models:
1. Load in Keras 2.x
2. Save as SavedModel
3. Use `tf.lite.TFLiteConverter.from_saved_model()`

Or migrate your model to Keras 3.x first.

---

## 10. References

### 10.1 Implementation PRs

- **Keras Core LiteRT Export:** [keras#21674](https://github.com/keras-team/keras/pull/21674)
- **Keras-Hub LiteRT Export:** [keras-hub#2405](https://github.com/keras-team/keras-hub/pull/2405)

### 10.2 Design Inspirations

- **TensorFlow Lite:** [Official Documentation](https://www.tensorflow.org/lite)
- **Hugging Face Optimum:** Registry pattern for model export [Docs](https://huggingface.co/docs/optimum)
- **Keras Model Serialization:** [Guide](https://keras.io/guides/serialization_and_saving/)

### 10.3 File Locations

**Source Code Structure (approximate line counts as of October 2025):**

```
keras/src/export/
  ├─ litert.py           ← Core exporter (~183 lines)
  ├─ export_utils.py     ← Signature utilities (~127 lines)
  └─ litert_test.py      ← Unit tests

keras_hub/src/export/
  ├─ base.py             ← Abstract base (~144 lines)
  ├─ configs.py          ← Model configs (~298 lines)
  ├─ litert.py           ← Adapter + exporter (~237 lines)
  ├─ registry.py         ← Registry init (~45 lines)
  └─ *_test.py           ← Test files (4 files)
```

**To explore the code:**
1. Start with `keras/src/export/litert.py` for core export logic
2. Then `keras_hub/src/export/litert.py` for Keras-Hub integration
3. Review `configs.py` to understand model-specific configurations

### 10.4 Key Design Insights Summary

**From Code Review:**

| Insight | Reviewer (Role) | Impact |
|---------|-----------------|--------|
| Functional models need list wrapping | fchollet (Keras Lead) | Ensures correct tf.function signature |
| Registry over isinstance chains | mattdangerw (Keras-Hub Lead) | Extensible, maintainable pattern |
| Subclass registration order matters | mattdangerw (Keras-Hub Lead) | Correct config for inherited models |
| Use model.build() not dummy data | SuryaPratapSingh37 (Contributor) | Memory efficient initialization |
| Adapter pattern for dict→list | mattdangerw (Keras-Hub Lead) | Preserves Keras Core exporter |
| TensorFlow backend only (for now) | divyashreepathihalli (Keras Team) | TFLite is TF-specific |

---

## Appendix: Architectural Decisions

This appendix documents alternative approaches considered during design and why they were rejected, providing context for the chosen architecture.

### A.1 Adapter Pattern Rationale

**Problem:** Keras-Hub models use dict inputs; TFLite expects lists.

**Why Adapter?**
- ✅ Preserves Keras Core exporter (no duplication)
- ✅ Clean separation of concerns
- ✅ Extensible to new model types
- ❌ Alternative (modify TFLite converter): Too invasive - would require forking TensorFlow Lite

**Alternative Considered:** Modify model's `call()` signature directly
- Rejected: Would break existing model code and user training scripts

### A.2 Registry Pattern Rationale

**Problem:** Map model types → configurations.

**Why Registry?**
- ✅ O(1) lookup vs O(n) isinstance chains
- ✅ Easy to add new model types (just register)
- ✅ Inspired by production systems (HuggingFace Optimum)
- ❌ Alternative (factory methods): Scattered logic across codebase

**Alternative Considered:** Single giant if-elif chain
- Rejected: O(n) performance, hard to maintain, doesn't scale

### A.3 Build Strategy Rationale

**Problem:** Ensure model variables exist before export.

**Why model.build(shapes)?**
- ✅ Memory efficient (no tensor data allocation)
- ✅ Works for all model types
- ✅ Same result as calling with data
- ❌ Alternative (dummy data): Memory intensive - 5GB model needs 5GB dummy data

**Alternative Considered:** Require user to always build manually
- Rejected: Poor UX - most models already built, automatic is better

### A.4 Signature Wrapping Rationale

**Problem:** TFLite expects specific tf.function signature.

**Why single-element list for Functional models?**
- ✅ Matches Functional model's call signature (single positional arg)
- ✅ Preserves nested input structure
- ✅ Works with TensorFlow's SavedModel conversion
- ❌ Without wrapping: Signature mismatch errors

---

**Document Metadata:**
- **Version:** 2.0
- **Date:** Based on PR review as of merge
- **Contributors:** Keras Team (@fchollet, @divyashreepathihalli), Keras-Hub Team (@mattdangerw, @SuryaPratapSingh37)
- **License:** Apache 2.0
