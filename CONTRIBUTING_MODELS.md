# Model Contribution Guide

KerasHub has a plethora of pre-trained large language models ranging from BERT 
to OPT. We are always looking for more models and are always open to 
contributions!

In this guide, we will walk you through the steps needed to contribute a new 
pre-trained model to KerasHub. For illustration purposes, let's assume that you 
want to contribute the DistilBERT model. Before we dive in, we encourage you to 
go through our [Getting Started Guide](https://keras.io/guides/keras_nlp/getting_started/) 
for an introduction to the library, and our [Contribution Guide](https://github.com/keras-team/keras-hub/blob/master/CONTRIBUTING.md).

---

## Checklist

This to-do list is a brief outline of how a model can be contributed.
Keep this checklist handy!

### Step 1: Open an Issue or Find an Issue

- [ ] Open an issue or find an issue to contribute a backbone model.

### Step 2: PR #1 - Model Folder

- [ ] Create your model folder `xx` in [`keras_hub/src/models`](https://github.com/keras-team/keras-hub/tree/master/keras_hub/src/models)

### Step 3: PR #1 - Add `XXBackbone`

- [ ] An `xx/xx_backbone.py` file which has the model graph  
      [Example](https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/distil_bert/distil_bert_backbone.py)

- [ ] An `xx/xx_backbone_test.py` file which has unit tests for the backbone  
      [Example](https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/distil_bert/distil_bert_backbone_test.py)

- [ ] A Colab notebook link in the PR description that matches the outputs of 
the implemented backbone model with the original source  
      [Example](https://colab.research.google.com/drive/1SeZWJorKWmwWJax8ORSdxKrxE25BfhHa?usp=sharing)

### Step 4: PR #2 - Data Converter - Add `XXTokenizer` or `XXImageConverter` or `XXAudioConverter`

- [ ] If contributing a language model, add an `xx/xx_tokenizer.py` file  
      [Example](https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/distil_bert/distil_bert_tokenizer.py)

- [ ] Add `xx/xx_tokenizer_test.py` file with unit tests  
      [Example](https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/distil_bert/distil_bert_tokenizer_test.py)

- [ ] A Colab notebook link in the PR description demonstrating that the 
      tokenizer output matches the original  
      [Example](https://colab.research.google.com/drive/1MH_rpuFB1Nz_NkKIAvVtVae2HFLjXZDA?usp=sharing)

- [ ] For image models: Add `xx/xx_image_converter.py` file with image 
      transformations  
      [Example](https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/clip/clip_image_converter.py)

- [ ] For audio models: Add `xx/xx_audio_converter.py` file with audio 
      transformations  
      [Example](https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/moonshine/moonshine_audio_converter.py)

### Step 5: PR #3 - Add `XX` Tasks and Preprocessors (Optional)

- [ ] Add `xx/xx_<task>.py` for adding task models (e.g., classifier, masked LM)  
      [Example](https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/distil_bert/distil_bert_classifier.py)

- [ ] Add `xx/xx_<task>_preprocessor.py` for preprocessing inputs to the task 
      model  
      [Example](https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/distil_bert/distil_bert_preprocessor.py)

- [ ] Add unit tests: `xx/xx_<task>_test.py` and `xx/xx_<task>_preprocessor_test.py`  
      [Example 1](https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/distil_bert/distil_bert_classifier_test.py),  
      [Example 2](https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/distil_bert/distil_bert_preprocessor_test.py)

- [ ] Colab notebook link in the PR description to validate that the 
      preprocessor output matches the original  
      [Example](https://colab.research.google.com/drive/1GFFC7Y1I_2PtYlWDToqKvzYhHWv1b3nC?usp=sharing)

- [ ] Add a Colab notebook demonstrating end-to-end usage of the task model, 
      showing matching outputs and a fine-tuning demo

### Step 6: PR #4 and Beyond - Add `XXPresets`, Weights, and End-to-End Validation

- [ ] Add `xx/xx_presets.py` with links to weights uploaded to Kaggle KerasHub  
      [Example](https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/distil_bert/distil_bert_presets.py)

- [ ] Stage the model presets on KerasHub’s [Kaggle org page](https://www.kaggle.com/organizations/kerashub) using this [invite link](https://kaggle.com/organizations/kerashub/invite/c4b8baa532b8436e8df8f1ed641b9cb5)

- [ ] Add `tools/checkpoint_conversion/convert_xx_checkpoints.py`, a reusable 
      script for converting checkpoints  
      [Example](https://github.com/keras-team/keras-hub/blob/master/tools/checkpoint_conversion/convert_distilbert_checkpoints.py)

- [ ]  A Colab notebook link in the PR description, showing an end-to-end task 
      such as text classification, etc. The task model can be built using the 
      backbone model, with the task head on top \[[Example](https://gist.github.com/mattdangerw/bf0ca07fb66b6738150c8b56ee5bab4e)\]. Show that the numerics 
      and outputs are matching

---

## Detailed Instructions

This section discusses, in details, every necessary step.
### Step 1: Open an Issue / Find an Open Issue

Before getting started with the code, it's important to check if there are any
[open issues](https://github.com/keras-team/keras-hub/issues?q=is%3Aissue+is%3Aopen+label%3Amodel-contribution)
related to the model you wish to contribute. If there is an open issue, you can
claim it by commenting on the issue and letting us know that you're interested
in working on it. This helps us keep track of who is working on what and avoid
duplicated effort.

If there aren't any open issues, you can create one by clicking the "New Issue"
button on our repository page.

Note that you need not have all the answers or complete knowledge of the inner
workings of the model at the time of opening the issue. But it is appreciated if
you can furnish as much detail as possible to enable us to help you with the
contribution! 🙂

### Step 2: PR #1 - Add XXBackbone

#### Add the backbone class

Once you've identified the required layers, implement the backbone using [Keras’ functional API](https://keras.io/guides/functional_api/) wrapped in a class.

Compare your code with [`DistilBertBackbone`](https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/distil_bert/distil_bert_backbone.py) for structure.

##### Implementation

- Use standard inputs (`token_ids`, `padding_mask`, `pixel_values`, `audio_features`)
- Use `keras.layers` and `keras_nlp.layers` when possible
- For architectural deviations, implement custom layers

Examples:
- Major changes: [`DebertaV3`](https://github.com/keras-team/keras-hub/tree/master/keras_hub/models/deberta_v3)
- Minor tweaks: [Whisper attention layer](https://github.com/keras-team/keras-hub/pull/801/files#diff-8533ae3a7755c0dbe95ccbb71f85c677297f687bf3884fadefc64f1d0fdce51aR22)

Do **not** include `from_presets()` in this PR.

##### Validation Colab

- Load original model weights
- Manually set weights in your KerasHub model
- Compare outputs on sample input for closeness

##### Unit Tests (`xx_backbone_test.py`)

- Check forward pass, output shapes
- Ensure model can be saved and loaded correctly

---

### Step 3: PR #2 – Data Converter

#### Tokenizer / ImageConverter / AudioConverter

The converter transforms raw input into numerical tensors suitable for 
preprocessing.

##### Implementation

- **Text**: `XXTokenizer`, subclassing from KerasHub tokenizers
- **Image**: `XXImageConverter`, subclassing from KerasHub ImageConverter - for 
  resizing, normalization, augmentation
- **Audio**: `XXAudioConverter`, subclassing from KerasHub AudioConverter for 
  extracting features like spectrograms

Include asset loading (e.g., vocab files, normalization stats).

##### Validation Colab

- Show that converted output (tokens, pixels, features) match original behavior

##### Unit Tests

- Validate core logic, asset loading, and consistency

---

### Step 4: PR #3 – Tasks and Preprocessors

#### Preprocessor (`xx_<task>_preprocessor.py`)

Transforms raw input into model-ready format.

##### Implementation

- Class: `XX<Task>Preprocessor`
- Internally uses the relevant `XX<Converter>`
- Handles padding, attention masks, batching, formatting

##### Inputs

- Accept strings, paths, or tensors

##### Outputs

- Dictionary of tensors compatible with the Backbone

##### Validation Colab

- Show that your preprocessor, given raw input, produces the same tensor inputs 
(e.g., token_ids, padding_mask, pixel_values) as the original model's complete 
preprocessing pipeline.

##### Unit Tests

- Test with various inputs, ensuring correct output shapes and values.

---

#### Task Model (`xx_<task>.py`)

Wraps the backbone and preprocessor with a task head.

##### Implementation

- Class: `XX<Task>`
- Instantiate backbone and preprocessor
- Add a task-specific head (e.g., classifier head, LM head)

##### API

- It should offer simple methods like predict(), fit(), generate() (for 
  generative models), detect() (for detection models)

##### Unit Tests

- Test basic usage: instantiation, forward pass with dummy data from the 
  preprocessor, and model compilation.

---

### Step 5: PR #4 – Presets and End-to-End Validation

After PRs 1–3 are merged, create:

#### Preset Configuration

- Add `xx_presets.py`
- Include model args, checkpoint URLs, vocabulary paths

Use the [Kaggle org page](https://www.kaggle.com/organizations/kerashub/models) 
to stage and test.

#### `from_preset()` Functions

Add this to:
- `XXBackbone`
- `XXTokenizer`

[Example](https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/distil_bert/distil_bert_backbone.py#L187-L189)

#### Preset Tests
The testing for presets is divided into two:
- "Large" tests: validate smallest preset numerics
- "Extra large" tests: loop over all presets, check for successful load/inference

#### Checkpoint Conversion Script (tools/checkpoint_conversion/convert_your_model_checkpoints.py)

- Provide a script that converts weights from their original format (e.g., 
PyTorch .bin, TensorFlow SavedModel) to the Keras H5 format expected by KerasHub.
- This script should be reusable and clearly documented.
- It's crucial for verifying weight conversion accuracy and for future updates.
End-to-End Validation Colab
- This is the most important validation step.

#### End-to-End Colab

- Load task model using `from_preset()`
- Run task (e.g., classification, generation)
- Compare output with original model

#### Numerics Test

- Add a test that loads a preset and compares outputs on fixed inputs

---

### Step 6: PR #5 and Beyond – Advanced Features (Optional)

Extend utility:

- New Task Models (e.g., TokenClassifier, ImageSegmentation)
- Parameter-Efficient Fine-Tuning (LoRA support)
- Quantization (QLoRA support)
- Model Parallelism (for large models)

---

## Conclusion

Once all four main PRs (and optionally the fifth) are merged, you've 
successfully contributed a model to KerasHub. Congratulations! 🔥
