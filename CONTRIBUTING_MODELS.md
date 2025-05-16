# Model Contribution Guide

KerasHub has a plethora of pre-trained large language models
ranging from BERT to OPT. We are always looking for more models and are always
open to contributions!

In this guide, we will walk you through the steps one needs to take in order to
contribute a new pre-trained model to KerasHub. For illustration purposes, let's
assume that you want to contribute the DistilBERT model. Before we dive in, we encourage you to go through
[our getting started guide](https://keras.io/guides/keras_nlp/getting_started/)
for an introduction to the library, and our
[contribution guide](https://github.com/keras-team/keras-hub/blob/master/CONTRIBUTING.md).

## Checklist

This to-do list is a brief outline of how a model can be contributed.
Keep this checklist handy!

### Step 1: Open an issue/find an issue

- [ ] Open an issue or find an issue to contribute a backbone model.

### Step 2: PR #1 - Model folder
- [ ] Create your model folder XX in https://github.com/keras-team/keras-hub/tree/master/keras_hub/src/models

### Step 3: PR #1 - Add XXBackbone

- [ ] An `xx/xx_backbone.py` file which has the model graph \[[Example](https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/distil_bert/distil_bert_backbone.py)\].
- [ ] An `xx/xx_backbone_test.py` file which has unit tests for the backbone \[[Example](https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/distil_bert/distil_bert_backbone_test.py)\].
- [ ] A Colab notebook link in the PR description which matches the outputs of the implemented backbone model with the original source \[[Example](https://colab.research.google.com/drive/1SeZWJorKWmwWJax8ORSdxKrxE25BfhHa?usp=sharing)\].

### Step 4: PR #2 -  Data Converter - Add XXTokenizer or XXImageConverter or XXAudioConverter, etc


- [ ] If you are contributing a language model add a `xx/xx_tokenizer.py` file which has the tokenizer for the model \[[Example](https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/distil_bert/distil_bert_tokenizer.py)\].
- [ ] An `xx/xx_tokenizer_test.py` file which has unit tests for the model tokenizer \[[Example](https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/distil_bert/distil_bert_tokenizer_test.py)\].
- [ ] A Colab notebook link in the PR description, demonstrating that the output of the tokenizer matches the original tokenizer \[[Example](https://colab.research.google.com/drive/1MH_rpuFB1Nz_NkKIAvVtVae2HFLjXZDA?usp=sharing)].
- [ ] If you are contributing an image model add a `xx/xx_image_converter.py` file which has the image transformations for the model \[[Example](https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/clip/clip_image_converter.py)\].
- [ ] If you are contributing an image model add a `xx/xx_audio_converter.py` file which has the audio transformations for the model \[[Example](https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/moonshine/moonshine_audio_converter.py)\].


### Step 5: PR #3 - Add XX Tasks and Preprocessors

This PR is optional.

- [ ] An `xx/xx_<task>.py` file for adding a task model like classifier, masked LM, etc. \[[Example](https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/distil_bert/distil_bert_classifier.py)\]
- [ ] An `xx/xx_<task>_preprocessor.py` file which has the preprocessor and can be used to get inputs suitable for the task model \[[Example](https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/distil_bert/distil_bert_preprocessor.py)\].
- [ ] `xx/xx_<task>_test.py` file and `xx/xx_<task>_preprocessor_test.py` files which have unit tests for the above two modules \[[Example 1](https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/distil_bert/distil_bert_classifier_test.py) and [Example 2](https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/distil_bert/distil_bert_preprocessor_test.py)\].
- [ ] A Colab notebook link in the PR description, demonstrating that the output of the preprocessor matches the output of the original preprocessor \[[Example](https://colab.research.google.com/drive/1GFFC7Y1I_2PtYlWDToqKvzYhHWv1b3nC?usp=sharing)].
- [ ] Add a Colab notebook to demonstate an end to end demo of the task model, show that teh outputs are matching the original implementation and also add a demo to show finetuning of the model.

### Step 4: PR #4 and beyond - Add XX Presets,  Weights, and End-to-End Validation
- [ ] An `xx/xx_presets.py` file with links to weights uploaded to Kaggle Keras page[[Example](https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/distil_bert/distil_bert_presets.py)\].
- [ ] You can test out the model presets and show the demo by staging the model presets to KerasHub org page on [Kaggle](https://www.kaggle.com/organizations/kerashub). Here is the invite [link](https://kaggle.com/organizations/kerashub/invite/c4b8baa532b8436e8df8f1ed641b9cb5) to join the org page.
- [ ] A `tools/checkpoint_conversion/convert_xx_checkpoints.py` which is reusable script for converting checkpoints \[[Example](https://github.com/keras-team/keras-hub/blob/master/tools/checkpoint_conversion/convert_distilbert_checkpoints.py)\].
- [ ] A Colab notebook link in the PR description, showing an end-to-end task such as text classification, etc. The task model can be built using the backbone model, with the task head on top \[[Example](https://gist.github.com/mattdangerw/bf0ca07fb66b6738150c8b56ee5bab4e)\]. Show that the numerics and outputs are matching


## Detailed Instructions

This section discusses, in details, every necessary step.

### Step 1: Open an issue/Find an open issue

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

Once you are done identifying all the required layers, you should implement the
model backbone class.

To keep the code simple and readable, we follow
[Keras' functional style model](https://keras.io/guides/functional_api/) wrapped
around by a class to implement our models.

A model is typically split into three/four sections. We would recommend you to
compare this side-by-side with the
[`keras_hub.layers.DistilBertBackbone` source code](https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/distil_bert/distil_bert_backbone.py)!

Implementation: Use Keras' functional API or subclass keras.Model. Refer to existing KerasHub backbones for structure.
Inputs: Define standard inputs (e.g., token_ids, padding_mask for text; pixel_values for vision; audio_features for audio).
Layers: Leverage standard keras.layers and relevant keras_hub_layers where possible. Implement custom layers if necessary, ensuring they are well-tested and documented.

<br/>

The standard layers provided in Keras and KerasHub are generally enough for
most of the usecases and it is recommended to do a thorough search
[here](https://keras.io/api/layers/) and [here](https://keras.io/api/keras_nlp/layers/).
However, sometimes, models have small tweaks/paradigm changes in their architecture.
This is when things might slightly get complicated.

If the model introduces a paradigm shift, such as using relative attention instead
of vanilla attention, the contributor will have to implement complete custom layers. A case
in point is `keras_hub.models.DebertaV3Backbone` where we had to [implement layers
from scratch](https://github.com/keras-team/keras-hub/tree/master/keras_hub/models/deberta_v3).

On the other hand, if the model has a small tweak, something simpler can be done.
For instance, in the Whisper model, the self-attention and cross-attention mechanism
is exactly the same as vanilla attention, with the exception that the key projection
layer does not have a bias term. In this case, we can inherit the custom layer
from one of the standard layers and make minor modifications. See [this PR](https://github.com/keras-team/keras-hub/pull/801/files#diff-8533ae3a7755c0dbe95ccbb71f85c677297f687bf3884fadefc64f1d0fdce51aR22) for
more details.

Since the first PR is only to add the model backbone class, you should omit the
`from_presets()` function; this will be added at a later stage when you open a PR
for adding presets.

Validation Colab: Create a Colab notebook that: Loads weights from the original model source. Manually loads these weights into an instance of your KerasHub backbone. Compares the output of your backbone with the original model's corresponding layer output on sample inputs, ensuring numerical closeness.
Unit Tests (your_model_backbone_test.py): Include tests for forward pass, save/load, and correct output shapes with various configurations.

#### Convert weights from the original source and check output!

Before you open a PR for adding the model backbone class, it is essential to check
whether the model has been implemented exactly as the source implementation. This
also helps in adding model "presets" at a later stage.

The preferred way of doing this is to add a Colab link in the PR description, which
1) converts the original preset weights to our format, and
2) checks whether the outputs of the original model and your implemented model are close enough.

It is okay if you demonstrate it for one preset at this stage; you can do the conversion
for the other presets when you officially add presets to the library at a later stage.

#### Add Unit Tests

It is essential to add units tests. These unit tests are basic and mostly check
whether the forward pass goes through successfully, whether the model can be saved
and loaded correctly, etc.

### Step 3: PR #2 - Data Converter - Add XXTokenizer or XXImageConverter or XXAudioConverter, etc

#### Tokenizer

The Data Converter transforms raw data of a specific modality into a numerical format suitable for the preprocessor and backbone.
Implementation:
Text: YourModelTokenizer - Converts raw text into sequences of token IDs. Inherit from a base tokenizer in KerasNLP (e.g., WordPieceTokenizer, SentencePieceTokenizer) or implement a custom one. Define special tokens (e.g., cls_token, pad_token) and handle vocabulary loading.
Image: YourModelImageConverter (or similar name like ImageProcessor) - Handles operations like resizing, rescaling, normalization, and potentially data augmentation strategy application. May utilize keras_cv.layers.
Audio: YourModelAudioConverter (or similar name like AudioFeatureExtractor) - Processes raw audio into features like spectrograms or MFCCs. May utilize Keras or other audio processing libraries.
Assets: Ensure your converter can load necessary assets (e.g., vocabulary files for tokenizers, mean/std deviation values for image normalization).
Validation Colab: Demonstrate that your data converter's output (e.g., token IDs, processed pixel tensors, audio feature tensors) matches the behavior of the original model's data conversion step.
Unit Tests (e.g., your_model_tokenizer_test.py): Test core functionality, asset loading, and output consistency.

#### Unit Tests

The last step here is to add unit tests.:  Test core functionality, asset loading, and output consistency.
### Step 4: PR #3 and Beyond: Add XXTasks and XXPreprocessors

This PR builds on the backbone and data converter to create a user-friendly Task Model.
#### Preprocessor (your_model_<task>_preprocessor.py)
The Preprocessor takes raw data (text, images, audio paths, etc.) and uses the appropriate Data Converter to transform it into the full format expected by the Backbone.
##### Implementation: 
- Create a class (e.g., YourModelCausalLMPreprocessor, YourModelImageClassifierPreprocessor).
- It will use the specific YourModel<DataConverterType> (e.g., YourModelTokenizer) internally.
- It handles tasks like adding special tokens, padding/truncation for sequences, creating attention masks, batching, and ensuring the output dictionary matches the backbone's expected input names.
##### Inputs: 
- Define how it accepts raw data (e.g., strings, file paths, raw tensors).
##### Outputs: 
- It should output a dictionary of tensors ready for the Backbone.
##### Validation Colab: 
- Show that your preprocessor, given raw input, produces the same tensor inputs (e.g., token_ids, padding_mask, pixel_values) as the original model's complete preprocessing pipeline.
##### Unit Tests (your_model_<task>_preprocessor_test.py): 
- Test with various inputs, ensuring correct output shapes and values.

#### Task Model (your_model_<task>.py)
The Task Model is the high-level entry point. It combines the Backbone, Preprocessor, and a task-specific head.
##### Implementation: 
- Create a class (e.g., YourModelCausalLM, YourModelImageClassifier).
- It should instantiate its Backbone and Preprocessor in its constructor.
- It will include a task-specific head (e.g., a dense layer for classification, a language modeling head, detection heads).
##### API: 
- It should offer simple methods like predict(), fit(), generate() (for generative models), detect() (for detection models).
##### Unit Tests (your_model_<task>_test.py): 
- Test basic usage: instantiation, forward pass with dummy data from the preprocessor, and model compilation.

### Step 5: PR #4 - Add Presets, Weights, and End-to-End Validation

Once the above 3 PRs are merged you can open a PR for adding presets. For every model, we have a separate file where we mention our preset configurations. This preset configuration has model-specific arguments such as number of layers, number of attention heads; preprocessor-specific arguments such as whether we want to lowercase the input text; checkpoint and vocabulary file URLs, etc. Please use this [invite link](https://kaggle.com/organizations/kerashub/invite/c4b8baa532b8436e8df8f1ed641b9cb5) and stage your model presets [here](https://www.kaggle.com/organizations/kerashub/models)

After wrapping up the preset configuration file, you need to
add the `from_preset` function to all three classes, i.e., `DistilBertBackbone`,
and `DistilBertTokenizer`. Here is an
[example](https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/models/distil_bert/distil_bert_backbone.py#L187-L189).

The testing for presets is divided into two: "large" and "extra large".
For "large" tests, we pick the smallest preset (in terms of number of parameters)
and verify whether the output is correct. For "extra large tests", we loop over
all the presets and just check whether the backbone and the tokenizer can
be called without any error.

Checkpoint Conversion Script (tools/checkpoint_conversion/convert_your_model_checkpoints.py)
- Provide a script that converts weights from their original format (e.g., PyTorch .bin, TensorFlow SavedModel) to the Keras H5 format expected by KerasHub.
- This script should be reusable and clearly documented.
- It's crucial for verifying weight conversion accuracy and for future updates.
End-to-End Validation Colab
- This is the most important validation step.
- Create a Colab notebook that demonstrates:
    - Loading your Task Model using YourModelTask.from_preset("your_model_preset_name").
    - Running an end-to-end task (e.g., text generation, image classification, object detection) on sample input.
    - Comparing the output (e.g., generated text, class probabilities, bounding boxes) with the output of the original model using its original pretrained weights and inference pipeline. Ensure numerical closeness.
Numerics Test: Add at least one unit test (often marked as "large" or "extra_large") that loads a small preset via from_preset(), runs inference on a fixed input, and asserts that the output matches known-good values (obtained from the original model). See existing tests for examples.

### Step 6: PR #5 and Beyond - Add More Tasks or Advanced Features (Optional)


Once the primary Task Model is merged, you can extend its utility:
Additional Task Models: Contribute other task models that use the same YourModelBackbone (e.g., YourModelTokenClassifier if you initially contributed YourModelCausalLM, or YourModelImageSegmentation if you contributed YourModelImageClassifier). Each new task will likely require its own YourModel<NewTask>Preprocessor and YourModel<NewTask> class.
Parameter-Efficient Fine-Tuning (PEFT): Add LoRA support (e.g., backbone.enable_lora()) if applicable. See KerasHub's fine-tuning documentation for guidance.
Quantization (QLoRA): If the model benefits, implement and document QLoRA support.
Model Parallelism: For very large models, provide configurations or guidance for model parallelism.

## Conclusion
Once all three PRs (and optionally, the fourth PR) have been merged, you have
successfully contributed a model to KerasHub. Congratulations! 🔥


