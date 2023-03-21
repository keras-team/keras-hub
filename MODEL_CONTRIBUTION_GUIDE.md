# Model Contribution Guide

KerasNLP has a plethora of cutting-edge "backbone" model ranging from BERT and
RoBERTa to FNet and DeBERTa. We are always looking for more models and are always
open to contributions!

In this guide, we will walk you through the steps one needs to take in order to contribute
a backbone model. For illustration purposes, let's assume that you want to
contribute the DistilBERT model. Before we dive in, we encourage you to go through
[our getting started guide](https://keras.io/guides/keras_nlp/getting_started/)
for an introduction to the library, and our
[contribution guide](https://github.com/keras-team/keras-nlp/blob/master/CONTRIBUTING.md).


## Checklist
This to-do list is a brief outline of how a backbone model can be contributed.
Keep this checklist handy!

### Step 1: Open an issue/find an issue.
- [ ] Open an issue or find an issue to contribute a backbone model.

### Step 2: PR #1 - Add XXBackbone
- [ ] A `xx/xx_backbone.py` file which has the model graph [[Example Link](https://github.com/keras-team/keras-nlp/blob/master/keras_nlp/models/distil_bert/distil_bert_backbone.py)].
- [ ] A `xx/xx_backbone_test.py` file which has unit tests for the backbone [[Example Link](https://github.com/keras-team/keras-nlp/blob/master/keras_nlp/models/distil_bert/distil_bert_backbone_test.py)].
- [ ] A Colab notebook link in the PR description which matches the outputs of the implemented backbone model with the original source [[Example Link](https://colab.research.google.com/drive/1SeZWJorKWmwWJax8ORSdxKrxE25BfhHa?usp=sharing)].

### Step 3: PR #2 - Add XXTokenizer and XXPreprocessor [The preprocessor is optional at this stage]
- [ ] A `xx/xx_tokenizer.py` file which has the tokenizer for the model [[Example Link](https://github.com/keras-team/keras-nlp/blob/master/keras_nlp/models/distil_bert/distil_bert_tokenizer.py)].
- [ ] [Optional] A `xx/xx_preprocessor.py` file which has the preprocessor and can be used to get inputs suitable for the model [[Example Link](https://github.com/keras-team/keras-nlp/blob/master/keras_nlp/models/distil_bert/distil_bert_preprocessor.py)].
- [ ] `xx/xx_tokenizer_test.py` and `xx/xx_preprocessor_test.py` which has unit tests for the above two modules [[Example Link 1](https://github.com/keras-team/keras-nlp/blob/master/keras_nlp/models/distil_bert/distil_bert_tokenizer_test.py) and [Example Link 2](https://github.com/keras-team/keras-nlp/blob/master/keras_nlp/models/distil_bert/distil_bert_preprocessor_test.py)].

### Step 4: PR #3 - Add XX Presets
- [ ] A `xx/xx_presets.py` file with links to weights uploaded to a personal GCP bucket/Google Drive [[Example Link](https://github.com/keras-team/keras-nlp/blob/master/keras_nlp/models/distil_bert/distil_bert_presets.py)].
- [ ] A `xx/xx_presets_test.py` file with runnable tests for each preset [[Example Link](https://github.com/keras-team/keras-nlp/blob/master/keras_nlp/models/distil_bert/distil_bert_presets_test.py)].
- [ ] A `tools/checkpoint_conversion/convert_xx_checkpoints.py` which is reusable script for converting checkpoints [[Example Link](https://github.com/keras-team/keras-nlp/blob/master/tools/checkpoint_conversion/convert_distilbert_checkpoints.py)].
- [ ] A Colab notebook showing an end-to-end task such as text classification, etc. The task model can be built using the backbone model, with the task head on top.

### Step 5: PR #4 and Beyond - Add XX Tasks and Preprocessors
This PR is optional and is for adding tasks like classifier, masked LM, etc. [[Example Link](https://github.com/keras-team/keras-nlp/blob/master/keras_nlp/models/distil_bert/distil_bert_classifier.py)]


## How is a Backbone model structured in KerasNLP?
To keep the code simple and readable, we follow
[Keras' functional style model](https://keras.io/guides/functional_api/) wrapped
around by a class to implement our models.

A model is typically split into three/four sections. We would recommend you to
compare this side-by-side with the
[`keras_nlp.layers.DistilBertBackbone` source code](https://github.com/keras-team/keras-nlp/blob/master/keras_nlp/models/distil_bert/distil_bert_backbone.py)!


### Inputs to the model
Generally, the standard inputs to any text model are:
    - `token_ids`: tokenised inputs (An integer representation of the text sequence).
    - `padding_mask`: Masks the padding tokens.


### Embedding layer(s)
Standard layers used: `keras.layers.Embedding`,
`keras_nlp.layers.PositionEmbedding`, `keras_nlp.layers.TokenAndPositionEmbedding`.


### Encoder layers
Standard layers used: `keras_nlp.layers.TransformerEncoder`, `keras_nlp.layers.FNetEncoder`.


### Decoder layers (possibly)
Standard layers used: `keras_nlp.layers.TransformerDecoder`.


### Other layers which might be used
`keras.layers.LayerNorm`, `keras.layers.Dropout`, `keras.layers.Conv1D`, etc.

<br/>

The standard layers provided in Keras and KerasNLP are generally enough for
most of the usecases and it is recommended to do a thorough search
[here](https://keras.io/api/layers/) and [here](https://keras.io/api/keras_nlp/layers/).
However, sometimes, models have small tweaks/paradigm changes in their architecture.
This is when things might slightly get complicated.

If the model introduces a paradigm shift, such as using relative attention instead
of vanilla attention, the contributor will have to implement complete custom layers. A case
in point is `keras_nlp.models.DebertaV3Backbone` where we had to [implement layers
from scratch](https://github.com/keras-team/keras-nlp/tree/master/keras_nlp/models/deberta_v3).

On the other hand, if the model has a small tweak, something simpler can be done.
For instance, in the Whisper model, the self-attention and cross-attention mechanism
is exactly the same as vanilla attention, with the exception that the key projection
layer does not have a bias term. In this case, we can inherit the custom layer
from one of the standard layers and make minor modifications. See [this PR](https://github.com/keras-team/keras-nlp/pull/801/files#diff-8533ae3a7755c0dbe95ccbb71f85c677297f687bf3884fadefc64f1d0fdce51aR22) for
more details.

Occasionally, if the model requires non-standard layers, we may ask you to first
contribute an example on keras.io, or demonstrate the working in a Colab notebook.
This will help us scope out the contribution, and possibly propose suggestions
on how to implement the model.


## Detailed Instructions
This section discusses, in details, every necessary step.

### Open an issue/Find an open issue
Before getting started with the code, it's important to check if there are any
[open issues](https://github.com/keras-team/keras-nlp/issues?q=is%3Aissue+is%3Aopen+label%3Amodel-contribution)
related to the model you wish to contribute. If there is an open issue, you can
claim it by commenting on the issue and letting us know that you're interested
in working on it. This helps us keep track of who is working on what and avoid
duplicated effort.

If there aren't any open issues, you can create one by clicking the "New Issue"
button on our repository page.

<br/>

Note that you need not have all the answers or complete knowledge of the inner
workings of the model at the time of opening the issue. But it is appreciated if
you can furnish as much detail as possible to enable us to help you with the
contribution! 🙂


### Step 2: PR #1 - Add XXBackbone
Once you are done identifying all the required layers, you should write down the
model backbone class. You might have to write custom layers as mentioned earlier.
Since the first PR is only to add the model backbone class, you should omit the
`from_presets()` function; this will be added at a later stage when you open a PR
for adding presets.


#### Convert weights from the original source and check output!
Before you open a PR for adding the model backbone class, it is essential to check
whether the model has been implemented exactly as the source implementation. This
also helps in adding model "presets" at a later stage.

The preferred way of doing this is to add a Colab link in the PR description, which
1) converts the original preset weights to our format, and
2) checks whether the outputs of the original model and your implemented model are close enough.

It is okay if you demonstrate it for one preset at this stage; you can do the conversion
for the other presets when you officially add presets to the library at a later stage.

It is essential to add units tests. These unit tests are basic and mostly check
whether the forward pass goes through successfully, whether the model can be saved
and loaded correctly, etc.

###  Step 3: PR #2 - Add XXTokenizer and XXPreprocessor [The preprocessor is optional at this stage]
Most text models nowadays use subword tokenizers such as WordPiece, SentencePiece
and BPE Tokenizer. Hence, it becomes easy to implement the tokenizer layer for
the model; the tokenizer class inherits from one of these base tokenizer classes.

For example, DistilBERT uses the WordPiece tokenizer. So, we can introduce a new
class, `DistilBertTokenizer`, which inherits from `keras_nlp.tokenizers.WordPieceTokenizer`.
All the underlying actual tokenization will be taken care of by the superclass.

The important thing here is adding "special tokens". Most models have
special tokens such as beginning-of-sequence token, end-of-sequence token, mask token,
pad token, etc. These have to be
[added as member attributes](https://github.com/keras-team/keras-nlp/blob/master/keras_nlp/models/distil_bert/distil_bert_tokenizer.py#L91-L105)
to the tokenizer class. These member attributes are then accessed by the preprocessor class.

For a full list of the tokenizers KerasNLP offers, please visit [this link](https://keras.io/api/keras_nlp/tokenizers/) and make use of the tokenizer your model uses!

The preprocessor class is responsible for making the inputs suitable for consumption
by the model - it packs multiple inputs together, i.e., given multiple input texts,
it will add appropriate special tokens, pad the inputs and return the dictionary
in the form expected by the model.

The preprocessor class might have a few intricacies depending on the model. For example,
the DeBERTaV3 tokenizer does not have the `[MASK]` in the provided sentencepiece
proto file, and we had to make some modifications [here](https://github.com/keras-team/keras-nlp/blob/master/keras_nlp/models/deberta_v3/deberta_v3_preprocessor.py). Secondly, we have
a separate preprocessor class for every task. This is because different tasks
might require different input formats. For instance, we have a [separate preprocessor](https://github.com/keras-team/keras-nlp/blob/master/keras_nlp/models/distil_bert/distil_bert_masked_lm_preprocessor.py)
for masked language modeling (MLM) for DistilBERT. So, we don't always recommend
adding the preprocessor right away; this can optionally be done in step 5.

The last step here is to add unit tests for both the tokenizer and the preprocessor.
A dummy vocabulary is created, and the output of both these layers is verified
including tokenization, detokenization, etc.

### Step 4: PR #3 - Add XX Presets
Once the two PR have been merged, you can open a PR for adding presets. For every
model, we have a separate file where we mention our preset configurations.
This preset configuration has model-specific arguments such as number of layers,
number of attention heads; preprocessor-specific arguments such as whether we want to
lowercase the input text; checkpoint and vocabulary file URLs, etc.
In the PR description, you can add Google Drive/personal GCP bucket links to the
checkpoint and the vocabulary files. These files will then be uploaded to GCP by us!

After wrapping up the preset configuration file, you need to
add the `from_preset` function to all three classes, i.e., `DistilBertBackbone`,
`DistilBertTokenizer` and `DistilBertPreprocessor`,. Here is an
[example](https://github.com/keras-team/keras-nlp/blob/master/keras_nlp/models/distil_bert/distil_bert_backbone.py#L187-L189).

The testing for presets is divided into two: "large" and "extra large".
For "large" tests, we pick the smallest preset (in terms of number of parameters)
and verify whether the output is correct. For "extra large tests", we loop over
all the presets and just check whether all three - backbone, tokenizer and
preprocessor - can be called without any error.

Additionally, a checkpoint conversion script should be added. This script
demonstrates that the outputs of our backbone model and outputs of the source
model match. This should be done for all presets.

### Step 5: PR #4 and Beyond: Add XX Tasks and Preprocessors
Once you are finished with Steps 1-4, you can add "task" models. Task models
are essentially models which have "task heads" on top of the backbone models.
For instance, for the text classification task, you can have a feedforward layer
on top of a backbone model like DistilBERT. Task models are very essential since
pretrained models are used extensively for downstream tasks like text classification,
token classification, text summarization, neural machine translation, etc.

For every task, we might need a preprocessor depending on the input
format. This has been described in Step 3.


## Conclusion
Once all three PRs (and optionally, the fourth PR) have been merged, you have
successfully contributed a model to KerasNLP. Congratulations! 🔥
