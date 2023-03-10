# Model Contribution Guide

KerasNLP has a plethora of cutting-edge "backbone" model ranging from BERT and
RoBERTa to FNet and DeBERTa. We are always looking for more models and are always
open to contributions!

In this guide, we will walk you through the steps one needs to take in order to contribute
a backbone model. For illustration purposes, let's assume that you want to
contribute the DistilBERT model. Before we dive in, we encourage you to go through
[our guide](https://keras.io/guides/keras_nlp/getting_started/) on getting started
with KerasNLP to learn more about the high-level symbols we have.


## How is a Backbone model structured in KerasNLP?
To keep the code simple and readable, we follow
[Keras' functional model](https://keras.io/guides/functional_api/) style wrapped
around by a class to implement our models.

A model is typically split into three/four sections. We would recommend you to
compare this side-by-side with the
[`keras_nlp.layers.DistilBERTBackbone` source code](https://github.com/keras-team/keras-nlp/blob/v0.4.1/keras_nlp/models/distil_bert/distil_bert_backbone.py#L108-L114)!

- The inputs to the model
Generally, the standard inputs to any text model are:
  - `token_ids`: tokenised inputs (IDs in the form of text).
  - `padding_mask`: Masks the padding tokens.

- the embedding layer(s)
The standard layers used here are:
`keras.layers.Embedding`, `keras_nlp.layers.PositionEmbedding`, `keras_nlp.layers.TokenAndPositionEmbedding`.

- the encoder layers
The standard layers used here are:
`keras_nlp.layers.TransformerEncoder`, `keras_nlp.layers.FNetEncoder`.

- (possibly) the decoder layers
The standard layers used: `keras_nlp.layers.TransformerDecoder`.

- Other layers which might be used
`keras.layers.LayerNorm`, `keras.layers.Dropout`, `keras.layers.Conv1D`, etc.

The standard layers provided in Keras and KerasNLP are generally enough for
99% of the usecases and it is recommended to do a thorough search
[here](https://keras.io/api/layers/) and [here](https://keras.io/api/keras_nlp/layers/).
However, sometimes, models have small tweaks/paradigm changes in their architecture.
This is when things might slightly get complicated.

If the model introduces a paradigm shift, such as using relative attention instead
of vanilla attention, the contributor will have to implement complete custom layers. A case
in point is `keras_nlp.models.DebertaV3Backbone` where we had to [implement layers
from scratch](https://github.com/keras-team/keras-nlp/tree/v0.4.1/keras_nlp/models/distil_bert).

On the other hand, if the model has a small tweak, something simpler can be done.
For instance, in the Whisper model, the self-attention and cross-attention mechanism
is exactly the same as vanilla attention, with the exception that the key projection
layer does not have a bias term. In this case, we can inherit the custom layer
from one of the standard layers and make minor modifications. See [this PR](https://github.com/keras-team/keras-nlp/pull/801/files#diff-8533ae3a7755c0dbe95ccbb71f85c677297f687bf3884fadefc64f1d0fdce51aR22) for
more details.

## Open an issue/Claim an open issue
Before getting started with the code, it's important to check if there are any open issues
related to the model you wish to contribute: https://github.com/keras-team/keras-nlp/issues?q=is%3Aissue+is%3Aopen+label%3Amodel-contribution.
If there aren't any open issues, you can create one by clicking the "New Issue"
button on our repository page. If there is an open issue, you can claim it by
commenting on the issue and letting us know that you're interested in working on
it. This helps us keep track of who is working on what and avoid duplicated
effort.

In case you open a new issue, please follow this template:

```md
I would like to contribute the DistilBERT model. The DistilBERT model is a more
efficient, distilled version of BERT; it is 60% smaller than BERT, and 40% faster.
It is trained with BERT in a student-teacher fashion. At the same time, it does
not lose much of BERT's performance on the GLUE benchmark. Because DistilBERT is
efficient and performs well on common downstream tasks, it is widely used in the
industry and hence, will be a good addition to KerasNLP's model library!

- Will any non-standard layers be needed?
No, non-standard layers will not be needed. We can use the native
`keras.layers.TokenAndPositionEmbedding` and `keras_nlp.layers.TransformerEncoder`
layers since the architecture is the same as BERT's.

- Important Links:
Paper: [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108)
Official Repository: https://github.com/huggingface/transformers/tree/v4.26.1/src/transformers/models/distilbert
```

Note that you need not have all the answers or complete knowledge of the inner
workings of the model at the time of opening the issue. But it is appreciated if
you can furnish as much detail as possible to enable us to help you with the
contribution! :)

Occasionally, if the model requires, say a variant of the original self-attention,
or other non-standard layers, we may ask you to first contribute an example on
keras.io, or demonstrate the working in a Colab notebook. This will help us scope
out the contribution, and possibly propose suggestions on how to implement the
model.


## Write down the `DistilBertBackbone` class!



## Convert weights from the original source and check output!


## Add presets
<<fill-this>>


## Add tokenizer and preprocessor
<<fill-this>>
