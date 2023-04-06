# KerasNLP Benchmarks

This directory houses a collection of scripts for benchmarking APIs and utility
functions which KerasNLP provides.

## Text Generation
For benchmarking text generation functions, the following command can be run
from the root of the repository:

```sh
python3 ./keras_nlp/benchmarks/text_generation.py
```

On running this script on Google Colab (with 3090 GPU, and TensorFlow 2.11.0),
the following results were obtained:

| **Decoding Strategy** 	| **Graph Mode (sec)** 	| **Graph Mode with XLA (sec)** 	|
|:---------------------:	|:--------------------:	|:-----------------------------:	|
|     Greedy Search     	|        470.23        	|              61.79            	|
|      Beam Search      	|        530.13        	|             189.61            	|
|      Top-k Search     	|        374.05        	|              62.87            	|
|      Top-p Search     	|        401.97        	|             260.31             	|

To change the configuration, say, for example, number of layers in the transformer
model used for inference, the user can modify the config dictionaries given at
the top of the script.

## Sentiment Analysis

For benchmarking classification models, the following command can be run
from the root of the repository:

```sh
python3 keras_nlp/benchmarks/sentiment_analysis.py \
    --model="BertClassifier" \
    --preset="bert_small_en_uncased" \
    --learning_rate=5e-5 \
    --num_epochs=5 \
    --batch_size=32
    --mixed_precision_policy="mixed_float16"
```

flag `--model` specifies the model name, and `--preset` specifies the preset under testing. `--preset` could be None, 
while `--model` is required. Other flags are common training flags.

This script outputs:

- validation accuracy for each epoch.
- testing accuracy after training is done.
- total elapsed time (in seconds).