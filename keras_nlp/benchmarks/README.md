# KerasNLP Benchmarks

This directory houses a collection of scripts for benchmarking APIs and utility
functions which KerasNLP provides.

## Text Generation
For benchmarking text generation functions, the following command can be run
from the root of the repository:

```sh
python3 ./keras_nlp/benchmarks/text_generation.py
```

On running this script on Google Colab (with Tesla T4 GPU, and TensorFlow 2.10.0),
the following results were obtained:

| **Decoding Strategy** 	| **Graph Mode (sec)** 	| **Graph Mode with XLA (sec)** 	|
|:---------------------:	|:--------------------:	|:-----------------------------:	|
|     Greedy Search     	|        495.78        	|             293.77            	|
|      Beam Search      	|        564.23        	|             615.17            	|
|     Random Search     	|        446.55        	|             296.21            	|
|      Top-k Search     	|        458.68        	|             302.66            	|
|      Top-p Search     	|        468.63        	|             565.50             	|

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