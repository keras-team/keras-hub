# KerasNLP Benchmarks

This directory houses a collection of scripts for benchmarking APIs and utility
functions which KerasNLP provides.

## Text Generation
For benchmarking text generation functions, the following command can be run
from the root of the repository:

```sh
python3 ./keras_nlp/benchmarks/text_generation.py
```

On running this script on Google Colab, the following results were obtained:

```
*************************************

Running greedy_search in xla mode
500/500 [==============================] - 294s 562ms/step
Time taken:  293.77657198905945

Running greedy_search in graph mode
500/500 [==============================] - 496s 986ms/step
Time taken:  495.7888162136078

*************************************
Running random_search in xla mode
500/500 [==============================] - 296s 567ms/step
Time taken:  296.20745635032654

Running random_search in graph mode
500/500 [==============================] - 446s 888ms/step
Time taken:  446.5564649105072

*************************************
Running top_k_search in xla mode
500/500 [==============================] - 303s 575ms/step
Time taken:  302.6635549068451

Running top_k_search in graph mode
500/500 [==============================] - 459s 910ms/step
Time taken:  458.6870086193085

*************************************
Running top_p_search in xla mode
500/500 [==============================] - 565s 1s/step
Time taken:  565.503867149353

Running top_p_search in graph mode
500/500 [==============================] - 469s 932ms/step
Time taken:  468.6358585357666

*************************************
```

To change the configuration, say, for example, number of layers in the transformer
model used for inference, the user can modify the config dictionaries given at
the top of the script.