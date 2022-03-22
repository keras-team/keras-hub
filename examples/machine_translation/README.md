# English-Spanish machine translation with keras-nlp

This example will show how to train a Transformer-based machine translation 
model using APIs provided by Keras-NLP. This instruction shows how to train the 
model, and evaluate with customized English sentences.

## Installing dependencies

Pip dependencies for all keras-nlp examples are listed in `setup.py`. To install
both the keras-nlp library from source and all other dependencies required to
run the example, run the below command. You may want to install to a self
contained environment (e.g. a container or a virtualenv).

```shell
pip install -e ".[examples]"
```

## Train the machine translation model and save to disk

At the root directory of keras-nlp, run the following command:

```shell
python ./examples/machine_translation/train.py \
    --num_epochs=3 \
    --saved_model_path="saved_models/machine_translation"
```

If it finishes successfully, you should see your console print out the 
following information:
```
Successfully saved model to saved_models/machine_translation.
```

## Running machine translation on customized inputs

Once you have a model saved successfully, you can play around it via the 
inference.py script. To run inference on customized inputs, please run the 
following command:

```shell
python ./examples/machine_translation/train.py \
    --inputs="Have a nice day" \
    --saved_model_path=saved_models/machine_translation"
```

You can set the inputs value as any English sentence, or you can leave it unset, 
then the script will run against some predefined English sentences. 

