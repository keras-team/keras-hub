# GLUE Finetuning Script

This script is written to help you evaluate your model on GLUE benchmarking.
It provides the functionalities below:

- GLUE dataset loading & common preprocessing
- Finetuning flow 
- Generate GLUE submission files

To use the script, you need to change the code to load your pretrained model,
and run the command below:

```shell
python glue.py --task_name="mrpc" --batch_size=32 \
    --submission_file_path="glue_submissions/"
```

By default the script finetunes on KerasNLP BERT model 
`keras_nlp.models.Bert.from_preset("bert_tiny_uncased_en")`.

To make a real GLUE leaderboard submission, you need to call the finetuning on 
all tasks, and enter the submission directory then zip the submission files:

```shell
cd glue_submissions
zip -r submission.zip *.tsv
```

GLUE submission requires the `submission.zip` contains `.tsv` file for all tasks
, otherwise it will be a failed submission. An empty `.tsv` will also fail 
because it checks the content. If you only want to evaluate on certain tasks, 
you can download the sample submission, and put the `.tsv` files for tasks you 
don't run inside your submission file. For example if you don't want to 
run the `ax` task, then you can do:

```
curl -O https://gluebenchmark.com/assets/CBOW.zip
unzip CBOW.zip -d sample_submissions
cp sample_submissions/AX.tsv glue_submissions
```

## How to Use the Script

To use this script on your model, you need to do 3 things:

1. Implement your custom preprocessing in `preprocess_fn()`.
2. Load your pretrained model.
3. Make the finetune model with your model.

Code needs customization is wrapped between comment
`### Custom code block starts ###` and 
`### Custom code block ends ###`. See instructions on each step below.

### Custom Preprocessing

In all GLUE dataset, each record comes with features of one or two sentences, 
and one label. In the script, we load GLUE dataset in the format 
`(features, labels)`,  where `features` is a tuple of either 1 sentence or 2
sentences. Your need to write custom preprocessing logic to convert to data
to the required input of your model. For example, in the current script 
(finetuning for KerasNLP BERT), it is doing:

```python
bert_preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
    "bert_tiny_uncased_en"
)
def preprocess_fn(feature, label):
    return bert_preprocessor(feature), label
```
It uses the `BertPreprocessor` to convert input feature formats.

### Load Pretrained Model

As long as it is a TF model, you can use it with this script. 

### Make the Finetuning Model

There are two ways to make the funetuning model - use the default classifier or 
write your own classifier. 

#### Use Default Classifier

The script provides a default classifier so that you can plug your pretrained 
model in. You need to make a wrapper on your pretrained model to output a single 
representation per input record, then pass the wrapper to `GlueClassifier` to 
create the finetuning model. For example, KerasNLP BERT model outputs a 
dictionary with keys `sequence_output` and `pooled_output`, and we only need 
the `pooled_output`. The easiest way to achieve it is to use Keras 
functional API, as in the script:

```python
inputs = bert_model.inputs
outputs = bert_model(inputs)["pooled_output"]
model_wrapper = keras.Model(inputs=inputs, outputs=outputs)

finetuning_model = GlueClassifier(
    backbone=model_wrapper,
    num_classes=3 if FLAGS.task_name in ("mnli", "ax") else 2,
)
```

#### Use Your Own Classifier

You can also implement your own classifier instead of 
using the default `GlueClassifier`. Remember to set flag 
`use_default_classifier=False` to use your own classifier.

## Flags Table

| Flags Name                 	| Explanation                                     	| Default 	|
|----------------------------	|-------------------------------------------------	|---------	|
| task_name                  	| The name of the GLUE task to finetune on.       	| "mrpc"  	|
| batch_size                 	| Data batch size                                 	| 32      	|
| epochs                     	| Number of epochs to run finetuning.             	| 2       	|
| learning_rate              	| The optimizer's learning rate                   	| 5e-5    	|
| submission_directory       	| The file path to save the glue submission file. 	| None    	|
| use_default_classifier     	| If using the default classifier.                	| True    	|
| finetuning_model_save_path 	| The path to save the finetuning model.          	| None    	|
