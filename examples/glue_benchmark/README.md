# GLUE Finetuning Script

This script is written to help you evaluate your model on GLUE benchmarking.
It provides the functionalities below:

- Load and preprocess GLUE data.
- Finetuning your Keras text classification model. 
- Generate GLUE submission files.

To use the script, you need to change the code to load your pretrained model,
and run the command below:

```shell
python glue.py --task_name="mrpc" --batch_size=32 \
    --submission_directory="glue_submissions/"
```

By default the script finetunes on the tiniest BERT model we have available 
(this will be fast but not top performing).

To make a real GLUE leaderboard submission, you need to call the finetuning on 
all tasks, then enter the submission directory then zip the submission files:
```shell
for task in cola sst2 mrpc rte stsb qnli qqp; do
  python glue.py --task_name="$task" --submission_directory="glue_submissions/"
done

python glue.py --task_name="mnli_matched" \
    --submission_directory="glue_submissions/" \
    --save_finetuning_model="saved/mnli"

python glue.py --task_name="mnli_mismatched" \
    --submission_directory="glue_submissions/" \
    --load_finetuning_model="saved/mnli"

python glue.py --task_name="ax" \
    --submission_directory="glue_submissions/" \
    --load_finetuning_model="saved/mnli"

cd glue_submissions
zip -r submission.zip *.tsv
```

Please note that `mnli_matched`, `mnli_mismatched` and `ax` share the same 
training set, so we only train once on `mnli_matched` and use the saved model 
to evaluate on `mnli_mismatched` and `ax`.

GLUE submission requires the `submission.zip` contains `.tsv` file for all 
tasks, otherwise it will be a failed submission. An empty `.tsv` will also fail 
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

Code needing customization is wrapped between comment
`Custom code block starts` and 
`Custom code block ends`. See instructions on each step below.

### Custom Preprocessing

In all GLUE dataset, each record comes with features of one or two sentences, 
and one label. In the script, we load GLUE dataset in the format 
`(features, labels)`,  where `features` is a tuple of either 1 sentence or 2
sentences. Your need to write custom preprocessing logic to convert to data
to the required input of your model. For example, in the current script 
(finetuning for KerasNLP BERT), it is doing:

```python
bert_preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
    "bert_tiny_en_uncased"
)
def preprocess_fn(feature, label):
    return bert_preprocessor(feature), label
```
It uses the `BertPreprocessor` to convert input feature formats.

### Load Pretrained Model

As long as it is a Keras model, you can use it with this script. 

### Make the Finetuning Model

Users need to make a classification model based on your pretrained model for 
evaluation purposes. For example, [`BertClassifier`](https://github.com/keras-team/keras-nlp/blob/master/keras_nlp/models/bert/bert_classifier.py) takes a `Bert` model as backbone,
and adds a dense layer on top of it. Please pay attention that different model 
could use different classifier structure, e.g., in [RoBERTa](https://github.com/huggingface/transformers/blob/94b3f544a1f5e04b78d87a2ae32a7ac252e22e31/src/transformers/models/roberta/modeling_roberta.py#L1437-L1456), 
it has 2 dense layers. If you are using pretrained model from an OSS package, 
please find the correct classifier. If you use a custom model, you can start 
experimenting with a simple dense layer, and adjust the structure based on 
its performance.

## Flags Table

| Flags Name                 	| Explanation                                     	| Default 	|
|----------------------------	|-------------------------------------------------	|---------	|
| task_name                  	| The name of the GLUE task to finetune on.       	| "mrpc"  	|
| batch_size                 	| Data batch size                                 	| 32      	|
| epochs                     	| Number of epochs to run finetuning.             	| 2       	|
| learning_rate              	| The optimizer's learning rate.                  	| 5e-5    	|
| tpu_name               	    | The name of TPU to connect to.                    | None    	|
| submission_directory       	| The file path to save the glue submission file. 	| None    	|
| load_finetuning_model 	    | The path to load the finetuning model.          	| None    	|
| save_finetuning_model 	    | The path to save the finetuning model.          	| None    	|
