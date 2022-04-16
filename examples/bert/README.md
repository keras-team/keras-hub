# BERT with keras-nlp

This example will show how to train a Bidirectional Encoder
Representations from Transformers (BERT) model end-to-end using the keras-nlp
library. This README contains instructions on how to run pretraining directly
from raw data, followed by fine tuning and evaluation on the GLUE dataset.

## Quickly test out the code

To exercise the code in this directory by training a tiny BERT model, you can
run the following commands from the base of the keras-nlp repository. This can
be useful to validate any code changes, but note that a useful BERT model would
need to be trained for much longer on a much larger dataset.

```shell
OUTPUT_DIR=~/bert_test_output
DATA_URL=https://storage.googleapis.com/tensorflow/keras-nlp/examples/bert

# Create a virtual env and install dependencies.
mkdir $OUTPUT_DIR
python3 -m venv $OUTPUT_DIR/env
source $OUTPUT_DIR/env/bin/activate
pip install -e ".[tests,examples]"

# Download example data.
wget ${DATA_URL}/bert_vocab_uncased.txt -O $OUTPUT_DIR/bert_vocab_uncased.txt
wget ${DATA_URL}/wiki_example_data.txt -O $OUTPUT_DIR/wiki_example_data.txt

# Run preprocessing.
python3 examples/bert/create_sentence_split_data.py \
    --input_files $OUTPUT_DIR/wiki_example_data.txt \
    --output_directory $OUTPUT_DIR/sentence-split-data --num_shards 1
python3 examples/bert/create_pretraining_data.py \
    --input_files $OUTPUT_DIR/sentence-split-data/ \
    --vocab_file $OUTPUT_DIR/bert_vocab_uncased.txt \
    --output_file $OUTPUT_DIR/pretraining-data/pretraining.tfrecord

# Run pretraining.
python3 examples/bert/run_pretraining.py \
    --input_files $OUTPUT_DIR/pretraining-data/ \
    --vocab_file $OUTPUT_DIR/bert_vocab_uncased.txt \
    --bert_config_file examples/bert/configs/bert_tiny.json \
    --num_warmup_steps 20 \
    --num_train_steps 200 \
    --saved_model_output $OUTPUT_DIR/model/

# Run finetuning.
python3 examples/bert/run_glue_finetuning.py \
    --saved_model_input $OUTPUT_DIR/model/ \
    --vocab_file $OUTPUT_DIR/bert_vocab_uncased.txt \
    --bert_config_file examples/bert/configs/bert_tiny.json
```

## Installing dependencies

Pip dependencies for all keras-nlp examples are listed in `setup.py`. To install
both the keras-nlp library from source and all other dependencies required to
run the example, run the below command. You may want to install to a self
contained environment (e.g. a container or a virtualenv).

```shell
pip install -e ".[examples]"
```

## Pretraining BERT

Training a BERT model happens in two stages. First, the model is "pretrained" on
a large corpus of input text. This is computationally expensive. After
pretraining, the model can be "fine tuned" on a downstream task with much
smaller amount of labeled data.

### Downloading pretraining data

The GLUE pretraining data (Wikipedia + BooksCorpus) is fairly large. The raw
input data takes roughly ~20GB of space, and after preprocessing, the full
corpus will take ~400GB.

The latest wikipedia dump can be downloaded [at this link](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2).
The dump can be extracted with the `wikiextractor` tool.

```shell
python -m wikiextractor.WikiExtractor enwiki-latest-pages-articles.xml.bz2
```

BooksCorpus is no longer hosted by
[it's creators](https://yknzhu.wixsite.com/mbweb), but you can find instructions
for downloading or reproducing the corpus in this
[repository](https://github.com/soskek/bookcorpus). We suggest the pre-made file
downloads listed at the top of the README. Alternatively, you can forgo it
entirely and pretrain solely on wikipedia.

Preparing the pretraining data will happen in two stages. First, raw text needs
to be split into lists of sentences per document. Second, this sentence split
data needs to use to create training examples with both masked words and
next sentence predictions.

### Splitting raw text into sentences

The `create_sentence_split_data.py` will process raw input files and split them
into output files where each line contains a sentence, and a blank line marks
the start of a new document.

The script supports two types of inputs files. Plain text files, where each
individual file is assumed to be an entire document, and wikipedia dump files
in the format outputted by the wikiextractor tool (each document is enclosed in
`<doc>` tags).

For example, if wikipedia files are located in `~/datasets/wikipedia` and
bookscorpus in `~/datasets/bookscorpus`, the following command will output
sentence split documents to a configurable number of output file shards:

```shell
python examples/bert/create_sentence_split_data.py \
    --input_files ~/datasets/wikipedia,~/datasets/bookscorpus \
    --output_directory ~/datasets/sentence-split-data
```

### Computing a WordPiece vocabulary

The `create_vocabulary.py` script allows you to compute your own WordPiece
vocabulary for use with BERT. In most cases however, it is desirable to use the
standard BERT vocabularies from the original models. You can download the
English uncased vocabulary
[here](https://storage.googleapis.com/tensorflow/keras-nlp/examples/bert/bert_vocab_uncased.txt).

### Tokenize, mask, and combine sentences into training examples

The `create_pretraining_data.py` scrip will take in a set of sentence split
files, and set up training examples for the next sentence prediction and masked
word tasks.

The output of the script will be TFRecord files with a number of fields per
example. Below shows a complete output example with the addition of a string
`tokens` field for clarity. The actual script will only serialize the token ids
to conserve disk space.

```python
tokens:  ['[CLS]', 'resin', '##s', 'are', 'today', '[MASK]', 'produced', 'by', 
          'ang', '##ios', '##per', '##ms', ',', 'and', 'tend', 'to', '[SEP]', 
          '[MASK]', 'produced', 'a', '[MASK]', '[MASK]', 'of', 'resin', ',', 
          'which', '[MASK]', 'often', 'found', 'as', 'amber', '[SEP]']
input_ids:  [101, 24604, 2015, 2024, 2651, 103, 2550, 2011, 17076, 10735, 4842,
             5244, 1010, 1998, 7166, 2000, 102, 103, 2550, 1037, 103, 103, 1997,
             24604, 1010, 2029, 103, 2411, 2179, 2004, 8994, 102]
input_mask:  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
segment_ids:  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
masked_lm_positions:  [5, 17, 20, 21, 26]
masked_lm_ids:  [2069, 3619, 2353, 2828, 2003]
masked_lm_weights:  [1.0, 1.0, 1.0, 1.0, 1.0]
next_sentence_labels:  [0]
```

In order to set up the next sentence prediction task, the script will load the
entire input into memory. As such, it is recommended to run this script on a
subset of the input data at a time.

For example, you can run the script on each file shard in a directory
with the following:

```shell
for file in path/to/sentence-split-data/*; do
    output="path/to/pretraining-data/$(basename -- "$file" .txt).tfrecord"
    python examples/bert/create_pretraining_data.py \
        --input_files ${file} \
        --vocab_file vocab.txt \
        --output_file ${output}
done
```

If memory is available, this could be further sped up by running this script
multiple times in parallel:

```shell
NUM_JOBS=5
for file in path/to/sentence-split-data/*; do
    output="path/to/pretraining-data/$(basename -- "$file" .txt).tfrecord"
    echo python examples/bert/create_pretraining_data.py \
        --input_files ${file} \
        --vocab_file vocab.txt \
        --output_file ${output}
done | parallel -j ${NUM_JOBS}
```

### Running BERT pretraining

After preprocessing, we can run pretraining with the `run_pretraining.py`
script. This will train a model and save it to the `--saved_model_output`
directory.

```shell
python3 examples/bert/run_pretraining.py \
    --input_files path/to/data/ \
    --vocab_file path/to/bert_vocab_uncased.txt \
    --bert_config_file examples/bert/configs/bert_tiny.json \
    --saved_model_output path/to/model/
```

## Evaluating BERT with GLUE

After pretraining, we can evaluate the performance of a BERT model with the
General Language Understanding Evaluation (GLUE) benchmark. This will
fine tune the model and running classification for a number of downstream tasks.

The `run_glue_finetuning.py` script downloads the GLUE data for a specific
tasks, reloads the pretraining model with appropriate finetuning heads, and runs
training for a few epochs to finetune the model.

```shell
python3 examples/bert/run_glue_finetuning.py \
    --saved_model_input path/to/model/ \
    --vocab_file path/to/bert_vocab_uncased.txt \
    --bert_config_file examples/bert/configs/bert_tiny.json \
```

The script could be easily adapted to any other text classification fine-tuning
tasks, where inputs can be any number of raw text sentences per sample.
