# BERT with KerasNLP

This example demonstrates how to train a Bidirectional Encoder
Representations from Transformers (BERT) model end-to-end using the KerasNLP
library. This README contains instructions on how to run pretraining directly
from raw data, followed by finetuning and evaluation on the GLUE dataset.

## Installing dependencies

Pip dependencies for all KerasNLP examples are listed in `setup.py`. The
following command will create a virtual environment, install all dependencies,
and install KerasNLP from source.

```shell
python3 -m venv path/to/venv && source path/to/venv/bin/activate
pip install -e ".[examples]"
```

## Test out the code

You can use a pytest target to test changes to these scripts. Testing will
run through all the scripts on a small amount of data.

```shell
pytest examples/bert/bert_test.py
```

## Pretraining BERT

Training a BERT model happens in two stages. First, the model is "pretrained" on
a large corpus of input text. This is computationally expensive. After
pretraining, the model can be "finetuned" on a downstream task with a much
smaller amount of labeled data.

### Downloading pretraining data

The GLUE pretraining data (Wikipedia + BooksCorpus) is fairly large. The raw
input data takes roughly ~20GB of space, and after preprocessing, the full
corpus will take ~400GB.

The latest wikipedia dump can be downloaded
[at this link](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2),
or via command line:

```shell
curl -O https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
```
The dump can be extracted with the `wikiextractor` tool.

```shell
python3 -m wikiextractor.WikiExtractor enwiki-latest-pages-articles.xml.bz2
```

BooksCorpus is no longer hosted by
[its creators](https://yknzhu.wixsite.com/mbweb), but you can find instructions
for downloading or reproducing the corpus in
[this repository](https://github.com/soskek/bookcorpus). We suggest the pre-made file
downloads listed at the top of the README. Alternatively, you can forgo it
entirely and pretrain solely on wikipedia.

Preparing the pretraining data will happen in two stages. First, raw text needs
to be split into lists of sentences per document. Second, this sentence split
data needs to use to create training examples with both masked words and
next sentence predictions.

### Splitting raw text into sentences

Next, use `examples/tools/split_sentences.py` to process raw input files and
split them into output files where each line contains a sentence, and a blank
line marks the start of a new document. We need this for the next-sentence
prediction task used by BERT.

For example, if Wikipedia files are located in `~/datasets/wikipedia` and
bookscorpus in `~/datasets/bookscorpus`, the following command will output
sentence split documents to a configurable number of output file shards:

```shell
python3 examples/tools/split_sentences.py \
    --input_files ~/datasets/wikipedia,~/datasets/bookscorpus \
    --output_directory ~/datasets/sentence-split-data
```

### Computing a WordPiece vocabulary

The easiest and best approach when training BERT is to use the official
vocabularies from the original project, which have become somewhat standard.

You can download the English uncased vocabulary
[here](https://storage.googleapis.com/tensorflow/keras-nlp/examples/bert/bert_vocab_uncased.txt),
or in your terminal run:

```shell
curl -O https://storage.googleapis.com/tensorflow/keras-nlp/examples/bert/bert_vocab_uncased.txt
```

You can also use `examples/tools/train_word_piece_vocab.py` to train your own.

### Tokenize, mask, and combine sentences into training examples

The `bert_preprocess.py` script will take in a set of sentence split files, and
set up training examples for the next sentence prediction and masked word tasks.

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
    python3 examples/bert/bert_preprocess.py \
        --input_files ${file} \
        --vocab_file bert_vocab_uncased.txt \
        --output_file ${output}
done
```

If enough memory is available, this could be further sped up by running this script
multiple times in parallel. The following will take 3-4 hours on the entire dataset
on an 8 core machine.

```shell
NUM_JOBS=5
for file in path/to/sentence-split-data/*; do
    output="path/to/pretraining-data/$(basename -- "$file" .txt).tfrecord"
    echo python3 examples/bert/bert_preprocess.py \
        --input_files ${file} \
        --vocab_file bert_vocab_uncased.txt \
        --output_file ${output}
done | parallel -j ${NUM_JOBS}
```

To preview a sample of generated data files, you can run the command below:

```shell
python3 -c "from examples.utils.data_utils import preview_tfrecord; preview_tfrecord('path/to/tfrecord_file')"
```

### Running BERT pretraining

After preprocessing, we can run pretraining with the `bert_train.py`
script. This will train a model and save it to the `--saved_model_output`
directory.

```shell
python3 examples/bert/bert_train.py \
    --input_files path/to/data/ \
    --vocab_file path/to/bert_vocab_uncased.txt \
    --saved_model_output path/to/model/
```

## Evaluating BERT with GLUE

After pretraining, we can evaluate the performance of a BERT model with the
General Language Understanding Evaluation (GLUE) benchmark. This will
finetune the model and running classification for a number of downstream tasks.

The `bert_finetune_glue.py` script downloads the GLUE data for a specific
tasks, reloads the pretraining model with appropriate finetuning heads, and runs
training for a few epochs to finetune the model.

```shell
python3 examples/bert/bert_finetune_glue.py \
    --saved_model_input path/to/model/ \
    --vocab_file path/to/bert_vocab_uncased.txt
```

The script could be easily adapted to any other text classification finetuning
tasks, where inputs can be any number of raw text sentences per sample.
