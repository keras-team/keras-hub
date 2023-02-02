# BERT with KerasNLP

This example demonstrates how to train a Bidirectional Encoder
Representations from Transformers (BERT) model end-to-end using the KerasNLP
library. This README contains instructions on how to run pretraining directly
from raw data, followed by finetuning and evaluation on the GLUE dataset.

## Quickly test out the code

To exercise the code in this directory by training a tiny BERT model, you can
run the following commands from the base directory of the repository. This can
be useful to validate any code changes, but note that a useful BERT model would
need to be trained for much longer on a much larger dataset.

```shell
OUTPUT_DIR=~/bert_test_output
DATA_URL=https://storage.googleapis.com/tensorflow/keras-nlp/examples/bert

# Download example data.
wget ${DATA_URL}/bert_vocab_uncased.txt -O $OUTPUT_DIR/bert_vocab_uncased.txt
wget ${DATA_URL}/wiki_example_data.txt -O $OUTPUT_DIR/wiki_example_data.txt

# Parse input data and split into sentences.
python3 examples/tools/split_sentences.py \
    --input_files $OUTPUT_DIR/wiki_example_data.txt \
    --output_directory $OUTPUT_DIR/sentence-split-data
# Preprocess input for pretraining.
python3 examples/bert_pretraining/bert_create_pretraining_data.py \
    --input_files $OUTPUT_DIR/sentence-split-data/ \
    --vocab_file $OUTPUT_DIR/bert_vocab_uncased.txt \
    --output_file $OUTPUT_DIR/pretraining-data/pretraining.tfrecord
# Run pretraining for 100 train steps only.
python3 examples/bert_pretraining/bert_pretrain.py \
    --input_directory $OUTPUT_DIR/pretraining-data/ \
    --vocab_file $OUTPUT_DIR/bert_vocab_uncased.txt \
    --saved_model_output $OUTPUT_DIR/model/ \
    --num_train_steps 100
```

## Installing dependencies

This example needs a few extra dependencies to run (e.g. wikiextractor for
using wikipedia downloads). You can install these into a KerasNLP development
environment with:

```shell
pip install -r "examples/bert_pretraining/requirements.txt"
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

The ` bert_create_pretraining_data.py` script will take in a set of sentence split files, and
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
    python3 examples/bert_pretraining/bert_create_pretraining_data.py \
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
    echo python3 examples/bert_pretraining/bert_create_pretraining_data.py \
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

After preprocessing, we can run pretraining with the `bert_pretrain.py`
script. This will train a model and save it to the `--saved_model_output`
directory. If you are willing to train from data stored on google cloud storage bucket (GCS), you can do it by setting the file path to
the URL of GCS bucket. For example, `--input_directory=gs://your-bucket-name/you-data-path`. You can also save models directly to GCS by the same approach.

```shell
python3 examples/bert_pretraining/bert_pretrain.py \
    --input_directory path/to/data/ \
    --vocab_file path/to/bert_vocab_uncased.txt \
    --model_size tiny \
    --saved_model_output path/to/model/
```
