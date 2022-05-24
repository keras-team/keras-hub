# KerasNLP Modeling Tools

This directory contains runnable scripts that are not specific to a specific
model architecture, but still useful for end-to-end workflows.

## split_sentences.py

The `split_sentences.py` script will process raw input files and split them into
output files where each line contains a sentence, and a blank line marks the
start of a new document. This is useful for tasks like next sentence prediction
where the boundaries between sentences are needed for training.

The script supports two types of inputs files. Plain text files, where each
individual file is assumed to be an entire document, and wikipedia dump files
in the format outputted by the wikiextractor tool (each document is enclosed in
`<doc>` tags).

Example usage:

```shell
python examples/tools/split_sentences.py \
    --input_files ~/datasets/wikipedia,~/datasets/bookscorpus \
    --output_directory ~/datasets/sentence-split-data
```

### train_word_piece_vocabulary.py

The `train_word_piece_vocabulary.py` script allows you to compute your own
WordPiece vocabulary.

Example usage:

```shell
python examples/tools/train_word_piece_vocabulary.py \
    --input_files ~/datasets/my-raw-dataset/ \
    --output_file vocab.txt
```
