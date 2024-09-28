import os
import time

import keras_hub

# List directories of parsed Wikipedia articles and vocab sizes
directories = [
    "eswiki_parsed",
    "frwiki_parsed",
    "hiwiki_parsed",
    "arwiki_parsed",
    "ruwiki_parsed",
    "bnwiki_parsed",
    "idwiki_parsed",
    "ptwiki_parsed",
]
vocab_sizes = [20000, 50000]
identifier = "v1"

# Runs the computation
for directory in directories:
    for vocab_size in vocab_sizes:
        print(f"Running directory {directory} with vocab size {vocab_size}")
        files = []
        for folder in os.listdir(directory):
            path = os.path.join(directory, folder)
            for file in os.listdir(path):
                if file[0] != ".":
                    files.append(os.path.join(path, file))

        if os.path.exists(f"{directory}_{vocab_size}_{identifier}.txt"):
            raise ValueError("already done.")

        start = time.time()
        keras_hub.tokenizers.compute_word_piece_vocabulary(
            files,
            vocabulary_size=vocab_size,
            lowercase=False,
            strip_accents=False,
            vocabulary_output_file=f"{directory}_{vocab_size}_{identifier}.txt",
        )
        end = time.time()
        print("Time taken:", end - start)
