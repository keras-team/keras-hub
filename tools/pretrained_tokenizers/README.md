# Training WordPiece Vocabularies on Wikipedia

This is unmaintained helper code for training the vocabularies on Wikipedia.
It is advised to run these scripts on GCS.

Note: use either `screen` or `tmux` when running these commands remotely to
avoiding killing long running scripts.

## Instructions
The steps are listed below. You will need to run 1 and 2 for all wikipedia data
dumps that you want to train on to download and extract the data. 

After, change the list in the cleaning script so that it matches the downloaded 
data folder names

Finally, change the list in the train vocabulary and run the script to train the 
vocabularies for each directory in the list.

### 1. Download Wikipedia Dataset from Wikipedia Dumps
Example: `curl -O https://dumps.wikimedia.org/ptwiki/20220801/ptwiki-20220801-pages-articles-multistream.xml.bz2`

### 2. Run Wikipedia Dataset Extractor
Example: `python3 -m wikiextractor.WikiExtractor arwiki-20220802-pages-articles-multistream.xml.bz2`

### 3. Additional Removals
`python3 word_piece_cleaning_script.py`

### 4. Run train vocabulary
`python3 word_piece_training_script.py`