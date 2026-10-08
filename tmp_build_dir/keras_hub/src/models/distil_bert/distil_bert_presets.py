"""DistilBERT model preset configurations."""

backbone_presets = {
    "distil_bert_base_en_uncased": {
        "metadata": {
            "description": (
                "6-layer DistilBERT model where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus using BERT as the "
                "teacher model."
            ),
            "params": 66362880,
            "path": "distil_bert",
        },
        "kaggle_handle": "kaggle://keras/distil_bert/keras/distil_bert_base_en_uncased/3",
    },
    "distil_bert_base_en": {
        "metadata": {
            "description": (
                "6-layer DistilBERT model where case is maintained. "
                "Trained on English Wikipedia + BooksCorpus using BERT as the "
                "teacher model."
            ),
            "params": 65190912,
            "path": "distil_bert",
        },
        "kaggle_handle": "kaggle://keras/distil_bert/keras/distil_bert_base_en/3",
    },
    "distil_bert_base_multi": {
        "metadata": {
            "description": (
                "6-layer DistilBERT model where case is maintained. Trained on "
                "Wikipedias of 104 languages"
            ),
            "params": 134734080,
            "path": "distil_bert",
        },
        "kaggle_handle": "kaggle://keras/distil_bert/keras/distil_bert_base_multi/3",
    },
}
