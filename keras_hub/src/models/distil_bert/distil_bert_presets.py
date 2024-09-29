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
            "official_name": "DistilBERT",
            "path": "distil_bert",
            "model_card": "https://huggingface.co/distilbert-base-uncased",
        },
        "kaggle_handle": "kaggle://keras/distil_bert/keras/distil_bert_base_en_uncased/2",
    },
    "distil_bert_base_en": {
        "metadata": {
            "description": (
                "6-layer DistilBERT model where case is maintained. "
                "Trained on English Wikipedia + BooksCorpus using BERT as the "
                "teacher model."
            ),
            "params": 65190912,
            "official_name": "DistilBERT",
            "path": "distil_bert",
            "model_card": "https://huggingface.co/distilbert-base-cased",
        },
        "kaggle_handle": "kaggle://keras/distil_bert/keras/distil_bert_base_en/2",
    },
    "distil_bert_base_multi": {
        "metadata": {
            "description": (
                "6-layer DistilBERT model where case is maintained. Trained on Wikipedias of 104 languages"
            ),
            "params": 134734080,
            "official_name": "DistilBERT",
            "path": "distil_bert",
            "model_card": "https://huggingface.co/distilbert-base-multilingual-cased",
        },
        "kaggle_handle": "kaggle://keras/distil_bert/keras/distil_bert_base_multi/2",
    },
}
