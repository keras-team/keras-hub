"""BERT model preset configurations."""

backbone_presets = {
    "bert_tiny_en_uncased": {
        "metadata": {
            "description": (
                "2-layer BERT model where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 4385920,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bert/keras/bert_tiny_en_uncased/3",
    },
    "bert_small_en_uncased": {
        "metadata": {
            "description": (
                "4-layer BERT model where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 28763648,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bert/keras/bert_small_en_uncased/3",
    },
    "bert_medium_en_uncased": {
        "metadata": {
            "description": (
                "8-layer BERT model where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 41373184,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bert/keras/bert_medium_en_uncased/3",
    },
    "bert_base_en_uncased": {
        "metadata": {
            "description": (
                "12-layer BERT model where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 109482240,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bert/keras/bert_base_en_uncased/3",
    },
    "bert_base_en": {
        "metadata": {
            "description": (
                "12-layer BERT model where case is maintained. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 108310272,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bert/keras/bert_base_en/3",
    },
    "bert_base_zh": {
        "metadata": {
            "description": (
                "12-layer BERT model. Trained on Chinese Wikipedia."
            ),
            "params": 102267648,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bert/keras/bert_base_zh/3",
    },
    "bert_base_multi": {
        "metadata": {
            "description": (
                "12-layer BERT model where case is maintained. Trained on "
                "trained on Wikipedias of 104 languages"
            ),
            "params": 177853440,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bert/keras/bert_base_multi/3",
    },
    "bert_large_en_uncased": {
        "metadata": {
            "description": (
                "24-layer BERT model where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 335141888,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bert/keras/bert_large_en_uncased/3",
    },
    "bert_large_en": {
        "metadata": {
            "description": (
                "24-layer BERT model where case is maintained. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "params": 333579264,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bert/keras/bert_large_en/3",
    },
    "bert_tiny_en_uncased_sst2": {
        "metadata": {
            "description": (
                "The bert_tiny_en_uncased backbone model fine-tuned on the "
                "SST-2 sentiment analysis dataset."
            ),
            "params": 4385920,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bert/keras/bert_tiny_en_uncased_sst2/5",
    },
    # Sentence-transformer models (BERT backbone, fine-tuned for embeddings).
    # "all-*" family: general-purpose sentence embedding models.
    "all_minilm_l6_v2_en": {
        "metadata": {
            "description": (
                "6-layer MiniLM sentence embedding model. Maps sentences "
                "to 384-dimensional dense vectors. Trained on 1B+ sentence "
                "pairs for semantic similarity, search, and clustering."
            ),
            "params": 22713216,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/sentence-transformers/keras/all_minilm_l6_v2_en/1",
    },
    "all_minilm_l6_v1_en": {
        "metadata": {
            "description": (
                "6-layer MiniLM sentence embedding model (v1). Maps "
                "sentences to 384-dimensional dense vectors."
            ),
            "params": 22713216,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/sentence-transformers/keras/all_minilm_l6_v1_en/1",
    },
    "all_minilm_l12_v2_en": {
        "metadata": {
            "description": (
                "12-layer MiniLM sentence embedding model. Maps sentences "
                "to 384-dimensional dense vectors. Higher accuracy than "
                "the L6 variant with moderate speed tradeoff."
            ),
            "params": 33360000,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/sentence-transformers/keras/all_minilm_l12_v2_en/1",
    },
    # "paraphrase-*" family: optimized for paraphrase detection.
    "paraphrase_minilm_l3_v2_en": {
        "metadata": {
            "description": (
                "3-layer MiniLM model for paraphrase detection. Ultra-fast "
                "with 384-dimensional sentence embeddings."
            ),
            "params": 17066496,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/sentence-transformers/keras/paraphrase_minilm_l3_v2_en/1",
    },
    "paraphrase_minilm_l6_v2_en": {
        "metadata": {
            "description": (
                "6-layer MiniLM model for paraphrase detection. Fast "
                "with 384-dimensional sentence embeddings."
            ),
            "params": 22713216,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/sentence-transformers/keras/paraphrase_minilm_l6_v2_en/1",
    },
    "paraphrase_minilm_l12_v2_en": {
        "metadata": {
            "description": (
                "12-layer MiniLM model for paraphrase detection with "
                "384-dimensional sentence embeddings."
            ),
            "params": 33360000,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/sentence-transformers/keras/paraphrase_minilm_l12_v2_en/1",
    },
    # "multi-qa-*" family: optimized for question answering / semantic search.
    "multi_qa_minilm_l6_cos_v1_en": {
        "metadata": {
            "description": (
                "6-layer MiniLM model for semantic search with cosine "
                "similarity. Trained on 215M QA pairs."
            ),
            "params": 22713216,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/sentence-transformers/keras/multi_qa_minilm_l6_cos_v1_en/1",
    },
    "multi_qa_minilm_l6_dot_v1_en": {
        "metadata": {
            "description": (
                "6-layer MiniLM model for semantic search with dot-product "
                "similarity. Trained on 215M QA pairs."
            ),
            "params": 22713216,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/sentence-transformers/keras/multi_qa_minilm_l6_dot_v1_en/1",
    },
    # "msmarco-*" family: optimized for information retrieval.
    "msmarco_minilm_l6_cos_v5_en": {
        "metadata": {
            "description": (
                "6-layer MiniLM model for information retrieval with "
                "cosine similarity. Trained on MS MARCO passage ranking."
            ),
            "params": 22713216,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/sentence-transformers/keras/msmarco_minilm_l6_cos_v5_en/1",
    },
    "msmarco_minilm_l12_cos_v5_en": {
        "metadata": {
            "description": (
                "12-layer MiniLM model for information retrieval with "
                "cosine similarity. Trained on MS MARCO passage ranking."
            ),
            "params": 33360000,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/sentence-transformers/keras/msmarco_minilm_l12_cos_v5_en/1",
    },
    # BGE family: BAAI General Embedding models optimized for dense retrieval.
    "bge_small_en": {
        "metadata": {
            "description": (
                "12-layer BGE small English embedding model (v1). Maps "
                "sentences to 384-dimensional L2-normalized dense vectors. "
                "Optimized for dense retrieval and semantic similarity."
            ),
            "params": 33360000,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bge/keras/bge_small_en/1",
    },
    "bge_base_en": {
        "metadata": {
            "description": (
                "12-layer BGE base English embedding model (v1). Maps "
                "sentences to 768-dimensional L2-normalized dense vectors. "
                "Optimized for dense retrieval and semantic similarity."
            ),
            "params": 109482240,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bge/keras/bge_base_en/1",
    },
    "bge_large_en": {
        "metadata": {
            "description": (
                "24-layer BGE large English embedding model (v1). Maps "
                "sentences to 1024-dimensional L2-normalized dense vectors. "
                "Highest accuracy in the BGE English v1 family."
            ),
            "params": 335141888,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bge/keras/bge_large_en/1",
    },
    "bge_small_en_v1.5": {
        "metadata": {
            "description": (
                "12-layer BGE small English embedding model (v1.5). Maps "
                "sentences to 384-dimensional L2-normalized dense vectors. "
                "Optimized for dense retrieval and semantic similarity."
            ),
            "params": 33360000,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bge/keras/bge_small_en_v1.5/1",
    },
    "bge_base_en_v1.5": {
        "metadata": {
            "description": (
                "12-layer BGE base English embedding model (v1.5). Maps "
                "sentences to 768-dimensional L2-normalized dense vectors. "
                "Optimized for dense retrieval and semantic similarity."
            ),
            "params": 109482240,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bge/keras/bge_base_en_v1.5/1",
    },
    "bge_large_en_v1.5": {
        "metadata": {
            "description": (
                "24-layer BGE large English embedding model (v1.5). Maps "
                "sentences to 1024-dimensional L2-normalized dense vectors. "
                "Highest accuracy in the BGE English family."
            ),
            "params": 335141888,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bge/keras/bge_large_en_v1.5/1",
    },
    "bge_small_zh": {
        "metadata": {
            "description": (
                "12-layer BGE small Chinese embedding model (v1). Maps "
                "sentences to 384-dimensional L2-normalized dense vectors. "
                "Optimized for dense retrieval on Chinese text."
            ),
            "params": 23953920,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bge/keras/bge_small_zh/1",
    },
    "bge_base_zh": {
        "metadata": {
            "description": (
                "12-layer BGE base Chinese embedding model (v1). Maps "
                "sentences to 768-dimensional L2-normalized dense vectors. "
                "Optimized for dense retrieval on Chinese text."
            ),
            "params": 102267648,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bge/keras/bge_base_zh/1",
    },
    "bge_large_zh": {
        "metadata": {
            "description": (
                "24-layer BGE large Chinese embedding model (v1). Maps "
                "sentences to 1024-dimensional L2-normalized dense vectors. "
                "Highest accuracy in the BGE Chinese v1 family."
            ),
            "params": 325522432,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bge/keras/bge_large_zh/1",
    },
    "bge_small_zh_v1.5": {
        "metadata": {
            "description": (
                "12-layer BGE small Chinese embedding model (v1.5). Maps "
                "sentences to 384-dimensional L2-normalized dense vectors. "
                "Optimized for dense retrieval on Chinese text."
            ),
            "params": 23953920,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bge/keras/bge_small_zh_v1.5/1",
    },
    "bge_base_zh_v1.5": {
        "metadata": {
            "description": (
                "12-layer BGE base Chinese embedding model (v1.5). Maps "
                "sentences to 768-dimensional L2-normalized dense vectors. "
                "Optimized for dense retrieval on Chinese text."
            ),
            "params": 102267648,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bge/keras/bge_base_zh_v1.5/1",
    },
    "bge_large_zh_v1.5": {
        "metadata": {
            "description": (
                "24-layer BGE large Chinese embedding model (v1.5). Maps "
                "sentences to 1024-dimensional L2-normalized dense vectors. "
                "Highest accuracy in the BGE Chinese v1.5 family."
            ),
            "params": 325522432,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bge/keras/bge_large_zh_v1.5/1",
    },
    "llm_embedder": {
        "metadata": {
            "description": (
                "BGE-LLM-Embedder: 12-layer embedding model for "
                "retrieval-augmented language model applications. Maps text to "
                "768-dimensional dense vectors and supports knowledge, memory, "
                "demonstration, and tool retrieval tasks."
            ),
            "params": 109482240,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bge/keras/llm_embedder/1",
    },
}
