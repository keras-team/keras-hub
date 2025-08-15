from tools.sentencepiece_testing.utils import train_sentencepiece


def main():
    train_sentencepiece(
        ["the quick brown fox", "the earth is round"],
        "llama_test_vocab.spm",
        vocab_size=10,
        model_type="WORD",
        pad_id=-1,
        unk_id=0,
        bos_id=1,
        eos_id=2,
    )

    train_sentencepiece(
        ["The quick brown fox jumped.", "I like pizza.", "This is a test."],
        "llama_export_vocab.spm",
        vocab_size=290,
        model_type="unigram",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        byte_fallback=True,
        pad_piece="<pad>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        unk_piece="<unk>",
        user_defined_symbols=["<s>", "</s>"],
        add_dummy_prefix=True,
    )


if __name__ == "__main__":
    main()
