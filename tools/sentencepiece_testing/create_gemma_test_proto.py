from tools.sentencepiece_testing.utils import train_sentencepiece


def main():
    train_sentencepiece(
        ["the quick brown fox", "the earth is round"],
        "gemma_test_vocab.spm",
        vocab_size=11,
        model_type="WORD",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_piece="<pad>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        unk_piece="<unk>",
    )

    train_sentencepiece(
        ["The quick brown fox jumped.", "I like pizza.", "This is a test."],
        "gemma_export_vocab.spm",
        vocab_size=290,
        model_type="unigram",
        pad_id=0,
        bos_id=2,
        eos_id=1,
        unk_id=3,
        byte_fallback=True,
        pad_piece="<pad>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        unk_piece="<unk>",
        user_defined_symbols=["<start_of_turn>", "<end_of_turn>"],
        add_dummy_prefix=False,
    )


if __name__ == "__main__":
    main()
