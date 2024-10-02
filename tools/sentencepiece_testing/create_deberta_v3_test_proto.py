from tools.sentencepiece_testing.utils import train_sentencepiece


def main():
    train_sentencepiece(
        ["the quick brown fox", "the earth is round"],
        "deberta_v3_test_vocab.spm",
        vocab_size=12,
        model_type="WORD",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_piece="[PAD]",
        bos_piece="[CLS]",
        eos_piece="[SEP]",
        unk_piece="[UNK]",
        user_defined_symbols="[MASK]",
    )


if __name__ == "__main__":
    main()
