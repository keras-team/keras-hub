from tools.sentencepiece_testing.utils import train_sentencepiece


def main():
    train_sentencepiece(
        ["the quick brown fox", "the earth is round"],
        "f_net_test_vocab.spm",
        vocab_size=12,
        model_type="WORD",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="[CLS]",
        eos_piece="[SEP]",
        user_defined_symbols="[MASK]",
    )


if __name__ == "__main__":
    main()
