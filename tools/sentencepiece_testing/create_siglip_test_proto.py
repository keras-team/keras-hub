from tools.sentencepiece_testing.utils import train_sentencepiece


def main():
    train_sentencepiece(
        ["the quick brown fox", "the earth is round"],
        "siglip_test_vocab.spm",
        vocab_size=11,
        model_type="WORD",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        unk_piece="<unk>",
    )


if __name__ == "__main__":
    main()
