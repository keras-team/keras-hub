from tools.sentencepiece_testing.utils import train_sentencepiece


def main():
    train_sentencepiece(
        ["the quick brown fox", "the earth is round"],
        "mistral_test_vocab.spm",
        vocab_size=10,
        model_type="WORD",
        pad_id=-1,
        unk_id=0,
        bos_id=1,
        eos_id=2,
    )


if __name__ == "__main__":
    main()
