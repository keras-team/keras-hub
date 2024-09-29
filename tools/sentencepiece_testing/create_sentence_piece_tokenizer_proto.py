from tools.sentencepiece_testing.utils import train_sentencepiece


def main():
    train_sentencepiece(
        ["the quick brown fox."],
        "tokenizer_test_vocab.spm",
        vocab_size=7,
        model_type="WORD",
    )


if __name__ == "__main__":
    main()
