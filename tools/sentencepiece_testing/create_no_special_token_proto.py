from tools.sentencepiece_testing.utils import train_sentencepiece


def main():
    train_sentencepiece(
        ["abc"],
        "no_special_token_vocab.spm",
        vocab_size=5,
        pad_id=-1,
        eos_id=-1,
        bos_id=-1,
    )


if __name__ == "__main__":
    main()
