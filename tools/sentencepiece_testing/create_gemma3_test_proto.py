from tools.sentencepiece_testing.utils import train_sentencepiece


def main():
    train_sentencepiece(
        ["the quick brown fox", "the earth is round"],
        "gemma3_test_vocab.spm",
        vocab_size=16,
        model_type="WORD",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_piece="<pad>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        unk_piece="<unk>",
        control_symbols=[
            "<start_of_image>",
            "<end_of_image>",
            "<start_of_turn>",
            "<end_of_turn>",
            "<img>",
        ],
    )


if __name__ == "__main__":
    main()
