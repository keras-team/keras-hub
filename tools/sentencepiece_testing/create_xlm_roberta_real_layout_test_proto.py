from tools.sentencepiece_testing.utils import train_sentencepiece


def main():
    # Unlike `xlm_roberta_test_vocab.spm`, this proto has the same layout as
    # the XLM-RoBERTa presets: `"<unk>"`, `"<s>"` and `"</s>"` are the first
    # three pieces, and there is no `"<pad>"` piece.
    train_sentencepiece(
        ["the quick brown fox", "the earth is round"],
        "xlm_roberta_real_layout_test_vocab.spm",
        vocab_size=10,
        model_type="WORD",
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=-1,
    )


if __name__ == "__main__":
    main()
