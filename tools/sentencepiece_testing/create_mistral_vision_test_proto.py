from tools.sentencepiece_testing.utils import train_sentencepiece


def main():
    train_sentencepiece(
        ["the quick brown fox", "the earth is round"],
        "mistral_vision_test_vocab.spm",
        vocab_size=32,
        # `model_type="WORD"` (used by `create_mistral_test_proto.py`) only
        # matches a `user_defined_symbol` against a whole whitespace-
        # delimited word, breaking a symbol like `[IMG]` sitting
        # mid-sentence. `BPE` extracts `user_defined_symbols` as atomic
        # substrings regardless of whitespace, matching real Mistral/
        # Pixtral vocabs.
        model_type="BPE",
        pad_id=-1,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        user_defined_symbols=["[IMG]", "[IMG_BREAK]", "[IMG_END]"],
    )


if __name__ == "__main__":
    main()
