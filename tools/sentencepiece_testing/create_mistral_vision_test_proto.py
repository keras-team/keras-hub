from tools.sentencepiece_testing.utils import train_sentencepiece


def main():
    train_sentencepiece(
        ["the quick brown fox", "the earth is round"],
        "mistral_vision_test_vocab.spm",
        vocab_size=32,
        # `model_type="WORD"` (used by `create_mistral_test_proto.py` for the
        # plain text-only vocab) treats each whitespace-delimited chunk as a
        # single opaque token, so a `user_defined_symbol` only matches when
        # it equals an entire whitespace-delimited word. That breaks a
        # symbol like `[IMG]` sitting mid-sentence between other words.
        # `BPE` extracts `user_defined_symbols` as atomic substrings
        # regardless of surrounding whitespace, matching how real
        # Mistral/Pixtral vocabs handle special tokens embedded in text.
        model_type="BPE",
        pad_id=-1,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        user_defined_symbols=["[IMG]", "[IMG_BREAK]", "[IMG_END]"],
    )


if __name__ == "__main__":
    main()
