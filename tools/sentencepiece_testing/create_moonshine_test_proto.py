import os

from tools.sentencepiece_testing.utils import train_sentencepiece


def create_moonshine_test_vocab():
    special_tokens = (
        [
            "<>",  # empty token
        ]
        + [f"<0x{i:02X}>" for i in range(256)]
        + [f"<<ST_{i}>>" for i in range(768)]
    )
    training_texts = ["the quick brown fox", "the earth is round"]
    test_data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "keras_hub",
        "src",
        "tests",
        "test_data",
    )
    os.makedirs(test_data_dir, exist_ok=True)
    model_prefix = "moonshine_test_vocab"
    target_path = os.path.join(test_data_dir, f"{model_prefix}.spm")
    train_sentencepiece(
        training_texts,
        target_path,
        vocab_size=11 + len(special_tokens),
        model_type="WORD",
        pad_id=0,  # <pad> token
        unk_id=1,  # <unk> token
        bos_id=2,  # <s> token
        eos_id=3,  # </s> token
        user_defined_symbols=special_tokens,
    )
    print(f"Moonshine test vocabulary created at: {target_path}")


if __name__ == "__main__":
    create_moonshine_test_vocab()
