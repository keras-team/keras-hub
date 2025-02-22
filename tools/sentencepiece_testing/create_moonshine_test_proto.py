import os

import sentencepiece as spm


def create_test_vocab():
    with open("temp_training.txt", "w", encoding="utf-8") as f:
        f.write(
            """
            Hello world!
            Test with <<ST_42>>
            Hex test <0x1F>
            The quick brown fox
            jumped over the lazy dog
            testing special tokens.
            """
        )
    model_prefix = "moonshine_test_vocab"
    spm.SentencePieceTrainer.train(
        input="temp_training.txt",
        model_prefix=model_prefix,
        vocab_size=1205,
        character_coverage=1.0,
        model_type="bpe",
        bos_piece="<s>",
        eos_piece="</s>",
        pad_piece="<pad>",
        user_defined_symbols=[
            "<>",
        ]
        + [f"<0x{i:02X}>" for i in range(256)]
        + [f"<<ST_{i}>>" for i in range(768)],
    )
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
    target_path = os.path.join(test_data_dir, "moonshine_test_vocab.spm")
    if os.path.exists(target_path):
        os.remove(target_path)
    os.rename(f"{model_prefix}.model", target_path)
    if os.path.exists("temp_training.txt"):
        os.remove("temp_training.txt")
    if os.path.exists(f"{model_prefix}.vocab"):
        os.remove(f"{model_prefix}.vocab")


if __name__ == "__main__":
    create_test_vocab()
