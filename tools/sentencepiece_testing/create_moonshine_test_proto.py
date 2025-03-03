import os

import sentencepiece as spm


def create_test_vocab():
    # Define all special tokens.
    special_tokens = (
        [
            "<s>",  # bos_piece
            "</s>",  # eos_piece
            "<pad>",  # pad_piece
            "<>",  # empty token
        ]
        + [f"<0x{i:02X}>" for i in range(256)]
        + [f"<<ST_{i}>>" for i in range(768)]
    )
    with open("temp_training.txt", "w", encoding="utf-8") as f:
        for token in special_tokens:
            f.write(f"{token}\n")
        f.write(
            """
            Hello world! The quick brown fox jumped over the lazy dog.
            This is a test of the tokenizer vocabulary creation.
            
            UPPERCASE TEXT
            lowercase text
            Mixed Case Text
            
            123456789 0.12345 -1234
            
            !@#$%^&*()_+-=[]{}|;':",./<>?`~
            
            Text with    multiple    spaces
            Text with tabs\t and newlines\n
            
            def example_function():
                return "This is code"
            
            <div>This is HTML</div>
            <script>console.log("JavaScript")</script>
            
            https://example.com
            user@example.com
            
            don't can't won't I'm you're they're
            John's Mary's the cat's
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
        user_defined_symbols=special_tokens,
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
