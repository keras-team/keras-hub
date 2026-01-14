import numpy as np
from transformers import AutoTokenizer
import keras_hub
from modernbert_tokenizer import ModernBertTokenizer 

def check_parity():
    text = "ModernBERT is remarkably fast! [MASK]"
    
    # HF output
    hf_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    # ModernBERT usually doesn't add a prefix space by default, 
    # but check if your text starts with one.
    hf_ids = hf_tokenizer.encode(text, add_special_tokens=False)

    # kerashub output
    kh_tokenizer = ModernBertTokenizer(
        vocabulary="keras_hub/src/models/modernbert/vocab.json",
        merges="keras_hub/src/models/modernbert/merges.txt"
    )
    kh_ids = kh_tokenizer.tokenize(text).numpy().tolist()

    print(f"HF Output: {hf_ids}")
    print(f"KH Output: {kh_ids}")

    # Check for exact match
    if hf_ids == kh_ids:
        print(" Perfect Match!")
    else:
        print("IDs differ. Checking special tokens...")
        print(f"HF Pad ID: {hf_tokenizer.pad_token_id} | KH Pad ID: {kh_tokenizer.pad_token_id}")

if __name__ == "__main__":
    check_parity()