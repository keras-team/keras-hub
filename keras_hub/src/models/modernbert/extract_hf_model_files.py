import json
import requests

# tokenizer.json from Hugging Face
url = "https://huggingface.co/answerdotai/ModernBERT-base/raw/main/tokenizer.json"
response = requests.get(url)
data = response.json()

# Extract the vocabulary from the JSON structure
# In tokenizer.json, the vocab is under ["model"]["vocab"]
raw_vocab = data["model"]["vocab"]

added_tokens = {obj["content"]: obj["id"] for obj in data["added_tokens"]}
raw_vocab.update(added_tokens)

sorted_vocab = dict(sorted(raw_vocab.items(), key=lambda item: item[1]))

print("FINAL VERIFICATION\n")
print(f"Total Vocab Size: {len(sorted_vocab)}")
print(f"Index of 'Ġair': {sorted_vocab.get('Ġair')}")
print(f"Index of '<|padding|>': {sorted_vocab.get('<|padding|>')}")
print(f"Index of '[MASK]': {sorted_vocab.get('[MASK]')}")


with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(sorted_vocab, f, ensure_ascii=False)