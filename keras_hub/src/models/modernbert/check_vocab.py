# import json
# with open("keras_hub/src/models/modernbert/vocab.json", "r") as f:
#     vocab = json.load(f)
#     print(f"Index of <|padding|>: {vocab.get('<|padding|>')}")
#     print(f"Index of [MASK]: {vocab.get('[MASK]')}")


from modernbert_tokenizer import ModernBertTokenizer

tokenizer = ModernBertTokenizer(vocabulary="keras_hub/src/models/modernbert/vocab.json", 
                                merges="keras_hub/src/models/modernbert/merges.txt")

print(f"Vocab size type: {type(tokenizer.vocabulary_size)}")
print(f"Vocab size value: {tokenizer.vocabulary_size}")
print(f"Mask ID value: {tokenizer.mask_token_id}")