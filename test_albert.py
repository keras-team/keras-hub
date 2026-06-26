import os
# set backend to numpy or tensorflow
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
from keras_hub.src.models.albert.albert_tokenizer import AlbertTokenizer

proto_path = "/Users/yammani/keras-hub-1/keras_hub/src/tests/test_data/albert_test_vocab.spm"
tokenizer = AlbertTokenizer(proto=proto_path)

input_data = [[tokenizer.cls_token_id, 5, 10, 6, 8, tokenizer.sep_token_id, tokenizer.pad_token_id]]
print("Token IDs:", input_data)
output = tokenizer.detokenize(input_data)
print("Output:", output)
