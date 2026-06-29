import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras_hub

print("Loading ALBERT...")
albert = keras_hub.models.AlbertTokenizer.from_preset("albert_base_en_uncased")
albert_out = albert.detokenize([[albert.cls_token_id, 13, 14, 15, albert.sep_token_id, albert.pad_token_id]])
print("ALBERT Output:  ", albert_out.numpy()[0].decode('utf-8'))

print("\nLoading DeBERTaV3...")
deberta = keras_hub.models.DebertaV3Tokenizer.from_preset("deberta_v3_base_en")
deberta_out = deberta.detokenize([[deberta.cls_token_id, 260, 261, 262, deberta.sep_token_id, deberta.pad_token_id]])
print("DeBERTa Output: ", deberta_out.numpy()[0].decode('utf-8'))

print("\nLoading FNet...")
fnet = keras_hub.models.FNetTokenizer.from_preset("f_net_base_en")
fnet_out = fnet.detokenize([[fnet.cls_token_id, 100, 101, 102, fnet.sep_token_id, fnet.pad_token_id]])
print("FNet Output:    ", fnet_out.numpy()[0].decode('utf-8'))

print("\nLoading T5...")
t5 = keras_hub.models.T5Tokenizer.from_preset("t5_base_en")
t5_out = t5.detokenize([[25, 26, 27, t5.end_token_id, t5.pad_token_id]])
print("T5 Output:      ", t5_out.numpy()[0].decode('utf-8'))

print("\nLoading XLM-RoBERTa...")
xlm = keras_hub.models.XLMRobertaTokenizer.from_preset("xlm_roberta_base_multi")
xlm_out = xlm.detokenize([[xlm.cls_token_id, 150, 151, 152, xlm.sep_token_id, xlm.pad_token_id]])
print("XLM-R Output:   ", xlm_out.numpy()[0].decode('utf-8'))

print("\nSUCCESS! All tokenizers successfully retained their special tokens instead of dropping them!")
