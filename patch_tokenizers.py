import os
import re

files_to_patch = [
    "keras_hub/src/models/albert/albert_tokenizer.py",
    "keras_hub/src/models/deberta_v3/deberta_v3_tokenizer.py",
    "keras_hub/src/models/f_net/f_net_tokenizer.py"
]

files_to_add = [
    "keras_hub/src/models/t5/t5_tokenizer.py",
    "keras_hub/src/models/xlm_roberta/xlm_roberta_tokenizer.py"
]

patch_code_albert = """    def detokenize(self, inputs):
        if not hasattr(self, "special_tokens_map"):
            self.special_tokens_map = {
                self.cls_token_id: "[CLS]",
                self.sep_token_id: "[SEP]",
                self.pad_token_id: "<pad>",
                self.mask_token_id: "[MASK]",
            }
            self.special_tokens_map = {k: v for k, v in self.special_tokens_map.items() if k is not None}

        import tensorflow as tf
        if not tf.executing_eagerly():
            return super().detokenize(inputs)
            
        inputs_list = tf.convert_to_tensor(inputs).numpy().tolist()
        is_scalar = isinstance(inputs_list, int) or (len(inputs_list) > 0 and isinstance(inputs_list[0], int))
        if is_scalar: inputs_list = [inputs_list] if isinstance(inputs_list, list) else [[inputs_list]]

        decoded_outputs = []
        for seq in inputs_list:
            words, current_chunk = [], []
            def decode_and_append():
                if current_chunk:
                    decoded = super(self.__class__, self).detokenize(current_chunk)
                    if hasattr(decoded, "numpy"): decoded = decoded.numpy()
                    if isinstance(decoded, list) and len(decoded) > 0: decoded = decoded[0]
                    if isinstance(decoded, bytes): decoded = decoded.decode('utf-8')
                    words.append(str(decoded))
                    current_chunk.clear()

            for token_id in seq:
                if token_id in self.special_tokens_map:
                    decode_and_append()
                    words.append(self.special_tokens_map[token_id])
                else:
                    current_chunk.append(token_id)
            decode_and_append()
            decoded_outputs.append(" ".join(words).strip())

        return tf.convert_to_tensor(decoded_outputs[0]) if is_scalar else tf.convert_to_tensor(decoded_outputs)
"""

# For Deberta:
patch_code_deberta = patch_code_albert.replace(
    """self.special_tokens_map = {
                self.cls_token_id: "[CLS]",
                self.sep_token_id: "[SEP]",
                self.pad_token_id: "<pad>",
                self.mask_token_id: "[MASK]",
            }
            self.special_tokens_map = {k: v for k, v in self.special_tokens_map.items() if k is not None}""",
    """self.special_tokens_map = {}
            for token, id_val in zip(self.special_tokens, self.special_token_ids):
                if id_val is not None: self.special_tokens_map[id_val] = token"""
)

# For FNet:
patch_code_fnet = patch_code_albert.replace(
    """self.special_tokens_map = {
                self.cls_token_id: "[CLS]",
                self.sep_token_id: "[SEP]",
                self.pad_token_id: "<pad>",
                self.mask_token_id: "[MASK]",
            }
            self.special_tokens_map = {k: v for k, v in self.special_tokens_map.items() if k is not None}""",
    """self.special_tokens_map = {
                self.cls_token_id: "[CLS]",
                self.sep_token_id: "[SEP]",
                self.pad_token_id: "<pad>",
                self.mask_token_id: "<mask>",
            }
            self.special_tokens_map = {k: v for k, v in self.special_tokens_map.items() if k is not None}"""
)

# For T5:
patch_code_t5 = patch_code_albert.replace(
    """self.special_tokens_map = {
                self.cls_token_id: "[CLS]",
                self.sep_token_id: "[SEP]",
                self.pad_token_id: "<pad>",
                self.mask_token_id: "[MASK]",
            }
            self.special_tokens_map = {k: v for k, v in self.special_tokens_map.items() if k is not None}""",
    """self.special_tokens_map = {
                self.end_token_id: "</s>",
                self.pad_token_id: "<pad>",
            }
            self.special_tokens_map = {k: v for k, v in self.special_tokens_map.items() if k is not None}"""
)

# For XLM-R:
patch_code_xlmr = patch_code_albert.replace(
    """self.special_tokens_map = {
                self.cls_token_id: "[CLS]",
                self.sep_token_id: "[SEP]",
                self.pad_token_id: "<pad>",
                self.mask_token_id: "[MASK]",
            }
            self.special_tokens_map = {k: v for k, v in self.special_tokens_map.items() if k is not None}""",
    """self.special_tokens_map = {
                self.cls_token_id: "<s>",
                self.sep_token_id: "</s>",
                self.pad_token_id: "<pad>",
                self.mask_token_id: "<mask>",
            }
            self.special_tokens_map = {k: v for k, v in self.special_tokens_map.items() if k is not None}"""
)

def update_file(path, code):
    with open(path, 'r') as f:
        content = f.read()
    
    if "def detokenize(self, inputs):" in content:
        # Regex to match the detokenize method and all its indented body
        pattern = r"    def detokenize\(self, inputs\):.*?(?=\n    def |\Z)"
        new_content = re.sub(pattern, code, content, flags=re.DOTALL)
    else:
        new_content = content + "\n" + code
        
    with open(path, 'w') as f:
        f.write(new_content)
    print(f"Updated {path}")

update_file(files_to_patch[0], patch_code_albert)
update_file(files_to_patch[1], patch_code_deberta)
update_file(files_to_patch[2], patch_code_fnet)
update_file(files_to_add[0], patch_code_t5)
update_file(files_to_add[1], patch_code_xlmr)

