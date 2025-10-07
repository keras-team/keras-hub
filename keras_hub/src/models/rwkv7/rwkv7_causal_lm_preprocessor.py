import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.rwkv7.rwkv7_backbone import RWKV7Backbone
from keras_hub.src.models.rwkv7.rwkv7_tokenizer import RWKVTokenizer


@keras_hub_export("keras_hub.models.RWKV7CausalLMPreprocessor")
class RWKV7CausalLMPreprocessor(CausalLMPreprocessor):
    backbone_cls = RWKV7Backbone
    tokenizer_cls = RWKVTokenizer

    def __init__(
        self,
        tokenizer,
        add_start_token=False,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer, add_start_token=add_start_token, **kwargs
        )

    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
    ):
        sequence_length = sequence_length or self.sequence_length
        # padding 长度到16的倍数，适应kernel的需求
        sequence_length = sequence_length + (16 - sequence_length % 16)
        x = self.tokenizer(x)

        token_ids, padding_mask = self.packer(
            x, sequence_length=sequence_length, add_end_value=False
        )

        # The last token does not have a next token, so we truncate it out.
        x = token_ids[..., :-1]
        # Target `y` will be the next token.
        y, sample_weight = token_ids[..., 1:], padding_mask[..., 1:]
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    def build(self, input_shape):
        # Defer packer creation to `build()` so that we can be sure tokenizer
        # assets have loaded when restoring a saved model.
        self.packer = StartEndPacker(
            start_value=None,
            end_value=None,
            pad_value=self.tokenizer.pad_token_id,
            sequence_length=self.sequence_length,
            return_padding_mask=True,
            padding_side="left",
        )
        self.built = True

    def generate_preprocess(
        self,
        x,
        sequence_length=None,
    ):
        """Convert strings to integer token input for generation.

        Similar to calling the layer for training, this method takes in strings
        or tensor strings, tokenizes and packs the input, and computes a padding
        mask masking all inputs not filled in with a padded value.

        Unlike calling the layer for training, this method does not compute
        labels and will never append a `tokenizer.end_token_id` to the end of
        the sequence (as generation is expected to continue at the end of the
        inputted prompt).
        """
        if not self.built:
            self.build(None)
        # 这么做的目的是为了对齐keras的api
        # 输入的sequence_length是生成的最大长度
        # 而本身sequence_length则对应于prefill的最大长度
        generate_length = sequence_length
        sequence_length = self.sequence_length

        # padding 长度到16的倍数，适应kernel的需求
        sequence_length = sequence_length + (16 - sequence_length % 16)
        generate_length = generate_length + (16 - generate_length % 16)

        x = [t[-sequence_length:] for t in self.tokenizer(x)]
        y = ops.zeros((len(x), generate_length), "int32")
        start_token = [[t[-1]] for t in x]
        x = [t[:-1] if len(t) > 1 else [0] for t in x]

        token_ids, __ = self.packer(
            x, sequence_length=sequence_length, add_end_value=False
        )
        start_token = ops.convert_to_tensor(start_token, "int32")
        y = ops.slice_update(y, [0, 0], start_token)
        padding_mask = ops.not_equal(y, 0)

        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "predict_token_ids": y,
        }

    def generate_postprocess(
        self,
        x,
    ):
        """Convert integer token output to strings for generation.

        This method reverses `generate_preprocess()`, by first removing all
        padding and start/end tokens, and then converting the integer sequence
        back to a string.
        """
        if not self.built:
            self.build(None)

        token_ids, padding_mask = x["token_ids"], x["padding_mask"]
        token_ids = ops.convert_to_numpy(token_ids)
        padding_mask = ops.convert_to_numpy(padding_mask)
        return self.tokenizer.detokenize(token_ids * padding_mask)
