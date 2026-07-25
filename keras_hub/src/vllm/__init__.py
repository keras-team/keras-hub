"""Serve KerasHub models through vLLM on TPU.

`KerasHubLLM("keras_hub:<preset>")` serves a KerasHub `CausalLM` through
tpu-inference's native flax/nnx path. This package provides the serving
context, the paged-attention bridge, the vLLM tokenizer (the preset's own
KerasHub tokenizer, served via `tokenizer_mode="keras_hub"`), and the
`KerasHubLLM` entry point; the serving model class (`KerasHubVllmModel`)
lives in tpu-inference.
"""

from keras_hub.src.vllm.registry import KerasHubLLM
from keras_hub.src.vllm.registry import setup_vllm_model
