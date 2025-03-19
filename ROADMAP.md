# Feature Roadmap for KerasHub

Here's an overview of the features we intend to work on in the near future. Feel free to comment on this issue below to suggest new directions for us to improve the library!

## Models

We are always adding new models to the library. Here's what is currently on our radar.

### NLP models

- **Continue Gemma**: As more models are released in the Gemma family of models, we will bring these to KerasHub on an ongoing basis.
- **DeepSeek R1**: Add DeepSeek R1 to KerasHub [#2077](#)
- **Llama 3.1, 3.2, 3.3**: Add Llama 3.1, 3.2, 3.3 to KerasHub [#2076](#)
- **Qwen 2.5**: Add Qwen 2.5 to KerasHub [#2078](#)
- **ModernBERT**: Add ModernBERT to KerasHub [#2079](#)
- **Mixtral**
- **all-MiniLM**

### RecSys models

- **BERT4Rec**: Add BERT4Rec to KerasHub [#2080](#)
- **SASRec**: Add SASRec to KerasHub [#2081](#)

### Vision models

- **LayoutLmV3**
- **DINOv2**: Add DINOv2 to KerasHub [#2082](#)
- **ControlNet**

### Audio models

- **Whisper**: Add a high-level Whisper speech-to-text task with `generate()` support.
- **Moonshine**: Add Moonshine to KerasHub [#2083](#)

## Feature Improvements

- **Feature extractor task support**
- **Weight file sharding** for large (e.g., 10GB+) models [#2084](#)
- **DoRA**: Add DoRA support in KerasHub [#2072](#)
- **Improved generation**:
  - Move generation functionality to base classes [#1861](#)
  - Directly use the backbone functional graph for `CausalLM.generate()` [#1862](#)
  - Add support for JetStream generative inference for all KerasHub LLMs [#1863](#)
- **Improved quantization support** (including int4 support, QAT, more quantization options)
- **Improved multi-host training support on JAX**:
  - Add auto variable sharding for all backbones/tasks [#1689](#)
  - Guide for multi-host distributed training with KerasHub [#1850](#)
- **Pythonic preprocessing** decoupled from `tf.data`
- **Support RLHF and other instruction fine-tuning options beyond supervised fine-tuning** [#2073](#)
- **High-level API for Whisper Streaming**: Real-time transcription [#2074](#)
- **Reducing inference latency using KVPress** [#2075](#)
- **Speculative decoding**

## Integrations

- **Continue to add conversion support** for Hugging Face Transformers and Timm checkpoints ([See blog post](#))
- **Support JetStream**: Add support for JetStream generative inference for all KerasHub LLMs [#1863](#)
- **Allow native, high-throughput Jax LLM inference on TPUs**
