# Notes to self

1. How to implement index_add_ op in moe sparse block layer?
2. How to adapt Backbone to use router logits?
3. Weight Conversion!!


Immediate TODOs

1. What about new caching mechanism in HF?


Reference - https://huggingface.co/docs/transformers/en/model_doc/qwen2_moe

Model Architecture:

```
Qwen2MoeForCausalLM(
  (model): Qwen2MoeModel(
    (embed_tokens): Embedding(151936, 2048)
    (layers): ModuleList(
      (0-23): 24 x Qwen2MoeDecoderLayer(
        (self_attn): Qwen2MoeSdpaAttention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (rotary_emb): Qwen2MoeRotaryEmbedding()
        )
        (mlp): Qwen2MoeSparseMoeBlock(
          (gate): Linear(in_features=2048, out_features=60, bias=False)
          (experts): ModuleList(
            (0-59): 60 x Qwen2MoeMLP(
              (gate_proj): Linear(in_features=2048, out_features=1408, bias=False)
              (up_proj): Linear(in_features=2048, out_features=1408, bias=False)
              (down_proj): Linear(in_features=1408, out_features=2048, bias=False)
              (act_fn): SiLU()
            )
          )
          (shared_expert): Qwen2MoeMLP(
            (gate_proj): Linear(in_features=2048, out_features=5632, bias=False)
            (up_proj): Linear(in_features=2048, out_features=5632, bias=False)
            (down_proj): Linear(in_features=5632, out_features=2048, bias=False)
            (act_fn): SiLU()
          )
          (shared_expert_gate): Linear(in_features=2048, out_features=1, bias=False)
        )
        (input_layernorm): Qwen2MoeRMSNorm((2048,), eps=1e-06)
        (post_attention_layernorm): Qwen2MoeRMSNorm((2048,), eps=1e-06)
      )
    )
    (norm): Qwen2MoeRMSNorm((2048,), eps=1e-06)
    (rotary_emb): Qwen2MoeRotaryEmbedding()
  )
  (lm_head): Linear(in_features=2048, out_features=151936, bias=False)
)
```