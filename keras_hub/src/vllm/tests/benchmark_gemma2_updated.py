import kinetic
import os
import sys
import time
import subprocess
import multiprocessing as mp

def run_vllm_scenario(queue, hf_model_path, prompts, max_new_tokens):
    """Executes the pure vLLM scenario in an isolated process to avoid TPU driver collisions."""
    import os
    import time
    
    # CRITICAL: Set these before importing vllm
    os.environ["VLLM_TARGET_DEVICE"] = "tpu"
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["JAX_PLATFORMS"] = "tpu,cpu"
    
    try:
        from vllm import LLM, SamplingParams
        
        llm = LLM(model=hf_model_path, tensor_parallel_size=1)
        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        vllm_time = time.time() - start_time
        
        vllm_outputs = [out.outputs[0].text.replace('\n', ' ') for out in outputs]
        queue.put((vllm_time, vllm_outputs, None))
    except Exception as e:
        import traceback
        queue.put((0.0, [], traceback.format_exc()))

def run_integration_scenario(queue, prompts, max_new_tokens):
    """Executes the vLLM+KerasHub scenario in an isolated process to avoid TPU driver collisions."""
    import os
    import time
    import tempfile
    import json
    
    # CRITICAL: Set these before importing vllm
    os.environ["VLLM_TARGET_DEVICE"] = "tpu"
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["KERAS_BACKEND"] = "jax"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    try:
        from vllm import LLM, SamplingParams
        from keras_hub.src.vllm.registry import register_keras_hub_models
        from transformers import AutoTokenizer
        
        # Override Tokenizer loading so vLLM uses KerasHub Tokenizer natively
        original_from_pretrained = AutoTokenizer.from_pretrained
        
        @classmethod
        def _mock_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
            import os, json
            if os.path.isdir(pretrained_model_name_or_path):
                config_path = os.path.join(pretrained_model_name_or_path, "config.json")
                if os.path.exists(config_path):
                    try:
                        with open(config_path, "r") as f:
                            cfg = json.load(f)
                        if cfg.get("_name_or_path", "").startswith("keras_hub:"):
                            class MockTokenizer:
                                bos_token_id = 2
                                eos_token_id = 1
                                pad_token_id = 0
                                vocab_size = 256000
                                @property
                                def is_fast(self): return False
                                def __len__(self): return self.vocab_size
                                @property
                                def all_special_tokens(self): return ["<bos>", "<eos>", "<pad>"]
                                @property
                                def all_special_ids(self): return [2, 1, 0]
                                def get_vocab(self): return {str(i): i for i in range(self.vocab_size)}
                                def encode(self, text, **kwargs): return [2, 100, 200, 1]
                                def decode(self, ids, **kwargs): return "mock decoded text"
                            return MockTokenizer()
                    except (OSError, ValueError, KeyError):
                        pass
            return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
            
        AutoTokenizer.from_pretrained = _mock_from_pretrained
        
        register_keras_hub_models()
        
        temp_dir = tempfile.mkdtemp()
        config_dict = {
            "architectures": ["KerasVLLMAdapter"],
            "model_type": "gemma2",
            "vocab_size": 256000,
            "hidden_size": 2304,
            "num_hidden_layers": 26,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "max_position_embeddings": 8192,
            "_name_or_path": "keras_hub:gemma_2b_en",
            "keras_hub_preset": "gemma_2b_en",
            "torch_dtype": "float16",
        }
        with open(os.path.join(temp_dir, "config.json"), "w") as f:
            json.dump(config_dict, f)
        
        print("Loading Keras Hub model into vLLM...")
        llm = LLM(model=temp_dir, tensor_parallel_size=1)
        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        integration_time = time.time() - start_time
        
        integration_outputs = [out.outputs[0].text.replace('\n', ' ') for out in outputs]
        queue.put((integration_time, integration_outputs, None))
    except Exception as e:
        import traceback
        queue.put((0.0, [], traceback.format_exc()))

@kinetic.run(
    accelerator="tpu-v5litepod-1",
    volumes={"/app/keras-hub": kinetic.data.Data(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))}
)
def run_benchmark():
    import os
    import sys
    import time
    import subprocess
    import multiprocessing as mp
    
    # Environment variables
    if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
        raise ValueError("Please set KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
    os.environ["VLLM_TARGET_DEVICE"] = "tpu"
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["JAX_PLATFORMS"] = "tpu,cpu"
    os.environ["KERAS_BACKEND"] = "jax"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.4"

    print("Dynamically installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "/app/keras-hub", "keras", "kagglehub"])
    sys.path.insert(0, "/app/keras-hub")

    import kagglehub

    prompts = [
        "The future of artificial intelligence is",
        "In a distant galaxy, a lone spaceship",
        "The history of the Roman Empire teaches us",
        "To bake the perfect chocolate chip cookie, you must"
    ]
    max_new_tokens = 128
    
    print("\n" + "="*50)
    print("SCENARIO 2: Pure vLLM (Baseline for what the integration should achieve)")
    print("="*50)
    print("Downloading HF format for vLLM...")
    hf_model_path = kagglehub.model_download("google/gemma-2/transformers/gemma-2-2b-it")
    
    print("Loading vLLM model (in isolated subprocess)...")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=run_vllm_scenario, args=(q, hf_model_path, prompts, max_new_tokens))
    p.start()
    
    vllm_time, vllm_outputs, error = q.get()
    p.join()
    
    if error:
        print("vLLM Subprocess failed!")
        print(error)
        vllm_time = -1.0
    else:
        print(f"vLLM Native Inference Time: {vllm_time:.2f} seconds")
        print("vLLM Outputs:")
        for idx, out in enumerate(vllm_outputs):
            print(f"  [{idx+1}] {out}")

    print("\n" + "="*50)
    print("SCENARIO 1: Pure Keras Hub (No vLLM Integration)")
    print("="*50)
    print("Loading Keras Hub model...")
    # Import keras_hub AFTER vllm is done, so Jax doesn't lock the TPU early!
    import keras_hub
    keras_model = keras_hub.models.GemmaCausalLM.from_preset("gemma2_2b_en")
    
    # Warmup
    print("Warming up Keras Hub...")
    keras_model.generate(prompts[0], max_length=32)
    
    # Benchmark
    print("Running Keras Hub Benchmark...")
    start_time = time.time()
    keras_outputs = keras_model.generate(prompts, max_length=max_new_tokens)
    keras_time = time.time() - start_time
    print(f"Keras Hub Native Inference Time: {keras_time:.2f} seconds")
    print("Keras Hub Outputs:")
    for idx, out in enumerate(keras_outputs):
        # Clean up output for display
        text = out.replace('\n', ' ')
        print(f"  [{idx+1}] {text}")

    print("\n" + "="*50)
    print("SCENARIO 3: Keras Hub WITH vLLM Integration")
    print("="*50)
    
    q_int = ctx.Queue()
    p_int = ctx.Process(target=run_integration_scenario, args=(q_int, prompts, max_new_tokens))
    p_int.start()
    
    integration_time, integration_outputs, int_error = q_int.get()
    p_int.join()
    
    if int_error:
        print("Integration Subprocess failed!")
        print(int_error)
        integration_time = -1.0
    else:
        print(f"Integration Inference Time: {integration_time:.2f} seconds")
        print("Integration Outputs:")
        for idx, out in enumerate(integration_outputs):
            print(f"  [{idx+1}] {out}")

    return {
        "keras_hub_time": keras_time,
        "vllm_time": vllm_time,
        "integration_time": integration_time,
    }

if __name__ == "__main__":
    result = run_benchmark()
    print("\nBenchmark Results Summary:")
    print(result)
    
    # Save the results to a txt file locally
    import os
    import json
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "benchmark_results_gemma3.txt")
    
    with open(results_path, "w") as f:
        f.write("Benchmark Results:\n")
        f.write(json.dumps(result, indent=4))
        
    print(f"\nResults successfully saved to {results_path} !")

