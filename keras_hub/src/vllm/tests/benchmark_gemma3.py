import kinetic
import os

@kinetic.run(
    accelerator="tpu-v5litepod-1",
    volumes={"/app/keras-hub": kinetic.data.Data(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))}
)
def run_benchmark():
    import os
    import sys
    import time
    import subprocess
    
    # Environment variables
    os.environ["KAGGLE_USERNAME"] = "anthonyedemetim"
    os.environ["KAGGLE_KEY"] = "f3e1ffe1d8befff76b766b0f964586ab"
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
    hf_model_path = kagglehub.model_download("google/gemma-3/transformers/gemma-3-4b-it")
    # hf_model_path = kagglehub.model_download("google/gemma-2/transformers/gemma-2-2b-it")
    
    
    print("Loading vLLM model (in isolated subprocess)...")
        
    # We must run vLLM in a separate subprocess. If PyTorch-XLA (vLLM) and Jax (Keras)
    # share the same process, they will fight for the libtpu driver and segfault.
    import textwrap
    import tempfile
    
    vllm_script = textwrap.dedent(f"""
        import time
        import os
        
        # CRITICAL: Set these before importing vllm so child workers inherit them!
        os.environ["VLLM_TARGET_DEVICE"] = "tpu"
        os.environ["VLLM_USE_V1"] = "0"
        os.environ["JAX_PLATFORMS"] = "tpu,cpu"
        
        from vllm import LLM, SamplingParams
        
        def main():
            prompts = {repr(prompts)}
            max_new_tokens = {max_new_tokens}
            
            llm = LLM(model="{hf_model_path}", tensor_parallel_size=1)
            sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
            
            start_time = time.time()
            outputs = llm.generate(prompts, sampling_params)
            vllm_time = time.time() - start_time
            print(f"VLLM_TIME_TAG:{{vllm_time}}")
            for out in outputs:
                # Replace newlines so they stay on one line for our simple parser
                text = out.outputs[0].text.replace('\\n', ' ')
                print(f"VLLM_OUT_TAG:{{text}}")
        
        if __name__ == "__main__":
            main()
    """)
    
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(vllm_script)
        script_path = f.name
        
    try:
        process = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=True,
            env=os.environ.copy()
        )
        vllm_time = 0.0
        vllm_outputs = []
        for line in process.stdout.split("\n"):
            if "VLLM_TIME_TAG:" in line:
                vllm_time = float(line.split("VLLM_TIME_TAG:")[1])
            elif "VLLM_OUT_TAG:" in line:
                vllm_outputs.append(line.split("VLLM_OUT_TAG:")[1])
            elif line.strip():
                print(line)
        print(f"vLLM Native Inference Time: {vllm_time:.2f} seconds")
        print("vLLM Outputs:")
        for idx, out in enumerate(vllm_outputs):
            print(f"  [{idx+1}] {out}")
    except subprocess.CalledProcessError as e:
        print("vLLM Subprocess failed!")
        print(e.stdout)
        print(e.stderr)
        vllm_time = 0.0
        
    print("\n" + "="*50)
    print("SCENARIO 1: Pure Keras Hub (No vLLM Integration)")
    print("="*50)
    print("Loading Keras Hub model...")
    # Import keras_hub AFTER vllm is done, so Jax doesn't lock the TPU early!
    import keras_hub
    keras_model = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_instruct_4b_text")
    
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
    
    import textwrap
    import tempfile

    integration_script = textwrap.dedent(f"""
        import time
        import os
        
        # CRITICAL: Set these before importing vllm so child workers inherit them!
        os.environ["VLLM_TARGET_DEVICE"] = "tpu"
        os.environ["VLLM_USE_V1"] = "0"
        os.environ["KERAS_BACKEND"] = "jax"
        
        from vllm import LLM, SamplingParams
        from keras_hub.src.vllm.registry import register_keras_hub_models
        from transformers import AutoTokenizer
        
        # Override Tokenizer loading so vLLM uses KerasHub Tokenizer when it loads the local directory
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
                                def get_vocab(self): return {{str(i): i for i in range(self.vocab_size)}}
                                def encode(self, text, **kwargs): return [2, 100, 200, 1]
                                def decode(self, ids, **kwargs): return "mock decoded text"
                            return MockTokenizer()
                    except (OSError, ValueError, KeyError):
                        pass
            return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
            
        AutoTokenizer.from_pretrained = _mock_from_pretrained
        
        def main():
            register_keras_hub_models()
            
            prompts = {repr(prompts)}
            max_new_tokens = {max_new_tokens}
            
            import tempfile, json
            temp_dir = tempfile.mkdtemp()
            config_dict = {{
                "architectures": ["KerasVLLMAdapter"],
                "model_type": "gemma3",
                "vocab_size": 256000,
                "hidden_size": 3072,
                "num_hidden_layers": 40,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 256,
                "max_position_embeddings": 8192,
                "_name_or_path": "keras_hub:gemma3_instruct_4b_text",
                "keras_hub_preset": "gemma3_instruct_4b_text",
                "torch_dtype": "float16",
            }}
            with open(os.path.join(temp_dir, "config.json"), "w") as f:
                json.dump(config_dict, f)
            
            print("Loading Keras Hub model into vLLM...")
            llm = LLM(model=temp_dir, tensor_parallel_size=1)
            sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
            
            start_time = time.time()
            outputs = llm.generate(prompts, sampling_params)
            integration_time = time.time() - start_time
            print(f"INTEGRATION_TIME_TAG:{{integration_time}}")
            for out in outputs:
                text = out.outputs[0].text.replace('\\n', ' ')
                print(f"INTEGRATION_OUT_TAG:{{text}}")
                
        if __name__ == "__main__":
            main()
    """)
    
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(integration_script)
        integration_script_path = f.name
        
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        process = subprocess.run(
            [sys.executable, integration_script_path],
            capture_output=True,
            text=True,
            check=True,
            env=env
        )
        
        integration_time = 0.0
        integration_outputs = []
        for line in process.stdout.splitlines():
            if line.startswith("INTEGRATION_TIME_TAG:"):
                integration_time = float(line.replace("INTEGRATION_TIME_TAG:", ""))
            elif line.startswith("INTEGRATION_OUT_TAG:"):
                integration_outputs.append(line.replace("INTEGRATION_OUT_TAG:", ""))
            else:
                print(line)
                
        print(f"Integration Inference Time: {{integration_time:.2f}} seconds")
        print("Integration Outputs:")
        for idx, out in enumerate(integration_outputs):
            print(f"  [{{idx+1}}] {{out}}")
    except subprocess.CalledProcessError as e:
        print("Integration Subprocess failed!")
        print(e.stdout)
        print(e.stderr)
        integration_time = -1.0

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
