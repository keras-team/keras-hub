import gc
import io
import os
import tarfile
import time
import threading
import logging
from contextlib import contextmanager

import numpy as np
import requests
import tensorflow as tf
from tqdm import tqdm
from datasets import load_dataset

import keras
import keras_hub
from keras import ops
from keras import losses
from keras.quantizers import GPTQConfig

# ---------------------------
# Logging
# ---------------------------

def setup_logging(level=logging.ERROR):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

setup_logging()

# ---------------------------
# Utilities: humanize bytes
# ---------------------------

def human_bytes(n: int | None) -> str:
    if n is None:
        return "N/A"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} EB"

# ---------------------------
# Dataset helpers
# ---------------------------

def get_dataset_text(dataset_name: str, split: str = "train") -> str:
    """
    Download and return text for small test corpora.
    """
    logging.info("Loading dataset '%s' split='%s'...", dataset_name, split)

    if dataset_name == "wikitext2":
        raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        return "\n\n".join(d["text"] for d in raw_dataset)

    if dataset_name == "ptb":
        url = "https://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
            file_path = f"./simple-examples/data/ptb.{split}.txt"
            text_bytes = tar.extractfile(file_path).read()
        return text_bytes.decode("utf-8")

    raise ValueError(f"Unsupported dataset name for testing: {dataset_name!r}")

def build_token_dataloader(all_tokens: np.ndarray, seq_len: int, max_batches: int = 50) -> np.ndarray:
    """
    Slice a long token stream into [B, T] windows for perplexity evaluation.
    """
    samples = []
    for i in range(max_batches):
        start = i * seq_len
        end = start + seq_len
        if end > len(all_tokens):
            break
        samples.append(np.reshape(all_tokens[start:end], (1, seq_len)))
    if not samples:
        raise ValueError("Not enough tokens to build evaluation batches. "
                         f"Need >= {seq_len}, got {len(all_tokens)}.")
    return np.array(samples, dtype=np.int32)

# ---------------------------
# Instrumentation: CPU/GPU memory + time
# ---------------------------

# psutil is optional; degrade gracefully
try:
    import psutil
    _PSUTIL_OK = True
except Exception:
    _PSUTIL_OK = False
    psutil = None  # type: ignore

def _gpu_devices():
    try:
        return tf.config.list_physical_devices("GPU")
    except Exception:
        return []

def _gpu_mem_supported() -> bool:
    # Requires TF 2.9+ (get_memory_info/reset_memory_stats)
    return (
        hasattr(tf.config.experimental, "get_memory_info") and
        hasattr(tf.config.experimental, "reset_memory_stats")
    )

def gpu_reset_peaks():
    if not _gpu_mem_supported():
        return
    for i, _ in enumerate(_gpu_devices()):
        tf.config.experimental.reset_memory_stats(f"GPU:{i}")

def gpu_peaks() -> dict[int, dict[str, int]]:
    """
    Returns {gpu_index: {'current': bytes, 'peak': bytes}} since last reset.
    """
    out: dict[int, dict[str, int]] = {}
    if not _gpu_mem_supported():
        return out
    for i, _ in enumerate(_gpu_devices()):
        info = tf.config.experimental.get_memory_info(f"GPU:{i}")  # {'current','peak'}
        out[i] = {"current": int(info.get("current", 0)), "peak": int(info.get("peak", 0))}
    return out

class CPUMemSampler:
    """
    Poll process RSS while running to estimate per-window peak main memory.
    """
    def __init__(self, interval_sec: float = 0.05):
        self.interval = interval_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak = 0
        self._proc = psutil.Process(os.getpid()) if _PSUTIL_OK else None

    def start(self):
        if self._proc is None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._proc is None:
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss  # bytes
            if rss > self._peak:
                self._peak = rss
            time.sleep(self.interval)

    @property
    def peak_bytes(self) -> int | None:
        return self._peak if self._proc is not None else None

@contextmanager
def profile_quantization():
    """
    Context manager that captures:
      - wall time
      - CPU RSS peak during window
      - per-GPU current + peak bytes during window

    Usage:
        with profile_quantization() as prof:
            model.quantize("gptq", config=cfg)
        logging.info(prof.summary())
    """
    # CPU sampling
    cpu = CPUMemSampler(interval_sec=0.05)
    cpu.start()

    # GPU: baseline + reset peaks for a clean window
    have_gpu = len(_gpu_devices()) > 0 and _gpu_mem_supported()
    baseline_gpu = gpu_peaks() if have_gpu else {}

    if have_gpu:
        gpu_reset_peaks()

    t0 = time.perf_counter()
    results = {"elapsed_sec": None, "cpu_peak_bytes": None, "gpu_stats": {}, "gpu_baseline": baseline_gpu}

    try:
        yield results
    finally:
        elapsed = time.perf_counter() - t0
        cpu.stop()
        results["elapsed_sec"] = elapsed
        results["cpu_peak_bytes"] = cpu.peak_bytes
        results["gpu_stats"] = gpu_peaks() if have_gpu else {}

def summarize_profile(results: dict) -> str:
    lines = []
    lines.append(f"Quantization time: {results['elapsed_sec']:.3f} s")
    lines.append(f"CPU peak RSS (window): {human_bytes(results['cpu_peak_bytes'])}")
    gpu_stats = results.get("gpu_stats") or {}
    if gpu_stats:
        for i, d in gpu_stats.items():
            cur, peak = d.get("current", 0), d.get("peak", 0)
            lines.append(f"GPU:{i} current: {human_bytes(cur)} | peak (window): {human_bytes(peak)}")
    else:
        if len(_gpu_devices()) == 0:
            lines.append("GPU metrics: no GPU detected")
        elif not _gpu_mem_supported():
            lines.append("GPU metrics: TF build lacks get_memory_info/reset_memory_stats")
    return "\n".join(lines)

def reset_resources():
    """
    Clear Python, Keras, and TF state to avoid memory carry-over
    between benchmarks.
    """
    logging.info("Resetting resources before benchmark...")
    try:
        keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()
    if _gpu_mem_supported():
        gpu_reset_peaks()
    logging.info("Resources reset complete.")

# ---------------------------
# Perplexity evaluation
# ---------------------------

def calculate_perplexity(model, dataloader: np.ndarray) -> float:
    """
    Compute perplexity on a token dataloader: [B, T] int32.
    Compatible with Keras ops; backend-agnostic.
    """
    logging.info("Evaluating perplexity on %d batches...", len(dataloader))
    total_nll = ops.zeros((), dtype="float32")
    total_tokens = ops.zeros((), dtype="float32")

    # Mask uses token id != 1 (commonly EOS in some presets). Adjust if needed.
    for batch in tqdm(dataloader, desc="PPL", leave=False):
        batch = ops.convert_to_tensor(batch, dtype="int32")
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]

        # If model has a preprocessor, pass the structured dict; else raw ids
        if hasattr(model, "preprocessor") and model.preprocessor is not None:
            inputs = {
                "token_ids": input_ids,
                "padding_mask": ops.ones_like(input_ids, dtype="bool"),
            }
        else:
            inputs = input_ids

        outputs = model(inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs

        loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True, reduction=None)
        token_loss = loss_fn(ops.expand_dims(targets, -1), logits)
        mask = ops.cast(ops.not_equal(targets, 1), dtype="float32")
        masked = token_loss * mask

        total_nll = total_nll + ops.sum(masked)
        total_tokens = total_tokens + ops.sum(mask)

    # Guard against zero tokens (e.g., degenerate slicing)
    if float(total_tokens) == 0.0:
        logging.warning("No tokens were evaluated; returning perplexity=inf.")
        return float("inf")

    ppl = ops.exp(total_nll / total_tokens)
    ppl_value = float(ppl)
    logging.info("Perplexity: %.4f", ppl_value)
    return ppl_value

# ---------------------------
# Main test runner
# ---------------------------

def run_quantization_test(
    model_class,
    model_preset: str,
    *,
    dataset_name: str = "wikitext2",
    seq_len: int = 128,
    eval_batches: int = 50,
    calib_samples: int = 128,
):
    """
    Load model + data, evaluate perplexity before/after GPTQ quantization,
    and report time + CPU/GPU memory statistics.

    Args:
        model_class: e.g., keras_hub.models.GPT2CausalLM
        model_preset: model preset string, e.g., "opt_125m_en"
        dataset_name: "wikitext2" or "ptb"
        seq_len: evaluation/calibration sequence length
        eval_batches: max number of [1, T] sequences for perplexity eval
        calib_samples: number of calibration text snippets (sentences)
    """
    if GPTQConfig is None:
        logging.error("GPTQConfig unavailable; cannot run test.")
        return

    reset_resources()

    logging.info("========== GPTQ Quantization Test ==========")
    logging.info("Model preset: %s", model_preset)
    logging.info("Dataset: %s | seq_len=%d | eval_batches=%d | calib_samples=%d",
                 dataset_name, seq_len, eval_batches, calib_samples)

    # 1) Load model + text, tokenize
    try:
        logging.info("Loading model...")
        model = model_class.from_preset(model_preset)

        logging.info("Loading text for eval/calibration...")
        test_text = get_dataset_text(dataset_name, split="test")
        train_text = get_dataset_text(dataset_name, split="train")

        logging.info("Tokenizing test split for eval windows...")
        all_tokens = model.preprocessor.tokenizer.tokenize(test_text)
    except Exception as e:
        logging.exception("Failed during model/data load: %s", e)
        return

    # Build dataloader for perplexity
    test_dataloader = build_token_dataloader(all_tokens, seq_len, max_batches=eval_batches)

    # 2) PPL before quantization
    logging.info("Calculating perplexity BEFORE quantization...")
    pre_ppl = calculate_perplexity(model, test_dataloader)
    logging.info("Pre-quantization perplexity: %.4f", pre_ppl)

    # Calibration dataset: simple sentence split (adjust as needed)
    calibration_dataset = [s.strip() + "." for s in train_text.split(".") if s.strip()][:calib_samples]

    # GPTQ config (adapt fields if your GPTQConfig differs)
    gptq_config = GPTQConfig(
        dataset=calibration_dataset,
        tokenizer=model.preprocessor.tokenizer if hasattr(model, "preprocessor") else None,
        weight_bits=4,
        num_samples=calib_samples,
        sequence_length=seq_len,
        group_size=128,
        hessian_damping=0.01,
        symmetric=False,
        activation_order=False,
    )

    # Optional: pre/post snapshots (RSS + GPU current)
    pre_cpu = psutil.Process(os.getpid()).memory_info().rss if _PSUTIL_OK else None
    pre_gpu = gpu_peaks() if _gpu_mem_supported() else {}

    # 3) Quantize with instrumentation
    logging.info("Quantizing with GPTQ...")
    model.compile(run_eagerly=True)  # Ensure eager for TF ops in quantization
    with profile_quantization() as prof:
        model.quantize("gptq", config=gptq_config)
    logging.info("Quantization complete.")
    logging.info("\n%s", summarize_profile(prof))

    post_cpu = psutil.Process(os.getpid()).memory_info().rss if _PSUTIL_OK else None
    post_gpu = gpu_peaks() if _gpu_mem_supported() else {}

    # Report snapshots
    logging.info(
        "CPU RSS pre/post: %s -> %s (Δ %s)",
        human_bytes(pre_cpu),
        human_bytes(post_cpu),
        human_bytes(None if (pre_cpu is None or post_cpu is None) else max(post_cpu - pre_cpu, 0)),
    )
    if pre_gpu and post_gpu:
        for i in sorted(post_gpu.keys()):
            pre_cur = pre_gpu.get(i, {}).get("current", 0)
            post_cur = post_gpu.get(i, {}).get("current", 0)
            logging.info("GPU:%d current pre/post: %s -> %s", i, human_bytes(pre_cur), human_bytes(post_cur))

    # 4) PPL after quantization
    logging.info("Calculating perplexity AFTER quantization...")
    post_ppl = calculate_perplexity(model, test_dataloader)
    logging.info("Post-quantization perplexity: %.4f", post_ppl)

    logging.info("============== Test Finished ==============\n")

    return {
        "pre_perplexity": pre_ppl,
        "post_perplexity": post_ppl,
        "profile": prof,
        "pre_cpu_bytes": pre_cpu,
        "post_cpu_bytes": post_cpu,
        "pre_gpu_stats": pre_gpu,
        "post_gpu_stats": post_gpu,
    }


#GPT2 with GPTQ 4-bit
run_quantization_test(
    keras_hub.models.GPT2CausalLM,
    "gpt2_base_en_cnn_dailymail"
)