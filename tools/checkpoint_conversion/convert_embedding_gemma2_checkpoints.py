import json
import os
import sys

import numpy as np
import torch
from absl import app
from absl import flags
from huggingface_hub import hf_hub_download
from keras import ops
from PIL import Image
from transformers import AutoModel
from transformers import AutoProcessor

from keras_hub.src.models.embedding_gemma2.embedding_gemma2_text_embedder import (  # noqa: E501
    EmbeddingGemma2TextEmbedder,
)

PRESET_MAP = {
    "embedding_gemma2": "google/embeddinggemma-2",
}

# Parity is measured in true float32. On NVIDIA GPUs cuDNN defaults to TF32
# convolutions, which puts the audio tower ~1e-4 off its CPU result.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset",
    None,
    f"Must be one of {','.join(PRESET_MAP.keys())}",
    required=True,
)


def _keras_embed(keras_model, prep):
    # Inference only, like the HF side below. On the torch backend a plain
    # `model(x)` records autograd state for every layer, which is enough to
    # exhaust a 40 GB GPU on the 8-frame video check.
    with torch.no_grad():
        return ops.convert_to_numpy(keras_model(prep))


def _get_hf_pool(hf_model, hf_inputs):
    with torch.no_grad():
        out = hf_model(**hf_inputs)
        mask = hf_inputs["attention_mask"]
        emb = out[0]
        mask_exp = mask.unsqueeze(-1).expand(emb.size()).float()
        pool = torch.sum(emb * mask_exp, 1) / torch.clamp(
            mask_exp.sum(1), min=1e-9
        )
        return torch.nn.functional.normalize(pool, p=2, dim=1).numpy()


def validate_output(keras_model, hf_model, hf_processor, hf_tokenizer):
    results = []
    failed = False

    def report(name, diff, mean_diff, tol, exact=False):
        nonlocal failed
        if exact:
            is_pass = diff
            status = "PASS" if is_pass else "FAIL"
            print(f"  -> {name:25s} exact match = {status}", flush=True)
            results.append((name, "N/A", "N/A", "exact", status))
        else:
            is_pass = diff < tol if diff is not None else True
            status = "PASS" if is_pass else "FAIL"
            diff_str = f"{diff:.6e}" if diff is not None else "N/A"
            mean_str = f"{mean_diff:.6e}" if mean_diff is not None else "N/A"
            tol_str = f"{tol:.0e}" if tol is not None else "N/A"
            print(
                f"  -> {name:25s} max_abs={diff_str}  mean_abs={mean_str}  "
                f"(tol {tol_str})",
                flush=True,
            )
            results.append((name, diff_str, mean_str, tol_str, status))

        if not is_pass:
            failed = True
        return is_pass

    preprocessor = keras_model.preprocessor

    # (1) Tokenizer ids
    print("\n=== Tokenizer IDs ===", flush=True)
    strings = ["Hello world!", "This is a longer string.", "query: Hello"]
    is_pass = True
    for s in strings:
        prep_dict = preprocessor(s)
        k_ids = ops.convert_to_numpy(prep_dict["token_ids"]).tolist()
        mask = ops.convert_to_numpy(prep_dict["padding_mask"]).tolist()
        if isinstance(mask[0], list):
            k_ids, mask = k_ids[0], mask[0]
        k_ids = [t for t, m in zip(k_ids, mask) if m]

        hf_ids = hf_tokenizer(s)["input_ids"]

        if k_ids != hf_ids:
            print(
                f"Mismatch for {s!r}:\nKeras: {k_ids}\nHF:    {hf_ids}",
                flush=True,
            )
            is_pass = False
    report("Tokenizer_IDs", is_pass, None, None, exact=True)

    # (2) Text embeddings (query, document, + config_sentence_transformers.json)
    print(
        "\n=== Text embeddings (query / document / third prompt) ===",
        flush=True,
    )
    tasks = ["query", "document"]
    st_name = "config_sentence_transformers.json"
    p = os.path.join(PRESET_MAP[FLAGS.preset], st_name)
    if not os.path.exists(p):
        p = hf_hub_download(PRESET_MAP[FLAGS.preset], st_name)
    with open(p) as f:
        config_st = json.load(f)
        prompts = config_st.get("prompts", {})
        for k in prompts:
            if k not in tasks:
                tasks.append(k)
                break

    for task in tasks:
        base_texts = ["Hello world!", "This is a longer test sentence."]
        prompt_str = prompts.get(task, "")
        texts = [prompt_str + t for t in base_texts]

        prep = preprocessor({"texts": texts})
        keras_out = _keras_embed(keras_model, prep)

        hf_inputs = hf_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )
        hf_out = _get_hf_pool(hf_model, hf_inputs)

        diff = float(np.max(np.abs(keras_out - hf_out)))
        mean_diff = float(np.mean(np.abs(keras_out - hf_out)))
        report(f"Text ({task})", diff, mean_diff, 1e-4)

    # (3) Vision
    print("\n=== Vision (encoder-only and end-to-end) ===", flush=True)
    diff, enc_diff = 0.0, 0.0
    mean_diff, enc_mean_diff = 0.0, 0.0
    for shape in [(224, 224, 3), (448, 256, 3)]:
        img = np.random.randint(0, 256, shape, dtype=np.uint8)
        prep = preprocessor({"images": img[None]})
        keras_out = _keras_embed(keras_model, prep)
        hf_inputs = hf_processor(
            images=[[Image.fromarray(img)]], return_tensors="pt"
        )
        hf_out = _get_hf_pool(hf_model, hf_inputs)

        d = float(np.max(np.abs(keras_out - hf_out)))
        m = float(np.mean(np.abs(keras_out - hf_out)))
        if d > diff:
            diff = d
            mean_diff = m

        with torch.no_grad():
            vo = hf_model.vision_tower(
                pixel_values=hf_inputs["pixel_values"],
                pixel_position_ids=hf_inputs["image_position_ids"],
            )
            hf_tokens = hf_model.embed_vision(
                inputs_embeds=vo.last_hidden_state
            ).numpy()
        k_tokens = ops.convert_to_numpy(
            keras_model.backbone.vision_encoder(
                {
                    "pixel_values": hf_inputs["pixel_values"]
                    .numpy()[None]
                    .astype("float32"),
                    "pixel_position_ids": hf_inputs["image_position_ids"]
                    .numpy()[None]
                    .astype("int32"),
                }
            )
        )[0, 0, : hf_tokens.shape[0]]

        d_enc = float(np.max(np.abs(k_tokens - hf_tokens)))
        m_enc = float(np.mean(np.abs(k_tokens - hf_tokens)))
        if d_enc > enc_diff:
            enc_diff = d_enc
            enc_mean_diff = m_enc

    report("Vision_Encoder", enc_diff, enc_mean_diff, 1e-4)
    report("Vision_end_to_end", diff, mean_diff, 1e-3)

    # (4) Audio
    print("\n=== Audio ===", flush=True)
    aud1 = np.random.uniform(-1, 1, (int(0.5 * 16000),))
    aud2 = np.random.uniform(-1, 1, (int(1.3 * 16000),))
    aud1, aud2 = aud1.astype("float32"), aud2.astype("float32")

    prep = preprocessor({"audio": [aud1, aud2]})
    keras_out = _keras_embed(keras_model, prep)

    hf_inputs = hf_processor(
        audio=[aud1, aud2], sampling_rate=16000, return_tensors="pt"
    )
    hf_out = _get_hf_pool(hf_model, hf_inputs)

    diff = float(np.max(np.abs(keras_out - hf_out)))
    mean_diff = float(np.mean(np.abs(keras_out - hf_out)))
    report("Audio", diff, mean_diff, 1e-4)

    # (5) Video
    print("\n=== Video ===", flush=True)
    vid1 = [
        np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        for _ in range(3)
    ]
    vid2 = [
        np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        for _ in range(8)
    ]
    soft = keras_model.backbone.vision_encoder.num_vision_tokens_per_image
    old_seq_len = preprocessor.sequence_length
    preprocessor.sequence_length = 8 * (soft + 2) + 16

    prep = preprocessor({"videos": [vid1, vid2]})
    keras_out = _keras_embed(keras_model, prep)
    preprocessor.sequence_length = old_seq_len

    vid1_pil = [Image.fromarray(f) for f in vid1]
    vid2_pil = [Image.fromarray(f) for f in vid2]
    hf_inputs = hf_processor(videos=[vid1_pil, vid2_pil], return_tensors="pt")
    hf_out = _get_hf_pool(hf_model, hf_inputs)

    diff = float(np.max(np.abs(keras_out - hf_out)))
    mean_diff = float(np.mean(np.abs(keras_out - hf_out)))
    report("Video", diff, mean_diff, 2e-3)

    # (6) Sliding window text
    print("\n=== Sliding-window text ===", flush=True)
    cfg = getattr(hf_model.config, "text_config", hf_model.config)
    sw = getattr(cfg, "sliding_window", 4)
    long_text = "word " * (sw * 2)

    preprocessor.sequence_length = max(2 * sw + 16, 64)
    prep = preprocessor({"texts": [long_text]})
    keras_out = _keras_embed(keras_model, prep)

    hf_inputs = hf_tokenizer([long_text], return_tensors="pt")
    hf_out = _get_hf_pool(hf_model, hf_inputs)

    diff = float(np.max(np.abs(keras_out - hf_out)))
    mean_diff = float(np.mean(np.abs(keras_out - hf_out)))
    report("SlidingWindow", diff, mean_diff, 1e-4)

    # (7) Cosine similarity retrieval
    print("\n=== Cosine-similarity retrieval ===", flush=True)
    q = "How to bake a cake?"
    doc_match = "Here is a recipe for a chocolate cake."
    doc_unrelated = "The capital of France is Paris."

    prep_q = preprocessor({"texts": [q]})
    prep_d_m = preprocessor({"texts": [doc_match]})
    prep_d_u = preprocessor({"texts": [doc_unrelated]})

    keras_q = _keras_embed(keras_model, prep_q)[0]
    keras_d_m = _keras_embed(keras_model, prep_d_m)[0]
    keras_d_u = _keras_embed(keras_model, prep_d_u)[0]

    k_sim_m = float(np.dot(keras_q, keras_d_m))
    k_sim_u = float(np.dot(keras_q, keras_d_u))

    def get_hf_emb(text):
        inp = hf_tokenizer([text], return_tensors="pt")
        return _get_hf_pool(hf_model, inp)[0]

    hf_q = get_hf_emb(q)
    hf_d_m = get_hf_emb(doc_match)
    hf_d_u = get_hf_emb(doc_unrelated)

    h_sim_m = float(np.dot(hf_q, hf_d_m))
    h_sim_u = float(np.dot(hf_q, hf_d_u))

    is_pass = (k_sim_m > k_sim_u) == (h_sim_m > h_sim_u)
    report("Cosine_Similarity", is_pass, None, None, exact=True)

    # (8) Parameter count
    print("\n=== Parameter count ===", flush=True)
    hf_total = sum(v.numel() for v in hf_model.state_dict().values())
    keras_params = keras_model.backbone.count_params()
    print(
        f"  -> HF state_dict total: {hf_total}, Keras params: {keras_params}",
        flush=True,
    )
    report("Parameter_Count", hf_total == keras_params, None, None, exact=True)

    print("\n" + "=" * 80, flush=True)
    print(
        f"{'Name':25s} | {'Max Abs':12s} | {'Mean Abs':12s} | "
        f"{'Tol':8s} | PASS",
        flush=True,
    )
    print("-" * 80, flush=True)
    for name, diff, mean, tol, status in results:
        print(
            f"{name:25s} | {diff:12s} | {mean:12s} | {tol:8s} | {status:4s}",
            flush=True,
        )
    print("=" * 80, flush=True)

    if failed:
        sys.exit(1)
    else:
        print("All validators passed.", flush=True)


def main(_):
    np.random.seed(0)
    preset = FLAGS.preset
    hf_model_name = PRESET_MAP[preset]

    print(f"Loading and converting {hf_model_name}...", flush=True)
    embedder = EmbeddingGemma2TextEmbedder.from_preset(
        f"hf://{hf_model_name}", dtype="float32"
    )

    output_dir = os.path.abspath(f"./{preset}")
    embedder.save_to_preset(output_dir)
    print(f"Preset saved to {output_dir}", flush=True)
    print(sorted(os.listdir(output_dir)), flush=True)

    print("Loading HF model...", flush=True)
    hf_model = AutoModel.from_pretrained(hf_model_name, dtype=torch.float32)
    hf_model.eval()

    hf_processor = AutoProcessor.from_pretrained(hf_model_name)
    hf_tokenizer = hf_processor.tokenizer

    validate_output(embedder, hf_model, hf_processor, hf_tokenizer)


if __name__ == "__main__":
    app.run(main)
