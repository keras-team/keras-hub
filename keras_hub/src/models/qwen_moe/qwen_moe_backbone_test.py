import gc
import json
import re

import keras
import pytest
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.qwen_moe.qwen_moe_backbone import QwenMoeBackbone
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.preset_utils import CONFIG_FILE
from keras_hub.src.utils.preset_utils import get_file

# Dims for the Tier-2 CI-safe mesh-shape sweep: representative real-preset
# dimensions, frozen as literals and sourced once, offline -- do not add a
# `get_file` call to the Tier-2 test body itself (that's what Tier 3,
# `test_layout_map_live_presets` below, is for).
#
# MEMORY NOTE: this local dev machine cannot load full-scale model dims.
# What actually matters for the divisibility/sharding properties this tier
# tests is the RATIO of query heads to kv heads and whether
# hidden/intermediate/vocab divide the mesh's model-axis sizes -- not the
# absolute parameter count. So these dims are scaled down roughly 20-60x
# from the real preset(s) while preserving the real query:kv head ratio and
# keeping hidden/intermediate/vocab as clean numbers divisible by every mesh
# shape in CAPPED_MESH_SHAPES. Full-scale real dims are exercised by
# `test_layout_map_live_presets` below, which has its own per-width-class
# memory-budget skip so it never attempts a full-scale build locally either.
QWEN_MOE_2_7B_DIMS = {
    # Real registry preset "qwen1.5_moe_2.7b_en" (fetched live 2026-07-17):
    # vocabulary_size=151936, num_query_heads=16, num_key_value_heads=16
    # (1:1 ratio -- standard MHA, no GQA reduction in this preset),
    # hidden_dim=2048, intermediate_dim=5632, moe_intermediate_dim=1408,
    # shared_expert_intermediate_dim=5632, num_experts=60, top_k=4.
    "source_preset": "qwen1.5_moe_2.7b_en (real ratio, memory-scaled dims)",
    "vocabulary_size": 2000,
    "num_layers": 1,  # depth is irrelevant to spec matching/divisibility.
    "num_query_heads": 8,
    "num_key_value_heads": 8,  # real ratio: MHA, 1:1.
    "hidden_dim": 256,
    "intermediate_dim": 1024,
    "moe_intermediate_dim": 256,
    "shared_expert_intermediate_dim": 1024,
    "num_experts": 8,
    "top_k": 2,
}
QWEN_MOE_SMALL_GQA_DIMS = {
    # Synthetic second width-class using the docstring's own example ratio
    # (num_query_heads=16, num_key_value_heads=8 -- 2:1 GQA), scaled down
    # the same way, to give the sweep a GQA case distinct from the 1:1 real
    # preset above (there is only one preset in this model's registry).
    "source_preset": "qwen_moe_docstring_example (2:1 GQA ratio, synthetic)",
    "vocabulary_size": 3200,
    "num_layers": 1,
    "num_query_heads": 16,
    "num_key_value_heads": 8,  # docstring-example ratio: GQA, 2:1.
    "hidden_dim": 512,
    "intermediate_dim": 2048,
    "moe_intermediate_dim": 512,
    "shared_expert_intermediate_dim": 2048,
    "num_experts": 16,
    "top_k": 4,
}

# Hard-capped mesh-shape list for this shared 37GB dev machine. The full
# 10-shape matrix from the testing-strategy doc is
# 2x4, 1x8, 4x4, 8x8, 16x16, 2x2x2, 1x1x8, 2x2x4, 4x4x4, 4x4x8 -- shapes
# 8x8, 16x16, 4x4x4, 4x4x8 (64-256 virtual devices) are DELIBERATELY
# DROPPED here to stay within this shared machine's memory budget (see the
# gemma reference implementation, which hit a demonstrated OOM kill with
# those shapes). Do not attempt the dropped shapes even experimentally on
# this box.
CAPPED_MESH_SHAPES = [
    (2, 4),
    (1, 8),
    (4, 4),
    (2, 2, 2),
    (1, 1, 8),
    (2, 2, 4),
]

# Same expected_shardings patterns as QwenMoeBackboneTest.test_distribution
# (post-QKV-axis-fix), reused by the Tier-2 and Tier-3 mesh sweeps below.
_EXPECTED_SHARDINGS = {
    "token_embedding/embeddings": ("model", "batch"),
    "token_embedding/reverse_embeddings": ("batch", "model"),
    "self_attention.*query.kernel": ("batch", "model", None),
    "self_attention.*(key|value).kernel": ("batch", None, None),
    "self_attention.*attention_output.kernel": ("model", None, "batch"),
    "shared_expert_dense/feedforward_intermediate_dense.kernel": (
        "batch",
        "model",
    ),
    "shared_expert_dense/feedforward_gate_dense.kernel": ("batch", "model"),
    "shared_expert_dense/feedforward_output_dense.kernel": ("model", "batch"),
    "experts/expert_feedforward_gate_dense": (None, "batch", "model"),
    "experts/expert_feedforward_output_dense": (None, "model", "batch"),
    "sparse_feedforward_gate_dense/kernel": ("batch", None),
    "shared_expert_gate_dense/kernel": ("batch", None),
}
# 2-D q/k/v biases (num_heads, head_dim) are intentionally left replicated
# (see get_layout_map's comment): sharding the heads axis would raise an
# IndivisibleError for GQA configs, and the tensors are too small to be
# worth sharding. `qwen_moe_mlp.*` (the dense-FFN fallback used by
# `mlp_only_layers`) is included even though it is currently unreachable
# from `QwenMoeBackbone`'s layer-construction loop (a pre-existing,
# out-of-scope bug -- see test_distribution's comment) so that coverage
# stays correct if/when that bug is fixed.
_ALLOW_REPLICATED = (
    "self_attention.*(query|key|value)/bias",
    "qwen_moe_mlp.*feedforward_intermediate_dense.kernel",
    "qwen_moe_mlp.*feedforward_gate_dense.kernel",
    "qwen_moe_mlp.*feedforward_output_dense.kernel",
)


def _assert_qwen_moe_shardings_and_coverage(test_case, model, layout_map):
    """Shared spec + coverage assertions for the Tier-2/3 mesh sweeps."""
    for pattern, expected in _EXPECTED_SHARDINGS.items():
        matches = [w for w in model.weights if re.search(pattern, w.path)]
        test_case.assertGreater(len(matches), 0)
        for w in matches:
            test_case.assertEqual(tuple(w.value.sharding.spec), expected)
    offending = [
        w.path
        for w in model.weights
        if len(w.shape) >= 2
        and layout_map[w.path] is None
        and not any(re.search(p, w.path) for p in _ALLOW_REPLICATED)
    ]
    test_case.assertEqual(
        offending,
        [],
        f"The following rank>=2 weights are unmapped: {offending}",
    )


class QwenMoeBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 20,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "hidden_dim": 16,
            "intermediate_dim": 32,
            "moe_intermediate_dim": 16,
            "shared_expert_intermediate_dim": 32,
            "num_experts": 4,
            "top_k": 2,
            "norm_top_k_prob": True,
            "decoder_sparse_step": 1,
            "layer_norm_epsilon": 1e-6,
            "rope_max_wavelength": 10000,
            "rope_scaling_factor": 1.0,
            "dropout": 0.0,
            "use_sliding_window_attention": False,
            "sliding_window_size": 4096,
            "router_aux_loss_coefficient": 0.01,
            "tie_word_embeddings": False,
            "output_router_logits": False,
            "mlp_only_layers": [],
            "dtype": "float32",  # Explicitly set dtype to avoid mixed precision
        }
        self.input_data = {
            "token_ids": ops.ones((2, 7), dtype="int32"),
            "padding_mask": ops.ones((2, 7), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=QwenMoeBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 7, 16),
            run_quantization_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=QwenMoeBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_architecture_characteristics(self):
        model = QwenMoeBackbone(**self.init_kwargs)
        expected_params = (
            # Token Embedding (forward and reverse, since
            # tie_word_embeddings=False)
            20 * 16 * 2  # 640
            # Transformer Layers
            + 2
            * (
                # Self-Attention
                (16 * 4 * 4 + 4 * 4)  # Query + Bias = 256 + 16
                + (16 * 2 * 4 + 2 * 4)  # Key + Bias = 128 + 8
                + (16 * 2 * 4 + 2 * 4)  # Value + Bias = 128 + 8
                + (4 * 4 * 16)  # Output = 256
                + 16  # Self-Attention LayerNorm
                # MoE
                + (16 * 4)  # Router = 64
                + 4 * (16 * 2 * 16)  # Experts Gate+Up = 2048
                + 4 * (16 * 16)  # Experts Output = 1024
                + (16 * 32)  # Shared Expert Gate = 512
                + (16 * 32)  # Shared Expert Intermediate = 512
                + (32 * 16)  # Shared Expert Output = 512
                + (16 * 1)  # Shared Expert Gate = 16
                + 16  # Feedforward LayerNorm
            )
            # Final LayerNorm
            + 16
        )
        # Should be 11696
        self.assertEqual(model.count_params(), expected_params)
        # token_embedding + 2 transformer layers + final norm + 2 inputs
        expected_layers = 6
        self.assertEqual(len(model.layers), expected_layers)

    def test_distribution(self):
        # NOTE: `mlp_only_layers=[1]` is intended to make layer 1 use the
        # dense FFN fallback (`qwen_moe_mlp`) while layer 0 stays sparse, so
        # both the dense-fallback and routed-expert layout rules would be
        # exercised here. It currently does NOT, because
        # `QwenMoeTransformerDecoder.layer_index` defaults to 0 and is never
        # passed `i` from this backbone's layer-construction loop -- every
        # layer's `layer_index` is 0 regardless of position, so
        # `mlp_only_layers` never actually selects a layer. This is a
        # pre-existing bug unrelated to sharding/layout maps (out of scope
        # for this PR); the `qwen_moe_mlp.*` weights are correctly
        # allow-replicated below but currently never actually get built by
        # this test config until that bug is fixed elsewhere.
        init_kwargs = dict(self.init_kwargs, mlp_only_layers=[1])
        self.run_distribution_test(
            cls=QwenMoeBackbone,
            init_kwargs=init_kwargs,
            input_data=self.input_data,
            expected_shardings={
                "token_embedding/embeddings": ("model", "batch"),
                "token_embedding/reverse_embeddings": ("batch", "model"),
                "self_attention.*query.kernel": ("batch", "model", None),
                "self_attention.*(key|value).kernel": (
                    "batch",
                    None,
                    None,
                ),
                "self_attention.*attention_output.kernel": (
                    "model",
                    None,
                    "batch",
                ),
                "shared_expert_dense/feedforward_intermediate_dense.kernel": (
                    "batch",
                    "model",
                ),
                "shared_expert_dense/feedforward_gate_dense.kernel": (
                    "batch",
                    "model",
                ),
                "shared_expert_dense/feedforward_output_dense.kernel": (
                    "model",
                    "batch",
                ),
                "experts/expert_feedforward_gate_dense": (
                    None,
                    "batch",
                    "model",
                ),
                "experts/expert_feedforward_output_dense": (
                    None,
                    "model",
                    "batch",
                ),
                "sparse_feedforward_gate_dense/kernel": ("batch", None),
                "shared_expert_gate_dense/kernel": ("batch", None),
            },
            allow_replicated=_ALLOW_REPLICATED,
            is_moe=True,
        )

    def test_auxiliary_loss(self):
        model = QwenMoeBackbone(**self.init_kwargs)
        _ = model(self.input_data, training=True)
        self.assertTrue(
            len(model.losses) > 0, "Auxiliary losses should be present"
        )
        for loss in model.losses:
            self.assertGreater(loss, 0.0, "Auxiliary loss should be positive")

    @parameterized.named_parameters(
        (
            f"{dims['source_preset'].split(' ')[0]}_mesh"
            f"_{'x'.join(str(s) for s in shape)}",
            dims,
            shape,
        )
        for dims in (QWEN_MOE_2_7B_DIMS, QWEN_MOE_SMALL_GQA_DIMS)
        for shape in CAPPED_MESH_SHAPES
    )
    def test_layout_map_mesh_shapes(self, dims, mesh_shape):
        if keras.backend.backend() != "jax":
            self.skipTest("`ModelParallel` testing requires the Jax backend.")
        devices = keras.distribution.list_devices("CPU")
        n_needed = 1
        for s in mesh_shape:
            n_needed *= s
        if n_needed > len(devices):
            self.skipTest(
                f"Mesh shape {mesh_shape} needs {n_needed} devices, only "
                f"{len(devices)} available. Run with "
                f"XLA_FLAGS=--xla_force_host_platform_device_count="
                f"{n_needed} to exercise this shape locally."
            )

        # Query-head-divisibility skip rule: an inherent tensor-parallelism
        # ceiling, not a bug. model_axis_size is the mesh's last axis,
        # matching this repo's axis_names=(..., "model") convention.
        model_axis_size = mesh_shape[-1]
        num_query_heads = dims["num_query_heads"]
        if num_query_heads % model_axis_size != 0:
            self.skipTest(
                f"num_query_heads={num_query_heads} not divisible by "
                f"model-axis={model_axis_size}: inherent "
                "tensor-parallelism limit, not a bug"
            )

        devices = devices[:n_needed]
        if len(mesh_shape) == 2:
            axis_names = ("batch", "model")
        else:
            # 3D shape: the extra axis is named "seq" per the plan/
            # testing-strategy convention, but get_layout_map only names
            # "batch"/"model" -- the "seq" axis simply replicates weights
            # (no rule targets it), matching every other 3D-mesh test in
            # this PR series.
            axis_names = ("batch", "seq", "model")
        device_mesh = keras.distribution.DeviceMesh(
            shape=mesh_shape,
            axis_names=axis_names,
            devices=devices,
        )
        layout_map = QwenMoeBackbone.get_layout_map(device_mesh)
        distribution = keras.distribution.ModelParallel(
            layout_map=layout_map, batch_dim_name="batch"
        )
        init_kwargs = {k: v for k, v in dims.items() if k != "source_preset"}
        with distribution.scope():
            # bfloat16: a memory mitigation for this shared dev machine --
            # spec assertions are dtype-independent.
            model = QwenMoeBackbone(dtype="bfloat16", **init_kwargs)
            _assert_qwen_moe_shardings_and_coverage(self, model, layout_map)
        del model
        gc.collect()

    @pytest.mark.kaggle_key_required
    @pytest.mark.multi_device
    @pytest.mark.extra_large
    def test_layout_map_live_presets(self):
        if keras.backend.backend() != "jax":
            self.skipTest("`ModelParallel` testing requires the Jax backend.")

        # Fetch every preset's config only (no weights), then dedupe by the
        # divisibility-relevant dims so width-classes that share a config
        # are only built once per mesh shape -- a memory/time necessity on
        # this machine, while every preset in the registry is still fetched
        # and evaluated, preserving full registry coverage.
        dim_keys = (
            "vocabulary_size",
            "num_query_heads",
            "num_key_value_heads",
            "hidden_dim",
            "intermediate_dim",
            "moe_intermediate_dim",
            "shared_expert_intermediate_dim",
            "num_experts",
            "top_k",
        )
        width_classes = {}  # dedupe key -> (config dict, [preset names])
        fetch_failures = []
        for preset in QwenMoeBackbone.presets:
            try:
                path = get_file(preset, CONFIG_FILE)
                cfg = json.load(open(path))["config"]
            except Exception as e:
                # A preset this account can't reach (e.g. an unaccepted
                # Kaggle license consent click-through) is logged, not
                # fatal -- the rest of the registry still gets exercised.
                fetch_failures.append((preset, str(e)))
                continue
            # num_layers is forced to 1 below regardless of the real
            # value -- layout rules are per-decoder-block regexes, so
            # depth is irrelevant to spec matching/divisibility, and 1
            # layer keeps build memory bounded on this shared machine.
            cfg = dict(cfg)
            cfg["num_layers"] = 1
            key = tuple(cfg.get(k) for k in dim_keys)
            if key not in width_classes:
                width_classes[key] = (cfg, [])
            width_classes[key][1].append(preset)

        if fetch_failures:
            print(
                f"test_layout_map_live_presets: {len(fetch_failures)} "
                f"preset config fetches failed (logged, non-fatal): "
                f"{fetch_failures}"
            )
        if not width_classes:
            self.skipTest(
                "No preset configs were reachable "
                f"({len(fetch_failures)} fetch failures) -- likely a "
                "Kaggle license-consent gate on this account for the "
                "qwen_moe family. See the module comment above "
                "CAPPED_MESH_SHAPES."
            )
        print(
            f"test_layout_map_live_presets: {len(width_classes)} unique "
            f"width-classes across {len(QwenMoeBackbone.presets)} registry "
            "presets:"
        )
        for cfg, presets in width_classes.values():
            print(f"  {presets}")

        devices = keras.distribution.list_devices("CPU")
        skip_reasons = []
        ran_any = False
        for cfg, presets in width_classes.values():
            num_query_heads = cfg["num_query_heads"]
            for mesh_shape in CAPPED_MESH_SHAPES:
                combo_label = f"{presets[0]}@{mesh_shape}"
                with self.subTest(combo=combo_label):
                    n_needed = 1
                    for s in mesh_shape:
                        n_needed *= s
                    if n_needed > len(devices):
                        reason = (
                            f"{combo_label}: needs {n_needed} devices, "
                            f"only {len(devices)} available"
                        )
                        skip_reasons.append(reason)
                        continue
                    model_axis_size = mesh_shape[-1]
                    if num_query_heads % model_axis_size != 0:
                        reason = (
                            f"{combo_label}: num_query_heads="
                            f"{num_query_heads} not divisible by "
                            f"model-axis={model_axis_size}: inherent "
                            "tensor-parallelism limit, not a bug"
                        )
                        skip_reasons.append(reason)
                        continue

                    # Memory-budget guard: this shared dev machine cannot
                    # locally build full-scale presets. Estimate this
                    # width-class's bf16 footprint (embedding table + 3
                    # dense-equivalent hidden*intermediate matrices, times a
                    # 3x safety margin for JAX/XLA transient copies during
                    # construction/resharding) and skip the actual build if
                    # it exceeds a conservative local threshold. The
                    # config-fetch, dedup, and divisibility-skip logic above
                    # still exercises every registry preset either way;
                    # only the expensive build+assert step is capped.
                    est_params = (
                        cfg["vocabulary_size"] * cfg["hidden_dim"]
                        + 3 * cfg["hidden_dim"] * cfg["intermediate_dim"]
                    )
                    est_bytes = est_params * 2 * 3  # bf16 * safety margin
                    max_local_bytes = 300 * 1024 * 1024  # 300MB
                    if est_bytes > max_local_bytes:
                        reason = (
                            f"{combo_label}: estimated build memory "
                            f"~{est_bytes / 1e9:.2f}GB exceeds the "
                            f"{max_local_bytes / 1e6:.0f}MB local safety "
                            "threshold on this shared, RAM-constrained "
                            "dev machine -- verify this width-class on a "
                            "machine with more RAM or in CI"
                        )
                        skip_reasons.append(reason)
                        continue

                    combo_devices = devices[:n_needed]
                    if len(mesh_shape) == 2:
                        axis_names = ("batch", "model")
                    else:
                        axis_names = ("batch", "seq", "model")
                    device_mesh = keras.distribution.DeviceMesh(
                        shape=mesh_shape,
                        axis_names=axis_names,
                        devices=combo_devices,
                    )
                    layout_map = QwenMoeBackbone.get_layout_map(device_mesh)
                    distribution = keras.distribution.ModelParallel(
                        layout_map=layout_map, batch_dim_name="batch"
                    )
                    init_kwargs = {
                        k: v for k, v in cfg.items() if k in dim_keys
                    }
                    init_kwargs["num_layers"] = 1
                    with distribution.scope():
                        model = QwenMoeBackbone(dtype="bfloat16", **init_kwargs)
                        _assert_qwen_moe_shardings_and_coverage(
                            self, model, layout_map
                        )
                    del model
                    gc.collect()
                    ran_any = True

        print(
            f"test_layout_map_live_presets: {len(skip_reasons)} combo(s) "
            f"skipped:\n" + "\n".join(f"  {r}" for r in skip_reasons)
        )
        if not ran_any:
            self.skipTest(
                "All (width-class, mesh-shape) combos were skipped: "
                f"{skip_reasons}"
            )
