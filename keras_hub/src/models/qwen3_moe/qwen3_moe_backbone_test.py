import gc
import json
import os
import re

import keras
import pytest
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.qwen3_moe.qwen3_moe_backbone import Qwen3MoeBackbone
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.preset_utils import CONFIG_FILE
from keras_hub.src.utils.preset_utils import get_file

# Dims for the Tier-2 CI-safe mesh-shape sweep: representative real-preset
# dimensions, frozen as literals and sourced once, offline (from each
# preset's public HF/Kaggle config.json) -- do not add a `get_file` call to
# the Tier-2 test body itself (that's what Tier 3, `test_layout_map_live_
# presets` below, is for).
#
# MEMORY NOTE: memory-constrained local environments cannot load full-scale
# model dims (see CAPPED_MESH_SHAPES comment below for the mesh-size OOM
# history). What actually matters for the divisibility/sharding properties
# this tier tests is the RATIO of query heads to kv heads and whether
# hidden/intermediate/vocab divide the mesh's model-axis sizes -- not the
# absolute parameter count. So these dims are scaled down by roughly 24x from
# the real presets while preserving each preset's real query:kv head ratio
# (8:1 and 16:1, both GQA) and keeping hidden/intermediate/moe_intermediate/
# vocab as clean values divisible by every mesh shape's model-axis size in
# CAPPED_MESH_SHAPES (2, 4, 8). Full-scale real dims are exercised by
# `test_layout_map_live_presets` below, which has its own per-width-class
# memory-budget skip so it never attempts a full-scale build in a
# memory-constrained environment either -- true full-scale verification
# happens on a dedicated or CI machine with more memory.
QWEN3_MOE_30B_A3B_DIMS = {
    "source_preset": "qwen3_moe_30b_a3b_en (real ratio, memory-scaled dims)",
    "vocabulary_size": 2048,
    "num_layers": 1,  # depth is irrelevant to spec matching/divisibility.
    "num_query_heads": 8,
    "num_key_value_heads": 1,  # real ratio: GQA, 32:4 == 8:1.
    "hidden_dim": 128,
    "intermediate_dim": 256,
    "moe_intermediate_dim": 32,
    "num_experts": 4,
    "top_k": 2,
    "head_dim": 32,
}
QWEN3_MOE_235B_A22B_DIMS = {
    "source_preset": "qwen3_moe_235b_a22b_en (real ratio, memory-scaled dims)",
    "vocabulary_size": 2048,
    "num_layers": 1,
    "num_query_heads": 16,
    "num_key_value_heads": 1,  # real ratio: GQA, 64:4 == 16:1.
    "hidden_dim": 256,
    "intermediate_dim": 512,
    "moe_intermediate_dim": 64,
    "num_experts": 4,
    "top_k": 2,
    "head_dim": 32,
}

# Hard-capped mesh-shape list for memory-constrained local environments. The
# full 10-shape matrix from the testing-strategy doc is
# 2x4, 1x8, 4x4, 8x8, 16x16, 2x2x2, 1x1x8, 2x2x4, 4x4x4, 4x4x8 -- shapes
# 8x8, 16x16, 4x4x4, 4x4x8 (64-256 virtual devices) are DELIBERATELY
# DROPPED here because they exhausted memory and triggered an OOM kill during
# an earlier run of this pipeline in a memory-constrained environment. These
# shapes require a dedicated or CI machine with more memory.
CAPPED_MESH_SHAPES = [
    (2, 4),
    (1, 8),
    (4, 4),
    (2, 2, 2),
    (1, 1, 8),
    (2, 2, 4),
]

# Same expected_shardings patterns as Qwen3MoeBackboneTest.test_distribution
# (post-QKV-axis-fix), reused by the Tier-2 and Tier-3 mesh sweeps below.
# `mlp_only_layers` is left at its default (all-sparse) for these sweeps, so
# only the routed-expert/router rules are exercised here, not the dense-FFN
# fallback -- that fallback path is covered by test_distribution's explicit
# `mlp_only_layers=[1]` config instead.
_EXPECTED_SHARDINGS = {
    "token_embedding/embeddings": ("model", "batch"),
    "token_embedding/reverse_embeddings": ("batch", "model"),
    "self_attention.*query.kernel": ("batch", "model", None),
    "self_attention.*(key|value).kernel": ("batch", None, None),
    "self_attention.*attention_output.kernel": ("model", None, "batch"),
    "experts/expert_feedforward_gate_dense": (None, "batch", "model"),
    "experts/expert_feedforward_output_dense": (None, "model", "batch"),
    "sparse_feedforward_gate_dense/kernel": ("batch", None),
}


def _assert_qwen3_moe_shardings_and_coverage(test_case, model, layout_map):
    """Shared spec + coverage assertions for the Tier-2/3 mesh sweeps."""
    for pattern, expected in _EXPECTED_SHARDINGS.items():
        matches = [w for w in model.weights if re.search(pattern, w.path)]
        test_case.assertGreater(len(matches), 0)
        for w in matches:
            test_case.assertEqual(tuple(w.value.sharding.spec), expected)
    offending = [
        w.path
        for w in model.weights
        if len(w.shape) >= 2 and layout_map[w.path] is None
    ]
    test_case.assertEqual(
        offending,
        [],
        f"The following rank>=2 weights are unmapped: {offending}",
    )


class Qwen3MoeBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 20,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "hidden_dim": 16,
            "intermediate_dim": 32,
            "head_dim": 2,
            "moe_intermediate_dim": 16,
            "num_experts": 4,
            "top_k": 2,
            "norm_top_k_prob": True,
            "decoder_sparse_step": 1,
            "layer_norm_epsilon": 1e-6,
            "rope_max_wavelength": 10000,
            "rope_scaling_factor": 1.0,
            "dropout": 0.0,
            "sliding_window_size": 4096,
            "router_aux_loss_coefficient": 0.01,
            "tie_word_embeddings": False,
            "mlp_only_layers": [],
            "dtype": "float32",  # Explicitly set dtype to avoid mixed precision
        }
        self.input_data = {
            "token_ids": ops.ones((2, 7), dtype="int32"),
            "padding_mask": ops.ones((2, 7), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=Qwen3MoeBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 7, 16),
            run_quantization_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen3MoeBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_architecture_characteristics(self):
        model = Qwen3MoeBackbone(**self.init_kwargs)
        expected_params = 7768
        self.assertEqual(model.count_params(), expected_params)
        expected_layers = 6
        self.assertEqual(len(model.layers), expected_layers)

    def test_auxiliary_loss(self):
        model = Qwen3MoeBackbone(**self.init_kwargs)
        _ = model(self.input_data, training=True)
        self.assertTrue(
            len(model.losses) > 0, "Auxiliary losses should be present"
        )
        for loss in model.losses:
            self.assertGreater(loss, 0.0, "Auxiliary loss should be positive")

    @pytest.mark.multi_device
    def test_distribution(self):
        # Note: the shared helper pins the mesh to exactly 2 devices (not
        # len(devices)), so the default test config's num_key_value_heads=2
        # exercises the ordinary GQA-replication path deterministically
        # regardless of how many virtual devices the test environment
        # exposes. `mlp_only_layers=[1]` makes layer 1 use the dense FFN
        # fallback (`qwen3_moe_mlp`) while layer 0 stays sparse, so both the
        # dense-fallback and routed-expert layout rules are exercised in one
        # call. This model defaults to `is_moe=True` tolerances since
        # `num_experts > 0` is always active (unlike qwen_moe/mixtral, there
        # is no dense-only mode for this backbone).
        init_kwargs = dict(self.init_kwargs, mlp_only_layers=[1])
        self.run_distribution_test(
            cls=Qwen3MoeBackbone,
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
                "qwen3_moe_mlp.*feedforward_intermediate_dense.kernel": (
                    "batch",
                    "model",
                ),
                "qwen3_moe_mlp.*feedforward_gate_dense.kernel": (
                    "batch",
                    "model",
                ),
                "qwen3_moe_mlp.*feedforward_output_dense.kernel": (
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
            },
            allow_replicated=(),
            is_moe=True,
        )

    @parameterized.named_parameters(
        (
            f"{dims['source_preset'].split(' ')[0]}_mesh"
            f"_{'x'.join(str(s) for s in shape)}",
            dims,
            shape,
        )
        for dims in (QWEN3_MOE_30B_A3B_DIMS, QWEN3_MOE_235B_A22B_DIMS)
        for shape in CAPPED_MESH_SHAPES
    )
    @pytest.mark.multi_device
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
        layout_map = Qwen3MoeBackbone.get_layout_map(device_mesh)
        distribution = keras.distribution.ModelParallel(
            layout_map=layout_map, batch_dim_name="batch"
        )
        init_kwargs = {k: v for k, v in dims.items() if k != "source_preset"}
        with distribution.scope():
            # bfloat16: a memory mitigation for memory-constrained
            # environments -- spec assertions are dtype-independent.
            model = Qwen3MoeBackbone(dtype="bfloat16", **init_kwargs)
            _assert_qwen3_moe_shardings_and_coverage(self, model, layout_map)
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
        # are only built once per mesh shape -- a memory/time necessity in
        # memory-constrained environments, while every preset in the registry
        # is still fetched and evaluated, preserving full registry coverage.
        dim_keys = (
            "vocabulary_size",
            "num_query_heads",
            "num_key_value_heads",
            "hidden_dim",
            "intermediate_dim",
            "moe_intermediate_dim",
            "num_experts",
            "top_k",
            "head_dim",
        )
        width_classes = {}  # dedupe key -> (config dict, [preset names])
        fetch_failures = []
        for preset in Qwen3MoeBackbone.presets:
            try:
                path = get_file(preset, CONFIG_FILE)
                with open(path) as f:
                    cfg = json.load(f)["config"]
            except Exception as e:
                # A preset this account can't reach (e.g. an unaccepted
                # Kaggle license consent click-through) is logged, not
                # fatal -- the rest of the registry still gets exercised.
                fetch_failures.append((preset, str(e)))
                continue
            # num_layers is forced to 1 below regardless of the real
            # value -- layout rules are per-decoder-block regexes, so
            # depth is irrelevant to spec matching/divisibility, and 1
            # layer keeps build memory bounded.
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
                "qwen3_moe family. See the module comment above "
                "CAPPED_MESH_SHAPES."
            )
        print(
            f"test_layout_map_live_presets: {len(width_classes)} unique "
            f"width-classes across {len(Qwen3MoeBackbone.presets)} "
            "registry presets:"
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

                    # Memory-budget guard: memory-constrained environments
                    # cannot build full-scale presets. Estimate this
                    # width-class's single-decoder-block bf16 footprint.
                    # Qwen3-MoE has UNTIED embeddings (tie_word_embeddings
                    # defaults False and real untied presets exist), so the
                    # embedding term is counted twice (forward + reverse
                    # output projection). Attention projections map
                    # hidden<->hidden for query/output and hidden<->(hidden
                    # scaled by the kv/query head ratio) for GQA's key/value;
                    # none touch intermediate_dim (attention has no bias in
                    # this model). The parameter bulk is the MoE expert bank:
                    # gate_up is (num_experts, hidden, 2*moe_intermediate)
                    # and down is (num_experts, moe_intermediate, hidden), so
                    # the estimate includes an explicit num_experts term over
                    # moe_intermediate_dim (the expert width, not the dense
                    # intermediate_dim) -- omitting it would dangerously
                    # undercount a MoE model's real footprint. Times a 3x
                    # safety margin for JAX/XLA transient copies during
                    # construction/resharding. The config-fetch, dedup, and
                    # divisibility-skip logic above still exercises every
                    # registry preset either way; only the expensive
                    # build+assert step is capped.
                    hidden = cfg["hidden_dim"]
                    moe_inter = cfg["moe_intermediate_dim"]
                    num_experts = cfg["num_experts"]
                    num_kv_heads = cfg["num_key_value_heads"]
                    kv_ratio = num_kv_heads / num_query_heads
                    est_params = (
                        2 * cfg["vocabulary_size"] * hidden  # untied embed
                        + 2 * hidden * hidden  # attention q/o
                        + 2 * hidden * hidden * kv_ratio  # attention k/v
                        + num_experts
                        * (hidden * 2 * moe_inter + moe_inter * hidden)
                    )
                    est_bytes = est_params * 2 * 3  # bf16 * safety margin
                    max_local_bytes = int(
                        os.environ.get(
                            "KERAS_HUB_DISTRIBUTION_TEST_MEM_BUDGET",
                            300 * 1024 * 1024,
                        )
                    )
                    if est_bytes > max_local_bytes:
                        reason = (
                            f"{combo_label}: estimated build memory "
                            f"~{est_bytes / 1e9:.2f}GB exceeds the "
                            f"{max_local_bytes / 1e6:.0f}MB local safety "
                            "threshold for memory-constrained environments "
                            "-- verify this width-class on a machine with "
                            "more memory or in CI (raise the budget via the "
                            "KERAS_HUB_DISTRIBUTION_TEST_MEM_BUDGET env var)"
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
                    layout_map = Qwen3MoeBackbone.get_layout_map(device_mesh)
                    distribution = keras.distribution.ModelParallel(
                        layout_map=layout_map, batch_dim_name="batch"
                    )
                    # `cfg` is the preset's full serialized `get_config()`
                    # dict; pass all of it (real architecture flags such as
                    # rope scaling, MoE routing, tie-embeddings, etc. must
                    # survive to actually exercise the live preset's real
                    # architecture) and only drop `"dtype"` so the explicit
                    # `dtype="bfloat16"` override below doesn't collide with a
                    # duplicate keyword argument. `num_layers` is forced to 1
                    # (layout rules are per-decoder-block, so depth is
                    # irrelevant and 1 layer keeps build memory bounded).
                    init_kwargs = {k: v for k, v in cfg.items() if k != "dtype"}
                    init_kwargs["num_layers"] = 1
                    with distribution.scope():
                        model = Qwen3MoeBackbone(
                            dtype="bfloat16", **init_kwargs
                        )
                        _assert_qwen3_moe_shardings_and_coverage(
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
