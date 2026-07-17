import gc
import re

import keras
import pytest
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.qwen3_5_moe.qwen3_5_moe_backbone import (
    Qwen3_5MoeBackbone,
)
from keras_hub.src.tests.test_case import TestCase

# Dims for the Tier-2 CI-safe mesh-shape sweep: representative
# Qwen3.5-MoE-35B-A3B-class shapes, frozen as literals and scaled down for
# this shared, RAM-constrained dev box -- do NOT add a live `get_file` call to
# the Tier-2 test body (that is what Tier 3, `test_layout_map_live_presets`,
# is for).
#
# MEMORY NOTE: this local dev machine cannot load full-scale model dims, so
# every dimension (vocab, hidden, intermediate, expert widths, head dims) is
# scaled down ~20-30x from the real `Qwen/Qwen3.5-35B-A3B` config while
# preserving the two properties this tier actually tests:
#   * the real query:kv head RATIO (8:1 GQA) and the real
#     linear key:value head ratio (16:32 = 1:2), which drive the
#     divisibility/sharding behaviour under test, and
#   * clean divisibility of every model-axis-sharded dimension (vocab,
#     hidden, num_query_heads, moe_intermediate, shared-expert intermediate,
#     and the linear value/qkv projection widths) by every mesh model-axis
#     size in CAPPED_MESH_SHAPES (2, 4, 8).
# num_layers is 1 always (layout rules are per-decoder-block regexes, so depth
# is irrelevant to spec matching/divisibility) and the single layer is a
# `full_attention` layer so the sweep exercises the corrected QKV-axis
# divisibility -- the core of this fix -- alongside the always-present MoE
# expert/router/shared-expert rules. The linear-attention projection rules are
# exercised by the four-layer `test_distribution` config above, which runs a
# real forward+backward pass over both layer types. Full-scale real dims are
# exercised offline / by `test_layout_map_live_presets` (which has its own
# per-width-class memory-budget skip so it never attempts a full-scale build
# locally either).
QWEN3_5_MOE_35B_A3B_DIMS = {
    "source_preset": "qwen3_5_moe_35b_a3b (real ratios, memory-scaled dims)",
    "vocabulary_size": 2048,
    "num_layers": 1,
    "num_query_heads": 32,
    "num_key_value_heads": 4,  # real ratio: GQA, 8:1.
    "head_dim": 32,
    "hidden_dim": 512,
    "moe_intermediate_dim": 64,
    "shared_expert_intermediate_size": 64,
    "num_experts": 8,
    "top_k": 2,
    "layer_types": ["full_attention"],
    "partial_rotary_factor": 0.25,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,  # real ratio: 16:32 = 1:2.
    "linear_key_head_dim": 8,
    "linear_value_head_dim": 8,
    "linear_conv_kernel_dim": 4,
    "router_aux_loss_coefficient": 0.001,
}
# A second, smaller width-class with the same real head-count ratios, to
# exercise a distinct set of divisible dimensions through the same sweep.
QWEN3_5_MOE_SMALL_DIMS = {
    "source_preset": "qwen3_5_moe_small (real ratios, memory-scaled dims)",
    "vocabulary_size": 2048,
    "num_layers": 1,
    "num_query_heads": 16,
    "num_key_value_heads": 2,  # real ratio: GQA, 8:1.
    "head_dim": 32,
    "hidden_dim": 256,
    "moe_intermediate_dim": 32,
    "shared_expert_intermediate_size": 32,
    "num_experts": 8,
    "top_k": 2,
    "layer_types": ["full_attention"],
    "partial_rotary_factor": 0.25,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,  # real ratio: 16:32 = 1:2.
    "linear_key_head_dim": 8,
    "linear_value_head_dim": 8,
    "linear_conv_kernel_dim": 4,
    "router_aux_loss_coefficient": 0.001,
}

# Hard-capped mesh-shape list for this shared 37GB dev machine. The full
# 10-shape matrix from the testing-strategy doc is
# 2x4, 1x8, 4x4, 8x8, 16x16, 2x2x2, 1x1x8, 2x2x4, 4x4x4, 4x4x8 -- shapes
# 8x8, 16x16, 4x4x4, 4x4x8 (64-256 virtual devices) are DELIBERATELY DROPPED
# here (they need >16 simulated devices and OOM-kill this shared box). Do not
# add them even experimentally on this machine; they belong on a dedicated or
# CI runner. Every shape below fits within 16 virtual devices.
CAPPED_MESH_SHAPES = [
    (2, 4),
    (1, 8),
    (4, 4),
    (2, 2, 2),
    (1, 1, 8),
    (2, 2, 4),
]

# Expected sharding specs shared by the Tier-2 and Tier-3 mesh sweeps. These
# cover the always-present (embedding + full-attention + MoE) weight classes;
# the single sweep layer is `full_attention`, so linear-attention specs are
# intentionally absent here (they are asserted by test_distribution instead).
_SWEEP_EXPECTED_SHARDINGS = {
    "token_embedding/embeddings": ("model", "batch"),
    "token_embedding/reverse_embeddings": ("batch", "model"),
    "self_attention.*query.kernel": ("batch", "model", None),
    "self_attention.*key.kernel": ("batch", None, None),
    "self_attention.*value.kernel": ("batch", None, None),
    "self_attention.*attention_output.kernel": ("model", None, "batch"),
    "shared_expert/feedforward_gate_dense.kernel": ("batch", "model"),
    "shared_expert/feedforward_intermediate_dense.kernel": ("batch", "model"),
    "shared_expert/feedforward_output_dense.kernel": ("model", "batch"),
    "experts/expert_feedforward_gate_dense": (None, "batch", "model"),
    "experts/expert_feedforward_output_dense": (None, "model", "batch"),
    "router_gate.kernel": ("batch", None),
}

# Rank>=2 weights that are intentionally left replicated (see
# get_layout_map's comments): the tiny linear-attention a/b projections and
# conv1d kernel, and the (hidden, 1) shared-expert gate.
_ALLOW_REPLICATED = (
    "linear_attn.*in_proj_a.kernel",
    "linear_attn.*in_proj_b.kernel",
    "linear_attn.*conv1d_kernel",
    "shared_expert_gate.kernel",
)


def _assert_shardings_and_coverage(test_case, model, layout_map, expected):
    """Shared spec + coverage assertions for the Tier-2/3 mesh sweeps."""
    for pattern, spec in expected.items():
        matches = [w for w in model.weights if re.search(pattern, w.path)]
        test_case.assertGreater(
            len(matches),
            0,
            f"Expected sharding pattern {pattern!r} matched no weights.",
        )
        for w in matches:
            test_case.assertEqual(tuple(w.value.sharding.spec), spec)
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


class Qwen3_5MoeBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 4,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "hidden_dim": 16,
            "moe_intermediate_dim": 8,
            "shared_expert_intermediate_size": 8,
            "num_experts": 4,
            "top_k": 2,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "partial_rotary_factor": 0.25,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "router_aux_loss_coefficient": 0.01,
            "dtype": "float32",
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=Qwen3_5MoeBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 16),
            run_quantization_check=True,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen3_5MoeBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_num_parameters(self):
        model = Qwen3_5MoeBackbone(**self.init_kwargs)
        self.assertGreater(model.count_params(), 0)

    def test_auxiliary_loss(self):
        model = Qwen3_5MoeBackbone(**self.init_kwargs)
        _ = model(self.input_data, training=True)
        self.assertTrue(
            len(model.losses) > 0, "Auxiliary losses should be present"
        )
        for loss in model.losses:
            self.assertGreater(loss, 0.0, "Auxiliary loss should be positive")

    def test_distribution(self):
        # The default config keeps num_key_value_heads=2 (not divisible by
        # every host's device count) and includes both a linear_attention and
        # a full_attention layer, so the shared helper regression-tests that
        # key/value kernels stay replicated on the model axis while both
        # attention sublayers' and the MoE weights' layouts are asserted. The
        # helper pins the mesh to exactly 2 devices; is_moe=True selects the
        # looser MoE parity tolerance (sharded-reduction float noise is larger
        # for MoE routing -- see the helper docstring).
        self.run_distribution_test(
            cls=Qwen3_5MoeBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_shardings={
                "token_embedding/embeddings": ("model", "batch"),
                "token_embedding/reverse_embeddings": ("batch", "model"),
                "self_attention.*query.kernel": ("batch", "model", None),
                "self_attention.*key.kernel": ("batch", None, None),
                "self_attention.*value.kernel": ("batch", None, None),
                "self_attention.*attention_output.kernel": (
                    "model",
                    None,
                    "batch",
                ),
                "linear_attn.*in_proj_qkv.kernel": ("batch", "model"),
                "linear_attn.*in_proj_z.kernel": ("batch", "model"),
                "linear_attn.*out_proj.kernel": ("model", "batch"),
                "shared_expert/feedforward_gate_dense.kernel": (
                    "batch",
                    "model",
                ),
                "shared_expert/feedforward_intermediate_dense.kernel": (
                    "batch",
                    "model",
                ),
                "shared_expert/feedforward_output_dense.kernel": (
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
                "router_gate.kernel": ("batch", None),
            },
            allow_replicated=_ALLOW_REPLICATED,
            is_moe=True,
        )

    @parameterized.named_parameters(
        (
            f"{dims['source_preset'].split(' ')[0]}_mesh"
            f"_{'x'.join(str(s) for s in shape)}",
            dims,
            shape,
        )
        for dims in (QWEN3_5_MOE_35B_A3B_DIMS, QWEN3_5_MOE_SMALL_DIMS)
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
            # 3D shape: the extra axis is "seq"; get_layout_map only names
            # "batch"/"model", so the "seq" axis simply replicates weights
            # (no rule targets it), matching every other 3D-mesh test in
            # this PR series.
            axis_names = ("batch", "seq", "model")
        device_mesh = keras.distribution.DeviceMesh(
            shape=mesh_shape,
            axis_names=axis_names,
            devices=devices,
        )
        layout_map = Qwen3_5MoeBackbone.get_layout_map(device_mesh)
        distribution = keras.distribution.ModelParallel(
            layout_map=layout_map, batch_dim_name="batch"
        )
        init_kwargs = {k: v for k, v in dims.items() if k != "source_preset"}
        with distribution.scope():
            # bfloat16: a memory mitigation for this shared dev machine --
            # spec assertions are dtype-independent.
            model = Qwen3_5MoeBackbone(dtype="bfloat16", **init_kwargs)
            _assert_shardings_and_coverage(
                self, model, layout_map, _SWEEP_EXPECTED_SHARDINGS
            )
        del model
        gc.collect()

    @pytest.mark.kaggle_key_required
    @pytest.mark.multi_device
    @pytest.mark.extra_large
    def test_layout_map_live_presets(self):
        import json

        from keras_hub.src.utils.preset_utils import CONFIG_FILE
        from keras_hub.src.utils.preset_utils import get_file

        if keras.backend.backend() != "jax":
            self.skipTest("`ModelParallel` testing requires the Jax backend.")

        # Fetch every preset's config only (no weights), then dedupe by the
        # divisibility-relevant dims so width-classes that share a config are
        # only built once per mesh shape -- a memory/time necessity on this
        # machine -- while every preset in the registry is still fetched and
        # evaluated, preserving full registry coverage.
        dim_keys = (
            "vocabulary_size",
            "num_query_heads",
            "num_key_value_heads",
            "hidden_dim",
            "head_dim",
            "moe_intermediate_dim",
            "shared_expert_intermediate_size",
            "num_experts",
        )
        width_classes = {}  # dedupe key -> (config dict, [preset names])
        fetch_failures = []
        for preset in Qwen3_5MoeBackbone.presets:
            try:
                path = get_file(preset, CONFIG_FILE)
                with open(path) as f:
                    cfg = json.load(f)["config"]
            except Exception as e:
                # A preset this account can't reach (e.g. an unaccepted
                # Kaggle license consent) is logged, not fatal -- the rest
                # of the registry still gets exercised.
                fetch_failures.append((preset, str(e)))
                continue
            cfg = dict(cfg)
            cfg["num_layers"] = 1
            # Force a single full_attention layer: layout rules are
            # per-decoder-block regexes, so depth/type mix is irrelevant to
            # spec matching, and one attention layer keeps build memory
            # bounded while still exercising the QKV-axis + MoE rules.
            cfg["layer_types"] = ["full_attention"]
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
                f"({len(fetch_failures)} fetch failures) -- likely no "
                "presets registered yet or a Kaggle license-consent gate "
                "on this account. See the module comment above "
                "CAPPED_MESH_SHAPES."
            )
        print(
            f"test_layout_map_live_presets: {len(width_classes)} unique "
            f"width-classes across {len(Qwen3_5MoeBackbone.presets)} "
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
                        skip_reasons.append(
                            f"{combo_label}: needs {n_needed} devices, "
                            f"only {len(devices)} available"
                        )
                        continue
                    model_axis_size = mesh_shape[-1]
                    if num_query_heads % model_axis_size != 0:
                        skip_reasons.append(
                            f"{combo_label}: num_query_heads="
                            f"{num_query_heads} not divisible by "
                            f"model-axis={model_axis_size}: inherent "
                            "tensor-parallelism limit, not a bug"
                        )
                        continue

                    # Memory-budget guard: this shared dev machine cannot
                    # locally build full-scale presets. Estimate this
                    # width-class's single-decoder-block bf16 footprint
                    # (embedding table + one expert bank's two fused
                    # matrices, times a 3x safety margin for JAX/XLA
                    # transient copies) and skip the build if it exceeds a
                    # conservative local threshold. The fetch/dedup/
                    # divisibility logic above still exercises every
                    # registry preset regardless.
                    vocab = cfg["vocabulary_size"]
                    hidden = cfg["hidden_dim"]
                    moe_int = cfg["moe_intermediate_dim"]
                    n_exp = cfg["num_experts"]
                    est_params = (
                        vocab * hidden
                        + n_exp * hidden * 2 * moe_int
                        + n_exp * moe_int * hidden
                    )
                    est_bytes = est_params * 2 * 3  # bf16 * safety margin
                    max_local_bytes = 300 * 1024 * 1024  # 300MB
                    if est_bytes > max_local_bytes:
                        skip_reasons.append(
                            f"{combo_label}: estimated build memory "
                            f"~{est_bytes / 1e9:.2f}GB exceeds the "
                            f"{max_local_bytes / 1e6:.0f}MB local safety "
                            "threshold on this shared, RAM-constrained "
                            "dev machine -- verify this width-class on a "
                            "machine with more RAM or in CI"
                        )
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
                    layout_map = Qwen3_5MoeBackbone.get_layout_map(device_mesh)
                    distribution = keras.distribution.ModelParallel(
                        layout_map=layout_map, batch_dim_name="batch"
                    )
                    # Build text-only from the constructor-relevant config
                    # keys (drop serialized sub-objects like vision_encoder
                    # and any get_config-only fields such as name/dtype).
                    build_keys = (
                        "vocabulary_size",
                        "num_layers",
                        "num_query_heads",
                        "num_key_value_heads",
                        "head_dim",
                        "hidden_dim",
                        "moe_intermediate_dim",
                        "shared_expert_intermediate_size",
                        "num_experts",
                        "top_k",
                        "layer_types",
                        "partial_rotary_factor",
                        "linear_num_key_heads",
                        "linear_num_value_heads",
                        "linear_key_head_dim",
                        "linear_value_head_dim",
                        "linear_conv_kernel_dim",
                    )
                    build_kwargs = {k: cfg[k] for k in build_keys if k in cfg}
                    with distribution.scope():
                        model = Qwen3_5MoeBackbone(
                            dtype="bfloat16", **build_kwargs
                        )
                        _assert_shardings_and_coverage(
                            self, model, layout_map, _SWEEP_EXPECTED_SHARDINGS
                        )
                    del model
                    gc.collect()
                    ran_any = True

        print(
            f"test_layout_map_live_presets: {len(skip_reasons)} combo(s) "
            "skipped:\n" + "\n".join(f"  {r}" for r in skip_reasons)
        )
        if not ran_any:
            self.skipTest(
                "All (width-class, mesh-shape) combos were skipped: "
                f"{skip_reasons}"
            )
