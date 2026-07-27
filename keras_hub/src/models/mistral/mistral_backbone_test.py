import gc
import os
import re

import keras
import pytest
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.mistral.mistral_backbone import MistralBackbone
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.preset_utils import load_json

# Dims for the Tier-2 CI-safe mesh-shape sweep: representative real-preset
# dimensions, frozen as literals and sourced once, offline -- do not add a
# `get_file` call to the Tier-2 test body itself (that's what Tier 3,
# `test_layout_map_live_presets` below, is for).
#
# MEMORY NOTE: memory-constrained local environments cannot load full-scale
# model dims (see gemma_backbone_test.py's identical note for the OOM
# history that motivated this rule). What matters for the divisibility/
# sharding properties this tier tests is the RATIO of query heads to kv
# heads and whether hidden/intermediate/vocab divide the mesh's model-axis
# sizes -- not the absolute parameter count. So these dims are scaled down
# ~16x from the real Mistral-7B preset config (`hidden_size=4096`,
# `num_attention_heads=32`, `num_key_value_heads=8`,
# `intermediate_size=14336`, `vocab_size=32000`, all five presets in
# `mistral_presets.py` share this single architecture width class -- there
# is no smaller/larger Mistral preset in the registry to source a second
# class from) while preserving the real 4:1 query:kv GQA ratio and keeping
# hidden/intermediate/vocab as clean values divisible by every mesh shape in
# CAPPED_MESH_SHAPES. A second, larger synthetic class is included to
# exercise an 8:1 GQA ratio and a bigger width tier (a plausible
# larger-Mistral extrapolation, not a real preset -- documented as such).
# Full-scale real dims are exercised by `test_layout_map_live_presets`
# below, which has its own per-width-class memory-budget skip so it never
# attempts a full-scale build in a memory-constrained environment either --
# true full-scale verification happens on a dedicated or CI machine with
# more memory.
MISTRAL_7B_DIMS = {
    "source_preset": "mistral_7b_en (real ratio, memory-scaled dims)",
    "vocabulary_size": 2000,  # real: 32000, scaled /16.
    "num_layers": 1,  # depth is irrelevant to spec matching/divisibility.
    "num_query_heads": 8,  # real: 32, scaled /4 (keeps 4:1 GQA ratio).
    "num_key_value_heads": 2,  # real: 8, scaled /4 -- 4:1 GQA ratio.
    "hidden_dim": 256,  # real: 4096, scaled /16.
    "intermediate_dim": 896,  # real: 14336, scaled /16.
    "sliding_window": 128,
}
MISTRAL_LARGE_DIMS = {
    # Not a real preset -- a plausible larger-Mistral extrapolation (doubled
    # hidden/intermediate vs. MISTRAL_7B_DIMS) used to exercise an 8:1 GQA
    # ratio and a bigger width tier under the mesh sweep.
    "source_preset": "mistral_large (synthetic, larger width class)",
    "vocabulary_size": 4000,
    "num_layers": 1,
    "num_query_heads": 16,
    "num_key_value_heads": 2,  # 8:1 GQA ratio.
    "hidden_dim": 512,
    "intermediate_dim": 1792,
    "sliding_window": 128,
}

# Hard-capped mesh-shape list for memory-constrained local environments. See
# gemma_backbone_test.py's identical constant for the full rationale (an
# OOM kill history from an earlier run of this pipeline in a
# memory-constrained environment) -- the dropped shapes (8x8, 16x16, 4x4x4,
# 4x4x8) are never attempted here either.
CAPPED_MESH_SHAPES = [
    (2, 4),
    (1, 8),
    (4, 4),
    (2, 2, 2),
    (1, 1, 8),
    (2, 2, 4),
]

# Same 8 expected_shardings patterns as MistralBackboneTest.test_distribution
# (post-QKV-axis-fix), reused by the Tier-2 and Tier-3 mesh sweeps below.
_EXPECTED_SHARDINGS = {
    "token_embedding/embeddings": ("model", "batch"),
    "token_embedding/reverse_embeddings": ("batch", "model"),
    "self_attention.*query.kernel": ("batch", "model", None),
    "self_attention.*(key|value).kernel": ("batch", None, None),
    "self_attention.*attention_output.kernel": ("model", None, "batch"),
    "feedforward_intermediate_dense.kernel": ("batch", "model"),
    "feedforward_gate_dense.kernel": ("batch", "model"),
    "feedforward_output_dense.kernel": ("model", "batch"),
}


def _assert_mistral_shardings_and_coverage(test_case, model, layout_map):
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


class MistralBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_query_heads": 8,
            "num_key_value_heads": 4,
            "hidden_dim": 16,
            "intermediate_dim": 8,
            "sliding_window": 2,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=MistralBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 16),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MistralBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_num_parameters(self):
        model = MistralBackbone(**self.init_kwargs)
        # Reference value calculated using the PyTorch model
        self.assertEqual(model.count_params(), 2704)

    def test_explicit_head_dim(self):
        # Magistral-style config: `head_dim` is set explicitly and does not
        # equal `hidden_dim // num_query_heads`. `sliding_window=None` is
        # also exercised here. Run the full backbone test so the new path
        # gets serialization and precision coverage.
        init_kwargs = {
            **self.init_kwargs,
            "sliding_window": None,
            "head_dim": 4,
        }
        self.run_backbone_test(
            cls=MistralBackbone,
            init_kwargs=init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 16),
        )
        model = MistralBackbone(**init_kwargs)
        attention = model.transformer_layers[0]._self_attention_layer
        self.assertEqual(attention._head_dim, 4)

    @pytest.mark.multi_device
    def test_distribution(self):
        # Note (preserved from the pre-refactor manual test): mesh is
        # pinned to exactly 2 devices (not len(devices), see the shared
        # helper) so that the default test config's num_key_value_heads=4
        # -- intentionally not divisible by every host's device count --
        # regression-tests that key/value kernels are left replicated on
        # the model axis rather than sharded. See get_layout_map's comment.
        self.run_distribution_test(
            cls=MistralBackbone,
            init_kwargs=self.init_kwargs,
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
                "feedforward_intermediate_dense.kernel": (
                    "batch",
                    "model",
                ),
                "feedforward_gate_dense.kernel": ("batch", "model"),
                "feedforward_output_dense.kernel": ("model", "batch"),
            },
            allow_replicated=(),
        )

    @parameterized.named_parameters(
        (
            f"{dims['source_preset'].split(' ')[0]}_mesh"
            f"_{'x'.join(str(s) for s in shape)}",
            dims,
            shape,
        )
        for dims in (MISTRAL_7B_DIMS, MISTRAL_LARGE_DIMS)
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
        layout_map = MistralBackbone.get_layout_map(device_mesh)
        distribution = keras.distribution.ModelParallel(
            layout_map=layout_map, batch_dim_name="batch"
        )
        init_kwargs = {k: v for k, v in dims.items() if k != "source_preset"}
        with distribution.scope():
            # bfloat16: a memory mitigation for memory-constrained
            # environments -- spec assertions are dtype-independent.
            model = MistralBackbone(dtype="bfloat16", **init_kwargs)
            _assert_mistral_shardings_and_coverage(self, model, layout_map)
        del model
        gc.collect()

    @pytest.mark.multi_device
    def test_query_head_divisibility_skip_triggers(self):
        """Regression test: the skip rule in `test_layout_map_mesh_shapes`
        is not dead code -- it must actually trigger for a config where
        `num_query_heads` does not divide the mesh's model-axis size, and
        that indivisibility must be real (i.e. building without the skip
        would raise, proving the guard is load-bearing rather than
        decorative).
        """
        if keras.backend.backend() != "jax":
            self.skipTest("`ModelParallel` testing requires the Jax backend.")
        devices = keras.distribution.list_devices("CPU")
        # Pinned to exactly 2 devices, matching run_distribution_test's
        # convention -- deterministic regardless of how many virtual
        # devices the test environment exposes.
        if len(devices) < 2:
            self.skipTest("`ModelParallel` testing requires multiple devices.")
        devices = devices[:2]
        mesh_shape = (1, 2)
        model_axis_size = mesh_shape[-1]

        # num_query_heads=3 does not divide model_axis_size=2 -- this is
        # the condition the skip rule in test_layout_map_mesh_shapes
        # guards against.
        num_query_heads = 3
        self.assertNotEqual(num_query_heads % model_axis_size, 0)

        device_mesh = keras.distribution.DeviceMesh(
            shape=mesh_shape,
            axis_names=("batch", "model"),
            devices=devices,
        )
        layout_map = MistralBackbone.get_layout_map(device_mesh)
        distribution = keras.distribution.ModelParallel(
            layout_map=layout_map, batch_dim_name="batch"
        )
        # Proves the skip rule is necessary: building this genuinely
        # indivisible config under the same layout map that
        # test_layout_map_mesh_shapes would have used (had it not skipped)
        # actually raises, rather than silently succeeding.
        with self.assertRaises(Exception):
            with distribution.scope():
                MistralBackbone(
                    vocabulary_size=40,
                    num_layers=1,
                    num_query_heads=num_query_heads,
                    num_key_value_heads=1,
                    hidden_dim=24,
                    intermediate_dim=16,
                    sliding_window=4,
                    dtype="bfloat16",
                )
        gc.collect()

    @pytest.mark.kaggle_key_required
    @pytest.mark.multi_device
    @pytest.mark.extra_large
    def test_layout_map_live_presets(self):
        if keras.backend.backend() != "jax":
            self.skipTest("`ModelParallel` testing requires the Jax backend.")

        # Fetch every preset's config only (no weights), then dedupe by the
        # divisibility-relevant dims so width-classes that share a config
        # (e.g. base vs instruct variants of the same size) are only built
        # once per mesh shape -- a memory/time necessity on this machine,
        # while every preset in the registry is still fetched and
        # evaluated, preserving full registry coverage.
        dim_keys = (
            "vocabulary_size",
            "num_query_heads",
            "num_key_value_heads",
            "hidden_dim",
            "intermediate_dim",
            "sliding_window",
        )
        width_classes = {}  # dedupe key -> (config dict, [preset names])
        fetch_failures = []
        for preset in MistralBackbone.presets:
            try:
                cfg = load_json(preset)["config"]
            except Exception as e:
                # A preset this account can't reach (e.g. an unaccepted
                # Kaggle license consent click-through) is logged, not
                # fatal -- the rest of the registry still gets exercised.
                fetch_failures.append((preset, str(e)))
                continue
            # num_layers is forced to 1 below regardless of the real
            # value -- layout rules are per-decoder-block regexes, so
            # depth is irrelevant to spec matching/divisibility, and 1
            # layer keeps build memory bounded in memory-constrained
            # environments.
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
                "mistral family. See the module comment above "
                "CAPPED_MESH_SHAPES."
            )
        print(
            f"test_layout_map_live_presets: {len(width_classes)} unique "
            f"width-classes across {len(MistralBackbone.presets)} registry "
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

                    # Memory-budget guard: memory-constrained local
                    # environments cannot build full-scale presets (see
                    # CAPPED_MESH_SHAPES' comment for the OOM history).
                    # Estimate this width-class's single-decoder-block bf16
                    # footprint and skip the actual build if it exceeds a
                    # conservative local threshold. Mistral has UNTIED
                    # embeddings (`tie_weights=False` in MistralBackbone),
                    # so the embedding term is counted twice (forward +
                    # reverse output projection). Attention projections map
                    # hidden<->hidden for query/output and hidden<->(hidden
                    # scaled by the kv/query head ratio) for GQA's key/value.
                    # The FFN is SwiGLU (gate + intermediate + output, 3
                    # hidden*intermediate matrices), matching this model's
                    # dense (non-MoE) feedforward block. Times a 3x safety
                    # margin for JAX/XLA transient copies during
                    # construction/resharding. The config-fetch, dedup, and
                    # divisibility-skip logic above still exercises every
                    # registry preset either way; only the expensive
                    # build+assert step is capped.
                    hidden = cfg["hidden_dim"]
                    num_kv_heads = cfg["num_key_value_heads"]
                    kv_ratio = num_kv_heads / num_query_heads
                    est_params = (
                        2 * cfg["vocabulary_size"] * hidden  # untied embed
                        + 2 * hidden * hidden  # attention q/o
                        + 2 * hidden * hidden * kv_ratio  # attention k/v
                        + 3 * hidden * cfg["intermediate_dim"]  # SwiGLU FFN
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
                    layout_map = MistralBackbone.get_layout_map(device_mesh)
                    distribution = keras.distribution.ModelParallel(
                        layout_map=layout_map, batch_dim_name="batch"
                    )
                    # `cfg` is the preset's full serialized `get_config()`
                    # dict; pass all of it (real architecture flags such as
                    # rope scaling, tie-embeddings, etc. must survive to
                    # actually exercise the live preset's real architecture)
                    # and only drop `"dtype"` so the explicit
                    # `dtype="bfloat16"` override below doesn't collide with
                    # a duplicate keyword argument. `num_layers` is forced to
                    # 1 (layout rules are per-decoder-block, so depth is
                    # irrelevant and 1 layer keeps build memory bounded).
                    init_kwargs = {k: v for k, v in cfg.items() if k != "dtype"}
                    init_kwargs["num_layers"] = 1
                    with distribution.scope():
                        model = MistralBackbone(dtype="bfloat16", **init_kwargs)
                        _assert_mistral_shardings_and_coverage(
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

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=MistralBackbone,
            preset="mistral_7b_en",
            input_data={
                "token_ids": ops.array([[1, 1824, 349, 524, 11234, 28804]]),
                "padding_mask": ops.ones((1, 6), dtype="int32"),
            },
            expected_output_shape=(1, 6, 4096),
            # The forward pass from a preset should be stable!
            # Reference values computed using PyTorch HF model.
            expected_partial_output=ops.array(
                [-1.6875, 0.5117, -1.7188, 2.3125, -0.0996]
            ),
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in MistralBackbone.presets:
            self.run_preset_test(
                cls=MistralBackbone,
                preset=preset,
                input_data=self.input_data,
            )
