import gc
import os
import re

import keras
import pytest
from absl.testing import parameterized

from keras_hub.src.models.t5gemma.t5gemma_backbone import T5GemmaBackbone
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.preset_utils import load_json

# Dims for the Tier-2 CI-safe mesh-shape sweep: representative real-preset
# dimensions, frozen as literals and sourced once, offline -- do not add a
# `get_file` call to the Tier-2 test body itself (that's what Tier 3,
# `test_layout_map_live_presets` below, is for).
#
# MEMORY NOTE: memory-constrained local environments cannot load full-scale
# model dims (see gemma_backbone_test.py's identical note for the OOM
# history that motivated this pattern). What actually matters for the
# divisibility/sharding properties this tier tests is the RATIO of query
# heads to kv heads and whether hidden/intermediate/vocab divide the mesh's
# model-axis sizes -- not the absolute parameter count. So these dims are
# scaled down by roughly 20-30x from the real presets while preserving each
# preset's real query:kv head ratio and keeping hidden/intermediate/vocab
# as clean powers of 2 divisible by every mesh shape in CAPPED_MESH_SHAPES.
#
# T5Gemma's real encoder/decoder configs are symmetric per preset (encoder
# and decoder share hidden/intermediate/head-count) and use plain
# multi-head attention -- num_attention_heads == num_key_value_heads, 1:1,
# no GQA -- confirmed from the two smallest and largest locally-cached HF
# configs (t5gemma-s-s-ul2, t5gemma-b-b-ul2, t5gemma-l-l-ul2; this repo's
# registry preset naming is t5gemma_s_s_*/t5gemma_l_l_* etc.):
#   S/S: hidden=512, intermediate=1024, heads=kv_heads=8, head_dim=64,
#        vocab=256000
#   L/L: hidden=1024, intermediate=2816, heads=kv_heads=16, head_dim=64,
#        vocab=256000
# `cross_attention_hidden_size` equals `encoder_hidden_dim` in both real
# configs, so it is omitted below (T5GemmaBackbone's own default already
# does this). Full-scale real dims are exercised by
# `test_layout_map_live_presets` below, which has its own per-width-class
# memory-budget skip so it never attempts a full-scale build locally either.
# `tie_word_embeddings: False` on both dicts below: T5GemmaBackbone's
# default is `True` (unlike e.g. Llama, whose default is `False`), under
# which `decoder_token_embedding` has no separate `reverse_embeddings`
# weight at all (see `keras.layers.ReversibleEmbedding.build`: the weight
# is only created when `tie_weights=False`) -- the `reverse_embeddings`
# entry in `_EXPECTED_SHARDINGS` would then match zero weights, a dead
# assertion. Untying here exercises the rule, matching the untied real
# presets referenced in `get_layout_map`'s comment.
T5GEMMA_S_S_DIMS = {
    "source_preset": "t5gemma_s_s (real ratio, memory-scaled dims)",
    "vocabulary_size": 2048,  # real 256000, scaled ~125x (vocab-heavy model).
    "encoder_num_layers": 1,  # depth is irrelevant to spec matching.
    "decoder_num_layers": 1,
    "encoder_num_attention_heads": 8,  # real ratio: MHA, 8:8 = 1:1.
    "encoder_num_key_value_heads": 8,
    "decoder_num_attention_heads": 8,
    "decoder_num_key_value_heads": 8,
    "encoder_hidden_dim": 128,  # real 512, scaled 4x.
    "decoder_hidden_dim": 128,
    "encoder_intermediate_dim": 256,  # real 1024, scaled 4x.
    "decoder_intermediate_dim": 256,
    "encoder_head_dim": 32,  # real 64, scaled 2x.
    "decoder_head_dim": 32,
    "tie_word_embeddings": False,
}
T5GEMMA_L_L_DIMS = {
    "source_preset": "t5gemma_l_l (real ratio, memory-scaled dims)",
    "vocabulary_size": 2048,
    "encoder_num_layers": 1,
    "decoder_num_layers": 1,
    "encoder_num_attention_heads": 16,  # real ratio: MHA, 16:16 = 1:1.
    "encoder_num_key_value_heads": 16,
    "decoder_num_attention_heads": 16,
    "decoder_num_key_value_heads": 16,
    "encoder_hidden_dim": 256,  # real 1024, scaled 4x.
    "decoder_hidden_dim": 256,
    "encoder_intermediate_dim": 512,  # real 2816, scaled ~5.5x.
    "decoder_intermediate_dim": 512,
    "encoder_head_dim": 32,  # real 64, scaled 2x.
    "decoder_head_dim": 32,
    "tie_word_embeddings": False,
}

# Hard-capped mesh-shape list for memory-constrained local environments.
# The full 10-shape matrix from the testing-strategy doc is
# 2x4, 1x8, 4x4, 8x8, 16x16, 2x2x2, 1x1x8, 2x2x4, 4x4x4, 4x4x8 -- shapes
# 8x8, 16x16, 4x4x4, 4x4x8 (64-256 virtual devices) are DELIBERATELY
# DROPPED here due to a demonstrated OOM kill of the entire desktop
# environment during an earlier attempt at this same pipeline on a
# memory-constrained machine. Do not attempt the dropped shapes even
# experimentally in such environments -- revisiting them requires a
# dedicated or CI machine with more memory.
CAPPED_MESH_SHAPES = [
    (2, 4),
    (1, 8),
    (4, 4),
    (2, 2, 2),
    (1, 1, 8),
    (2, 2, 4),
]

# Same expected_shardings patterns as T5GemmaBackboneTest.test_distribution
# (post-QKV-axis-fix), reused by the Tier-2 and Tier-3 mesh sweeps below.
# Covers encoder self-attention, decoder self-attention, and decoder
# cross-attention (all three share query/key/value/attention_output naming
# under the broad "attention" wildcard in get_layout_map).
_EXPECTED_SHARDINGS = {
    "encoder_token_embedding/embeddings": ("model", "batch"),
    "decoder_token_embedding/embeddings": ("model", "batch"),
    "decoder_token_embedding/reverse_embeddings": ("batch", "model"),
    "encoder_layer.*self_attention.*query.kernel": ("batch", "model", None),
    "encoder_layer.*self_attention.*key.kernel": ("batch", None, None),
    "encoder_layer.*self_attention.*value.kernel": ("batch", None, None),
    "encoder_layer.*attention_output.kernel": ("model", None, "batch"),
    "decoder_layer.*self_attention.*query.kernel": ("batch", "model", None),
    "decoder_layer.*self_attention.*key.kernel": ("batch", None, None),
    "decoder_layer.*self_attention.*value.kernel": ("batch", None, None),
    "decoder_layer.*cross_attention.*query.kernel": ("batch", "model", None),
    "decoder_layer.*cross_attention.*key.kernel": ("batch", None, None),
    "decoder_layer.*cross_attention.*value.kernel": ("batch", None, None),
    "decoder_layer.*attention_output.kernel": ("model", None, "batch"),
    "encoder_layer.*gate_proj.kernel": ("batch", "model"),
    "encoder_layer.*up_proj.kernel": ("batch", "model"),
    "encoder_layer.*down_proj.kernel": ("model", "batch"),
    "decoder_layer.*gate_proj.kernel": ("batch", "model"),
    "decoder_layer.*up_proj.kernel": ("batch", "model"),
    "decoder_layer.*down_proj.kernel": ("model", "batch"),
}


def _assert_t5gemma_shardings_and_coverage(test_case, model, layout_map):
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


class T5GemmaBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 100,
            "encoder_hidden_dim": 32,
            "encoder_intermediate_dim": 64,
            "encoder_num_layers": 2,
            "encoder_num_attention_heads": 4,
            "encoder_num_key_value_heads": 2,
            "encoder_head_dim": 8,
            "encoder_layer_types": ["sliding_attention", "full_attention"],
            "decoder_hidden_dim": 32,
            "decoder_intermediate_dim": 64,
            "decoder_num_layers": 2,
            "decoder_num_attention_heads": 4,
            "decoder_num_key_value_heads": 2,
            "decoder_head_dim": 8,
            "decoder_layer_types": ["sliding_attention", "full_attention"],
            "dropout_rate": 0.1,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": True,
            "query_pre_attn_scalar": 1.0,
            "attention_bias": False,
            "hidden_activation": "gelu_approximate",
            "sliding_window": 16,
            "cross_attention_hidden_size": 32,
            "attn_logit_softcapping": 50.0,
            "rope_max_wavelength": 10000.0,
            "initializer_range": 0.04,
            "attention_dropout": 0.1,
        }
        self.input_data = {
            "encoder_token_ids": keras.ops.ones((2, 16), dtype="int32"),
            "encoder_padding_mask": keras.ops.ones((2, 16), dtype="int32"),
            "decoder_token_ids": keras.ops.ones((2, 16), dtype="int32"),
            "decoder_padding_mask": keras.ops.ones((2, 16), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=T5GemmaBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "encoder_sequence_output": (2, 16, 32),
                "decoder_sequence_output": (2, 16, 32),
            },
        )

    def test_asymmetrical_backbone(self):
        asym_kwargs = {
            "vocabulary_size": 100,
            "encoder_hidden_dim": 48,
            "encoder_intermediate_dim": 96,
            "encoder_num_layers": 3,
            "encoder_num_attention_heads": 6,
            "encoder_num_key_value_heads": 3,
            "encoder_head_dim": 8,
            "encoder_layer_types": ["full_attention"] * 3,
            "decoder_hidden_dim": 32,
            "decoder_intermediate_dim": 64,
            "decoder_num_layers": 2,
            "decoder_num_attention_heads": 4,
            "decoder_num_key_value_heads": 2,
            "decoder_head_dim": 8,
            "decoder_layer_types": ["sliding_attention", "full_attention"],
            "sliding_window": 16,
            "dropout_rate": 0.1,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": True,
            "cross_attention_hidden_size": 48,
        }
        self.run_backbone_test(
            cls=T5GemmaBackbone,
            init_kwargs=asym_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "encoder_sequence_output": (2, 16, 48),
                "decoder_sequence_output": (2, 16, 32),
            },
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=T5GemmaBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in T5GemmaBackbone.presets:
            self.run_preset_test(
                cls=T5GemmaBackbone,
                preset=preset,
                input_data=self.input_data,
            )

    @pytest.mark.multi_device
    def test_distribution(self):
        # Note: mesh is pinned to exactly 2 devices (not len(devices), see
        # the shared helper) so that the default test config's
        # *_num_key_value_heads=2 -- intentionally not divisible by every
        # host's device count -- regression-tests that key/value kernels
        # are left replicated rather than sharded. See get_layout_map's
        # comment for why.
        #
        # `tie_word_embeddings` defaults to `True` (unlike e.g. Llama, whose
        # default is `False`), so `self.init_kwargs` alone would build a
        # `decoder_token_embedding` with no separate `reverse_embeddings`
        # weight at all (see `keras.layers.ReversibleEmbedding.build`: the
        # weight is only created when `tie_weights=False`) -- the sharding
        # rule would then match zero weights, a dead assertion. Override to
        # `False` here so the `reverse_embeddings` rule is actually
        # exercised, matching the untied real presets referenced in
        # `get_layout_map`'s comment.
        distribution_init_kwargs = dict(self.init_kwargs)
        distribution_init_kwargs["tie_word_embeddings"] = False
        self.run_distribution_test(
            cls=T5GemmaBackbone,
            init_kwargs=distribution_init_kwargs,
            input_data=self.input_data,
            expected_shardings={
                "encoder_token_embedding/embeddings": ("model", "batch"),
                "decoder_token_embedding/embeddings": ("model", "batch"),
                "decoder_token_embedding/reverse_embeddings": (
                    "batch",
                    "model",
                ),
                "encoder_layer.*self_attention.*query.kernel": (
                    "batch",
                    "model",
                    None,
                ),
                "encoder_layer.*self_attention.*key.kernel": (
                    "batch",
                    None,
                    None,
                ),
                "encoder_layer.*self_attention.*value.kernel": (
                    "batch",
                    None,
                    None,
                ),
                "encoder_layer.*attention_output.kernel": (
                    "model",
                    None,
                    "batch",
                ),
                "decoder_layer.*self_attention.*query.kernel": (
                    "batch",
                    "model",
                    None,
                ),
                "decoder_layer.*self_attention.*key.kernel": (
                    "batch",
                    None,
                    None,
                ),
                "decoder_layer.*self_attention.*value.kernel": (
                    "batch",
                    None,
                    None,
                ),
                "decoder_layer.*cross_attention.*query.kernel": (
                    "batch",
                    "model",
                    None,
                ),
                "decoder_layer.*cross_attention.*key.kernel": (
                    "batch",
                    None,
                    None,
                ),
                "decoder_layer.*cross_attention.*value.kernel": (
                    "batch",
                    None,
                    None,
                ),
                "decoder_layer.*attention_output.kernel": (
                    "model",
                    None,
                    "batch",
                ),
                "encoder_layer.*gate_proj.kernel": ("batch", "model"),
                "encoder_layer.*up_proj.kernel": ("batch", "model"),
                "encoder_layer.*down_proj.kernel": ("model", "batch"),
                "decoder_layer.*gate_proj.kernel": ("batch", "model"),
                "decoder_layer.*up_proj.kernel": ("batch", "model"),
                "decoder_layer.*down_proj.kernel": ("model", "batch"),
            },
            allow_replicated=(),
        )

    @pytest.mark.multi_device
    def test_layout_map_query_heads_fallback(self):
        # Regression test for the query-head-count-divisibility fallback:
        # a small mesh whose model axis does not divide
        # encoder_num_attention_heads/decoder_num_attention_heads must not
        # raise IndivisibleError. get_layout_map should fall back to fully
        # replicating the query/attention_output head axis, mirroring the
        # existing key/value fallback.
        if keras.backend.backend() != "jax":
            self.skipTest("`ModelParallel` testing requires the Jax backend.")
        devices = keras.distribution.list_devices("CPU")
        if len(devices) < 8:
            self.skipTest(
                "This test requires 8 devices. Run with "
                "XLA_FLAGS=--xla_force_host_platform_device_count=8 to "
                "exercise this locally."
            )
        devices = devices[:8]
        # num_attention_heads=4 does not divide a model-axis size of 8 --
        # an inherent tensor-parallelism ceiling that used to raise
        # IndivisibleError; get_layout_map should now fall back to
        # replication instead. 4 mirrors the smallest real T5Gemma preset's
        # head count (t5gemma_ml_ml).
        num_attention_heads = 4
        self.assertNotEqual(num_attention_heads % 8, 0)
        device_mesh = keras.distribution.DeviceMesh(
            shape=(1, 8),
            axis_names=("batch", "model"),
            devices=devices,
        )
        layout_map = T5GemmaBackbone.get_layout_map(
            device_mesh,
            encoder_num_attention_heads=num_attention_heads,
            decoder_num_attention_heads=num_attention_heads,
        )
        distribution = keras.distribution.ModelParallel(
            layout_map=layout_map, batch_dim_name="batch"
        )
        # vocabulary_size/hidden_dim are bumped from self.init_kwargs'
        # defaults to multiples of 8 -- this test targets the query-head
        # fallback specifically, so the embedding table (a separate,
        # already-known divisibility concern, out of scope here) must
        # itself divide the model-axis size cleanly, or its own
        # IndivisibleError would mask the assertion this test exists for.
        init_kwargs = dict(
            self.init_kwargs,
            vocabulary_size=64,
            encoder_hidden_dim=32,
            decoder_hidden_dim=32,
            cross_attention_hidden_size=32,
            encoder_num_attention_heads=num_attention_heads,
            encoder_num_key_value_heads=num_attention_heads,
            decoder_num_attention_heads=num_attention_heads,
            decoder_num_key_value_heads=num_attention_heads,
            tie_word_embeddings=False,
        )
        with distribution.scope():
            model = T5GemmaBackbone(**init_kwargs)
            for pattern in (
                "encoder_layer.*attention.*query.kernel",
                "decoder_layer.*attention.*query.kernel",
            ):
                query_weights = [
                    w for w in model.weights if re.search(pattern, w.path)
                ]
                self.assertGreater(len(query_weights), 0)
                for w in query_weights:
                    self.assertEqual(
                        tuple(w.value.sharding.spec), ("batch", None, None)
                    )
            for pattern in (
                "encoder_layer.*attention.*attention_output.kernel",
                "decoder_layer.*attention.*attention_output.kernel",
            ):
                output_weights = [
                    w for w in model.weights if re.search(pattern, w.path)
                ]
                self.assertGreater(len(output_weights), 0)
                for w in output_weights:
                    self.assertEqual(
                        tuple(w.value.sharding.spec), (None, None, "batch")
                    )
        del model
        gc.collect()

    @parameterized.named_parameters(
        (
            f"{dims['source_preset'].split(' ')[0]}_mesh"
            f"_{'x'.join(str(s) for s in shape)}",
            dims,
            shape,
        )
        for dims in (T5GEMMA_S_S_DIMS, T5GEMMA_L_L_DIMS)
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
        # matching this repo's axis_names=(..., "model") convention. Both
        # the encoder and decoder query-head counts must divide the
        # model-axis size for this width class -- T5Gemma's real presets are
        # symmetric (encoder_num_attention_heads ==
        # decoder_num_attention_heads), so a single check suffices, but both
        # are checked defensively in case a future width class isn't
        # symmetric.
        model_axis_size = mesh_shape[-1]
        encoder_num_query_heads = dims["encoder_num_attention_heads"]
        decoder_num_query_heads = dims["decoder_num_attention_heads"]
        if (
            encoder_num_query_heads % model_axis_size != 0
            or decoder_num_query_heads % model_axis_size != 0
        ):
            self.skipTest(
                f"encoder_num_attention_heads={encoder_num_query_heads} or "
                f"decoder_num_attention_heads={decoder_num_query_heads} "
                f"not divisible by model-axis={model_axis_size}: inherent "
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
        layout_map = T5GemmaBackbone.get_layout_map(device_mesh)
        distribution = keras.distribution.ModelParallel(
            layout_map=layout_map, batch_dim_name="batch"
        )
        init_kwargs = {k: v for k, v in dims.items() if k != "source_preset"}
        init_kwargs["encoder_layer_types"] = ["full_attention"] * init_kwargs[
            "encoder_num_layers"
        ]
        init_kwargs["decoder_layer_types"] = ["full_attention"] * init_kwargs[
            "decoder_num_layers"
        ]
        with distribution.scope():
            # bfloat16: a memory mitigation for memory-constrained local
            # environments -- spec assertions are dtype-independent.
            model = T5GemmaBackbone(dtype="bfloat16", **init_kwargs)
            _assert_t5gemma_shardings_and_coverage(self, model, layout_map)
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
        # (e.g. the ul2/prefixlm/ul2_it/prefixlm_it variants of the same
        # size) are only built once per mesh shape -- a memory/time
        # necessity in memory-constrained local environments, while every
        # preset in the registry is still fetched and evaluated, preserving
        # full registry coverage.
        # `load_json(preset)` fetches this repo's own `config.json` (a
        # serialized `T5GemmaBackbone.get_config()`, i.e. already in the
        # flat encoder_*/decoder_* backbone-kwarg shape used by
        # `T5GemmaBackbone.__init__` -- NOT the raw nested HF
        # `{"encoder": {...}, "decoder": {...}}` shape, which lives under
        # the separate `HF_CONFIG_FILE` used only by the Transformers
        # preset loader), wrapped under a top-level `"config"` key (see
        # `KerasPresetLoader`/`set_dtype_in_config` in `preset_utils.py`).
        dim_keys = (
            "vocabulary_size",
            "encoder_hidden_dim",
            "decoder_hidden_dim",
            "encoder_intermediate_dim",
            "decoder_intermediate_dim",
            "encoder_num_attention_heads",
            "decoder_num_attention_heads",
            "encoder_num_key_value_heads",
            "decoder_num_key_value_heads",
            "encoder_head_dim",
            "decoder_head_dim",
        )
        width_classes = {}  # dedupe key -> (config dict, [preset names])
        fetch_failures = []
        for preset in T5GemmaBackbone.presets:
            try:
                cfg = load_json(preset)["config"]
            except Exception as e:
                # A preset this account can't reach (e.g. an unaccepted
                # Kaggle license consent click-through, or no
                # KAGGLE_USERNAME/KAGGLE_KEY at all) is logged, not fatal --
                # the rest of the registry still gets exercised.
                fetch_failures.append((preset, str(e)))
                continue
            # num_layers is forced to 1 below regardless of the real
            # value -- layout rules are per-encoder/decoder-block regexes,
            # so depth is irrelevant to spec matching/divisibility, and 1
            # layer keeps build memory bounded in memory-constrained local
            # environments.
            cfg = dict(cfg)
            cfg["encoder_num_layers"] = 1
            cfg["decoder_num_layers"] = 1
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
                f"({len(fetch_failures)} fetch failures) -- likely missing "
                "KAGGLE_USERNAME/KAGGLE_KEY or a Kaggle license-consent "
                "gate on this account for the t5gemma family. See the "
                "module comment above CAPPED_MESH_SHAPES."
            )
        print(
            f"test_layout_map_live_presets: {len(width_classes)} unique "
            f"width-classes across {len(T5GemmaBackbone.presets)} registry "
            "presets:"
        )
        for cfg, presets in width_classes.values():
            print(f"  {presets}")

        devices = keras.distribution.list_devices("CPU")
        skip_reasons = []
        ran_any = False
        for cfg, presets in width_classes.values():
            encoder_num_query_heads = cfg["encoder_num_attention_heads"]
            decoder_num_query_heads = cfg["decoder_num_attention_heads"]
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
                    if (
                        encoder_num_query_heads % model_axis_size != 0
                        or decoder_num_query_heads % model_axis_size != 0
                    ):
                        reason = (
                            f"{combo_label}: encoder_num_attention_heads="
                            f"{encoder_num_query_heads} or "
                            "decoder_num_attention_heads="
                            f"{decoder_num_query_heads} not divisible by "
                            f"model-axis={model_axis_size}: inherent "
                            "tensor-parallelism limit, not a bug"
                        )
                        skip_reasons.append(reason)
                        continue

                    # Memory-budget guard: memory-constrained local
                    # environments cannot locally build full-scale presets
                    # (see CAPPED_MESH_SHAPES' comment for the OOM history
                    # that motivated this). Estimate this width-class's
                    # single-encoder-block + single-decoder-block bf16
                    # footprint (embedding table + each side's one FFN
                    # block's 3 matrices, times a 3x safety margin for
                    # JAX/XLA transient copies during construction/
                    # resharding) and skip the actual build if it exceeds a
                    # conservative local threshold. The config-fetch,
                    # dedup, and divisibility-skip logic above still
                    # exercises every registry preset either way; only the
                    # expensive build+assert step is capped.
                    est_params = (
                        cfg["vocabulary_size"] * cfg["encoder_hidden_dim"]
                        + cfg["vocabulary_size"] * cfg["decoder_hidden_dim"]
                        + 3
                        * cfg["encoder_hidden_dim"]
                        * cfg["encoder_intermediate_dim"]
                        + 3
                        * cfg["decoder_hidden_dim"]
                        * cfg["decoder_intermediate_dim"]
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
                            "threshold for memory-constrained local "
                            "environments -- verify this width-class on a "
                            "machine with more memory or in CI (see "
                            "KERAS_HUB_DISTRIBUTION_TEST_MEM_BUDGET)"
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
                    layout_map = T5GemmaBackbone.get_layout_map(device_mesh)
                    distribution = keras.distribution.ModelParallel(
                        layout_map=layout_map, batch_dim_name="batch"
                    )
                    # `cfg` is a full serialized `get_config()` dict, which
                    # may include a `"dtype"` key (a serialized dtype-policy
                    # dict, only present for quantized presets) -- drop it
                    # so the explicit `dtype="bfloat16"` override below
                    # doesn't collide with a duplicate keyword argument.
                    init_kwargs = {k: v for k, v in cfg.items() if k != "dtype"}
                    init_kwargs["encoder_num_layers"] = 1
                    init_kwargs["decoder_num_layers"] = 1
                    init_kwargs["encoder_layer_types"] = ["full_attention"]
                    init_kwargs["decoder_layer_types"] = ["full_attention"]
                    # Force untied embeddings regardless of this preset's
                    # real `tie_word_embeddings` value: some real presets
                    # are tied, under which `decoder_token_embedding` has
                    # no separate `reverse_embeddings` weight at all (see
                    # `keras.layers.ReversibleEmbedding.build`), which would
                    # make `_EXPECTED_SHARDINGS`'s `reverse_embeddings`
                    # pattern match zero weights -- a dead assertion. Both
                    # tied and untied real presets exist in this registry
                    # (see `get_layout_map`'s comment), so forcing untied
                    # here exercises the rule on every width-class rather
                    # than only on whichever presets happen to already be
                    # untied.
                    init_kwargs["tie_word_embeddings"] = False
                    with distribution.scope():
                        model = T5GemmaBackbone(dtype="bfloat16", **init_kwargs)
                        _assert_t5gemma_shardings_and_coverage(
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
