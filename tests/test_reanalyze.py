"""Stage 7 reanalysis sanity checks (PLAYBOOK2.md Stage 7).

These run against the real Phase 1 results/ directory (no mocking of
checkpoints or JSONL, per this project's testing conventions) but do no new
training: only JSONL reads and seeded/checkpoint state reconstruction.
"""

import numpy as np
import pytest
import torch

from qnc.reanalyze import (
    STAGE14B_REUPLOAD_FALSE_RUN_IDS,
    STAGE14B_REUPLOAD_FALSE_SWEEP_CONFIG,
    STAGE14_MECHANISM_BASE_CONFIG,
    _build_model_and_data,
    _max_relative_change,
    _parse_seed,
    _recompute_epoch_witnesses,
    _recompute_zspace_witnesses,
    _stage9_variant_overrides,
    build_stage12_supplement_records,
    discover_c2_anchor_run_ids,
    discover_random_init_run_ids,
    discover_run_ids,
    load_stage9_manifest,
    reanalyze_stage13_expressivity,
    recompute_fisher_from_jsonl,
    spearman_trace_w_vs_layers,
)
from qnc.sweep import merge_overrides
from qnc.metrics import total_within_class_hs_variance, class_means
from qnc.train import assert_physical_rhos, statevectors_to_rhos
import json
from pathlib import Path

import yaml

RESULTS_DIR = Path("results")
CONFIGS_DIR = Path("configs")


def test_parse_seed():
    assert _parse_seed("stage2_vqc_c2_diagnostic_s0_20260710-002127") == 0
    assert _parse_seed("stage5_random_init_s9_20260710-144816") == 9


def test_discover_c2_anchor_and_random_init_counts():
    assert len(discover_c2_anchor_run_ids()) == 5
    assert len(discover_random_init_run_ids()) == 10


def test_discover_run_ids_excludes_aggregate_and_sweeps_dirs():
    run_ids = discover_run_ids()
    assert "stage2_vqc_c2_diagnostic_aggregate" not in run_ids
    assert "sweeps" not in run_ids
    assert "stage2_vqc_c2_diagnostic_s0_20260710-002127" in run_ids


def test_fisher_matches_existing_qnc1_rel_field():
    """SPEC2 §11.1: qnc1_fisher is algebraically identical to the qnc1_rel
    field already logged in Phase 1 JSONLs."""
    run_id = "stage2_vqc_c2_diagnostic_s0_20260710-002127"
    fisher_by_run = recompute_fisher_from_jsonl()
    recomputed = {(r["epoch"], r["split"]): r["qnc1_fisher"] for r in fisher_by_run[run_id]}

    path = RESULTS_DIR / run_id / "metrics.jsonl"
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            key = (r["epoch"], r["split"])
            if key in recomputed and not np.isnan(recomputed[key]):
                assert recomputed[key] == pytest.approx(r["qnc1_rel"], rel=1e-9)


def test_epoch0_reconstruction_matches_logged_qnc1_trace():
    """Seeded reconstruction of the pre-training state must reproduce the
    already-logged epoch-0 qnc1_trace exactly (no new training occurred)."""
    run_id = "stage2_vqc_c2_diagnostic_s0_20260710-002127"
    config = yaml.safe_load(open(CONFIGS_DIR / "stage2_vqc_c2_diagnostic.yaml"))
    model, x_train, y_train = _build_model_and_data(config, seed=0)
    recomputed = _recompute_epoch_witnesses(model, x_train, y_train)

    path = RESULTS_DIR / run_id / "metrics.jsonl"
    with open(path) as f:
        logged_epoch0_train = json.loads(f.readline())
    assert logged_epoch0_train["epoch"] == 0
    assert logged_epoch0_train["split"] == "train"
    assert recomputed["qnc1_trace"] == pytest.approx(logged_epoch0_train["qnc1_trace"], rel=1e-6)
    assert recomputed["qnc1_hs"] == pytest.approx(logged_epoch0_train["qnc1_hs"], rel=1e-6)


def test_l8_anchor_zspace_epoch0_seeded_reconstruction_is_physical():
    """PLAYBOOK3.md Stage 12 Task C(i): the blobs3 seeded-reconstruction path
    (_build_model_and_data's new dataset branch) must produce section 18
    witnesses within their analytic bounds at epoch 0, for the manifest's
    seed-0 L=8 anchor run."""
    base_config = yaml.safe_load(open(CONFIGS_DIR / "stage9_transition.yaml"))
    overrides = _stage9_variant_overrides()["L8"]
    config = merge_overrides(base_config, overrides)

    manifest = load_stage9_manifest()
    assert set(manifest["L8"].keys()) == {"0", "1", "2", "3", "4"}

    model, x_train, y_train = _build_model_and_data(config, seed=0)
    result = _recompute_zspace_witnesses(model, x_train, y_train)

    assert result["nc1_z"] >= 0.0
    assert result["r_z"] >= 0.0
    assert result["d_box"] >= 0.0
    assert -1.0 <= result["equiangle_dev_z"] <= 3.0
    assert 0.0 <= result["fiber_fraction"] <= 1.0 + 1e-9


def test_spearman_trace_w_vs_layers_detects_monotonic_decrease(tmp_path):
    """PREDICTIONS.md Stage 14B HR4: trace_w strictly decreasing with L must
    give rho=-1 (perfect monotonic anti-correlation), p < 0.05."""
    sweep_config_path = tmp_path / "sweep.yaml"
    sweep_config_path.write_text(
        yaml.dump(
            {
                "variants": [
                    {"label": "L2", "value": 2},
                    {"label": "L8", "value": 8},
                    {"label": "L32", "value": 32},
                ]
            }
        )
    )
    sweep_final = {
        "L2": {"0": {"trace_w": 0.30}, "1": {"trace_w": 0.32}},
        "L8": {"0": {"trace_w": 0.10}, "1": {"trace_w": 0.11}},
        "L32": {"0": {"trace_w": 0.02}, "1": {"trace_w": 0.03}},
    }
    result = spearman_trace_w_vs_layers(sweep_final, sweep_config_path=sweep_config_path)
    assert result["rho"] < 0
    assert result["p_value"] < 0.05
    assert result["n"] == 6


def test_total_var_matches_direct_computation_on_real_checkpoint():
    """PREDICTIONS.md Stage 14B Task B: `_recompute_zspace_witnesses`'s
    `total_var` field (v_meas + v_fiber, via the measured-subspace basis
    decomposition) must equal the basis-independent direct computation
    (Parseval's identity) on a real checkpoint -- L8 anchor seed 0's final
    epoch, not synthetic data."""
    base_config = yaml.safe_load(open(CONFIGS_DIR / "stage9_transition.yaml"))
    overrides = _stage9_variant_overrides()["L8"]
    config = merge_overrides(base_config, overrides)

    manifest = load_stage9_manifest()
    run_id = manifest["L8"]["0"]
    model, x_train, y_train = _build_model_and_data(config, seed=0)

    ckpt_dir = RESULTS_DIR / run_id / "checkpoints"
    final_ckpt = sorted(ckpt_dir.glob("epoch_*.pt"))[-1]
    model.load_state_dict(torch.load(final_ckpt, weights_only=True))

    result = _recompute_zspace_witnesses(model, x_train, y_train)

    model.eval()
    with torch.no_grad():
        psi = model.state(x_train).detach().cpu().numpy()
    rhos = statevectors_to_rhos(psi)
    assert_physical_rhos(rhos)
    labels = y_train.detach().cpu().numpy()
    classes = np.unique(labels)
    class_idx = {c: idx for idx, c in enumerate(classes)}
    means, _global_mean = class_means(rhos, labels)
    deviations = np.stack([rhos[i] - means[class_idx[labels[i]]] for i in range(len(labels))])

    direct = total_within_class_hs_variance(deviations)
    assert result["total_var"] == pytest.approx(direct, rel=1e-9)
    assert result["total_var"] == pytest.approx(result["v_meas"] + result["v_fiber"], rel=1e-12)


def test_hr6_purity_overlap_invariant_epoch0_vs_final_one_seed():
    """PREDICTIONS.md Stage 17b HR6: per-class-mean purity and pairwise
    overlap must be invariant (rel. change < 1e-6) across training for a
    reupload=false run -- the same isometry argument HR2 already confirmed
    for total_var. Fast regression check: epoch-0 vs final checkpoint only,
    one seed (the full 5-seed x every-checkpoint sweep is scored once via
    `reanalyze_hr6_hn5_invariance`; results recorded in
    STAGE14B_FINDINGS.md / STAGE15_FINDINGS.md)."""
    base_config = yaml.safe_load(open(CONFIGS_DIR / STAGE14_MECHANISM_BASE_CONFIG))
    overrides = yaml.safe_load(open(STAGE14B_REUPLOAD_FALSE_SWEEP_CONFIG))["variants"][0]["overrides"]
    config = merge_overrides(base_config, overrides)

    run_id = STAGE14B_REUPLOAD_FALSE_RUN_IDS[0]
    model, x_train, y_train = _build_model_and_data(config, seed=0)
    epoch0 = _recompute_zspace_witnesses(model, x_train, y_train)

    ckpt_dir = RESULTS_DIR / run_id / "checkpoints"
    final_ckpt = sorted(ckpt_dir.glob("epoch_*.pt"))[-1]
    model.load_state_dict(torch.load(final_ckpt, weights_only=True))
    final = _recompute_zspace_witnesses(model, x_train, y_train)

    assert _max_relative_change([epoch0["mean_class_purity"], final["mean_class_purity"]]) < 1e-6
    assert _max_relative_change([epoch0["class_mean_overlap"], final["class_mean_overlap"]]) < 1e-6


def test_build_stage12_supplement_records_annotates_e0_and_cosines(tmp_path):
    """PLAYBOOK3.md Stage 13 Task 1: e0_epoch is read from the seed's own
    metrics.jsonl (not recomputed), and pairwise_cos has C(C,2)=3 entries
    for C=3, exactly the ETF-target cosine (-0.5) on an exact C=3 ETF."""
    run_id = "fake_l8_s0"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    with open(run_dir / "metrics.jsonl", "w") as f:
        f.write(json.dumps({"epoch": 0, "e0_epoch": None}) + "\n")
        f.write(json.dumps({"epoch": 3, "e0_epoch": 3}) + "\n")

    l8_records = {
        run_id: [
            {"epoch": 0, "zbar_c": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]},
        ]
    }
    out = build_stage12_supplement_records(l8_records, results_dir=tmp_path)
    rec = out[run_id][0]
    assert rec["e0_epoch"] == 3
    assert len(rec["pairwise_cos"]) == 3
    for c in rec["pairwise_cos"]:
        assert c == pytest.approx(-0.5, abs=1e-10)


def test_reanalyze_stage13_expressivity_plumbing_on_smoke_runs():
    """PLAYBOOK3.md Stage 13 Task 2: end-to-end plumbing check against the
    real (tiny, 3-epoch) smoke-test runs for each readout family, run before
    committing to the full 600-epoch x 20-run sweep. Checks structure and
    physical bounds only -- these are not the real sweep's numbers."""
    manifest = {}
    for label in ("R0", "R1", "R2", "R3"):
        matches = sorted(RESULTS_DIR.glob(f"smoke_stage13_{label}_s0_*"))
        if not matches:
            pytest.skip(f"no smoke_stage13_{label}_s0_* run dir found")
        manifest[label] = {"0": matches[-1].name}

    out = reanalyze_stage13_expressivity(manifest)
    for label in ("R0", "R1", "R2", "R3"):
        rec = out[label]["0"]
        assert rec["readout_family"] == label
        assert rec["m_params_measurement"] == {"R0": 0, "R1": 30, "R2": 57, "R3": 24}[label]
        assert 0.0 <= rec["fiber_fraction"] <= 1.0 + 1e-9
        assert rec["feature_nc1_z"] >= 0.0
        assert np.isfinite(rec["qnc1_fisher"]) or np.isnan(rec["qnc1_fisher"])
        assert 0.0 <= rec["purity_mean"] <= 1.0 + 1e-8
