"""Unit tests for src/qnc/report.py cross-run markdown tables (PLAYBOOK Stage 3/4).

Builds tiny synthetic metrics.jsonl files (2 epochs each) under tmp_path so the
report functions are tested against known values, independent of any real run.
"""

import json
from pathlib import Path

import pytest
from scipy.stats import mannwhitneyu

from qnc.report import (
    build_arm_comparison_table,
    build_cross_config_table,
    build_loss_comparison_table,
    build_parameter_accounting_table,
    build_stage13_expressivity_table,
    build_statistical_comparison_table,
    build_subspace_split_table,
    build_table,
    build_transition_group_statistics,
    build_transition_table,
)

_BASE_FIELDS = {
    "loss": 0.1,
    "acc": 1.0,
    "beta": 10.0,
    "qnc1_rel": 1.0,
    "equiangle_std": 0.0,
    "gram_offdiag_cos": [-1.0],
    "class_mean_norms": [1.0, 1.0],
    "entropy_class": [0.5, 0.5],
    "entropy_classmean": [0.5, 0.5],
    "git_sha": "abc123",
    "wall_time_s": 1.0,
}


def _write_run(
    results_dir: Path,
    run_id: str,
    seed: int,
    e0_epoch: int | None,
    final: dict,
) -> None:
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True)
    records = [
        {
            **_BASE_FIELDS,
            "run_id": run_id,
            "epoch": 0,
            "split": "train",
            "qnc1_trace": 0.9,
            "qnc1_hs": 0.9,
            "equinorm_cv": 0.5,
            "equiangle_dev": 0.5,
            "entropy_mean": 1.0,
            "tpt_reached": False,
            "e0_epoch": None,
            "seed": seed,
        },
        {
            **_BASE_FIELDS,
            "run_id": run_id,
            "epoch": 100,
            "split": "train",
            "tpt_reached": e0_epoch is not None,
            "e0_epoch": e0_epoch,
            "seed": seed,
            **final,
        },
    ]
    with open(run_dir / "metrics.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestBuildTable:
    def test_reads_final_epoch_not_first(self, tmp_path):
        _write_run(
            tmp_path,
            "run_a",
            seed=0,
            e0_epoch=10,
            final={
                "qnc1_trace": 0.01,
                "qnc1_hs": 0.02,
                "equinorm_cv": 0.03,
                "equiangle_dev": 0.04,
                "entropy_mean": 0.9,
            },
        )
        table = build_table(["run_a"], results_dir=tmp_path)
        assert "0.01" in table
        assert "0.9" in table
        assert "0.5" not in table  # epoch-0 values must not leak in

    def test_mean_and_std_row_across_seeds(self, tmp_path):
        _write_run(
            tmp_path, "run_s0", seed=0, e0_epoch=10,
            final={"qnc1_trace": 0.01, "qnc1_hs": 0.01, "equinorm_cv": 0.0,
                   "equiangle_dev": 0.0, "entropy_mean": 1.0},
        )
        _write_run(
            tmp_path, "run_s1", seed=1, e0_epoch=20,
            final={"qnc1_trace": 0.03, "qnc1_hs": 0.03, "equinorm_cv": 0.0,
                   "equiangle_dev": 0.0, "entropy_mean": 1.0},
        )
        table = build_table(["run_s0", "run_s1"], results_dir=tmp_path)
        # mean of qnc1_trace = 0.02, std = 0.01
        assert "0.02" in table
        assert "mean" in table.lower()

    def test_missing_e0_reported_as_no_tpt(self, tmp_path):
        _write_run(
            tmp_path, "run_no_tpt", seed=0, e0_epoch=None,
            final={"qnc1_trace": 0.5, "qnc1_hs": 0.5, "equinorm_cv": 0.5,
                   "equiangle_dev": 0.5, "entropy_mean": 1.0},
        )
        table = build_table(["run_no_tpt"], results_dir=tmp_path)
        assert "None" in table or "N/A" in table


class TestBuildCrossConfigTable:
    def test_one_row_per_variant_mean_across_seeds(self, tmp_path):
        _write_run(
            tmp_path, "run_true_s0", seed=0, e0_epoch=10,
            final={"qnc1_trace": 0.01, "qnc1_hs": 0.01, "equinorm_cv": 0.0,
                   "equiangle_dev": 0.0, "entropy_mean": 1.0},
        )
        _write_run(
            tmp_path, "run_true_s1", seed=1, e0_epoch=10,
            final={"qnc1_trace": 0.03, "qnc1_hs": 0.03, "equinorm_cv": 0.0,
                   "equiangle_dev": 0.0, "entropy_mean": 1.0},
        )
        _write_run(
            tmp_path, "run_false_s0", seed=0, e0_epoch=None,
            final={"qnc1_trace": 0.5, "qnc1_hs": 0.5, "equinorm_cv": 0.5,
                   "equiangle_dev": 0.5, "entropy_mean": 1.0},
        )
        manifest = {
            "entangling_true": {"0": "run_true_s0", "1": "run_true_s1"},
            "entangling_false": {"0": "run_false_s0"},
        }
        table = build_cross_config_table(manifest, results_dir=tmp_path)
        assert "entangling_true" in table
        assert "entangling_false" in table
        assert "0.02" in table  # mean qnc1_trace for entangling_true

    def test_raises_on_empty_manifest(self, tmp_path):
        with pytest.raises(ValueError):
            build_cross_config_table({}, results_dir=tmp_path)


class TestBuildStatisticalComparisonTable:
    """PLAYBOOK Stage 5: real-vs-shuffled and entangled-vs-not need mean+/-std
    per side plus a Mann-Whitney U p-value per metric, from a 2-variant
    manifest shaped exactly like sweep.py's (same as build_cross_config_table)."""

    def _manifest_with_values(self, tmp_path, label_a, vals_a, label_b, vals_b):
        manifest = {label_a: {}, label_b: {}}
        for label, vals in ((label_a, vals_a), (label_b, vals_b)):
            for seed, v in enumerate(vals):
                run_id = f"{label}_s{seed}"
                self._write(
                    tmp_path, run_id, seed,
                    final={"qnc1_trace": v, "qnc1_hs": v, "equinorm_cv": 0.0,
                           "equiangle_dev": 0.0, "entropy_mean": 1.0},
                )
                manifest[label][str(seed)] = run_id
        return manifest

    @staticmethod
    def _write(tmp_path, run_id, seed, final):
        _write_run(tmp_path, run_id, seed=seed, e0_epoch=10, final=final)

    def test_p_value_matches_scipy_directly(self, tmp_path):
        vals_a = [0.01, 0.02, 0.015]
        vals_b = [0.5, 0.6, 0.55]
        manifest = self._manifest_with_values(tmp_path, "group_a", vals_a, "group_b", vals_b)
        table = build_statistical_comparison_table(manifest, "group_a", "group_b", results_dir=tmp_path)

        expected_p = mannwhitneyu(vals_a, vals_b).pvalue
        assert f"{expected_p:.4g}" in table
        assert "group_a" in table and "group_b" in table
        assert "qnc1_trace" in table

    def test_identical_groups_give_p_near_one(self, tmp_path):
        vals = [0.1, 0.2, 0.3]
        manifest = self._manifest_with_values(tmp_path, "same_a", vals, "same_b", vals)
        table = build_statistical_comparison_table(manifest, "same_a", "same_b", results_dir=tmp_path)
        expected_p = mannwhitneyu(vals, vals).pvalue
        assert f"{expected_p:.4g}" in table

    def test_raises_on_missing_label(self, tmp_path):
        manifest = self._manifest_with_values(tmp_path, "group_a", [0.1, 0.2], "group_b", [0.3, 0.4])
        with pytest.raises(KeyError):
            build_statistical_comparison_table(manifest, "group_a", "nonexistent", results_dir=tmp_path)


_TRANSITION_FIELDS = {
    "qnc1_fisher": 0.5,
    "qnc1_ratio": 0.5,
    "overlap_offdiag": 0.1,
    "purity_mean": 0.8,
    "n_params": 24,
    "m_over_mc": 24 / 255,
    "qfi_rank": 20,
}


def _write_transition_run(
    tmp_path: Path,
    run_id: str,
    seed: int,
    e0_epoch: int | None,
    overrides: dict | None = None,
) -> None:
    final = {**_TRANSITION_FIELDS, **(overrides or {})}
    _write_run(
        tmp_path,
        run_id,
        seed=seed,
        e0_epoch=e0_epoch,
        final={
            "qnc1_trace": 0.01,
            "qnc1_hs": 0.01,
            "equinorm_cv": 0.0,
            "equiangle_dev": 0.05,
            "entropy_mean": 1.0,
            **final,
        },
    )


class TestBuildTransitionTable:
    def test_tpt_fraction_and_e0_only_over_reached_seeds(self, tmp_path):
        _write_transition_run(tmp_path, "L2_s0", seed=0, e0_epoch=10)
        _write_transition_run(tmp_path, "L2_s1", seed=1, e0_epoch=None)
        manifest = {"L2": {"0": "L2_s0", "1": "L2_s1"}}
        table = build_transition_table(manifest, results_dir=tmp_path)
        assert "L2" in table
        assert "1/2" in table  # TPT fraction
        assert "10" in table  # E0 mean over the single reached seed

    def test_all_seeds_no_tpt_reports_na_e0(self, tmp_path):
        _write_transition_run(tmp_path, "L2_s0", seed=0, e0_epoch=None)
        manifest = {"L2": {"0": "L2_s0"}}
        table = build_transition_table(manifest, results_dir=tmp_path)
        assert "0/1" in table
        assert "n/a" in table.lower() or "none" in table.lower()

    def test_reports_n_params_m_over_mc_and_qfi_rank(self, tmp_path):
        _write_transition_run(tmp_path, "L2_s0", seed=0, e0_epoch=5, overrides={"n_params": 24, "m_over_mc": 24 / 255, "qfi_rank": 20})
        manifest = {"L2": {"0": "L2_s0"}}
        table = build_transition_table(manifest, results_dir=tmp_path)
        assert "24" in table
        assert "20" in table

    def test_preserves_manifest_label_order(self, tmp_path):
        _write_transition_run(tmp_path, "L32_s0", seed=0, e0_epoch=5)
        _write_transition_run(tmp_path, "L2_s0", seed=0, e0_epoch=5)
        manifest = {"L32": {"0": "L32_s0"}, "L2": {"0": "L2_s0"}}
        table = build_transition_table(manifest, results_dir=tmp_path)
        assert table.index("L32") < table.index("L2")

    def test_raises_on_empty_manifest(self, tmp_path):
        with pytest.raises(ValueError):
            build_transition_table({}, results_dir=tmp_path)


class TestBuildLossComparisonTable:
    """PLAYBOOK2.md Stage 10 CE-vs-MSE arm (SPEC2_ADDENDUM.md §14): all
    collapse witnesses side by side (mean+/-std per loss) plus a Mann-Whitney
    U p-value per witness, and TPT fraction / E0 rows."""

    def _manifest(self, tmp_path, fisher_ce, fisher_mse, e0_ce=5, e0_mse=5):
        manifest = {"ce": {}, "mse": {}}
        for label, vals, e0 in (("ce", fisher_ce, e0_ce), ("mse", fisher_mse, e0_mse)):
            for seed, v in enumerate(vals):
                run_id = f"loss_{label}_s{seed}"
                _write_transition_run(
                    tmp_path, run_id, seed=seed, e0_epoch=e0,
                    overrides={"qnc1_fisher": v},
                )
                manifest[label][str(seed)] = run_id
        return manifest

    def test_p_value_matches_scipy_and_lists_all_witnesses(self, tmp_path):
        fisher_ce = [0.1, 0.2, 0.15]
        fisher_mse = [0.7, 0.8, 0.75]
        manifest = self._manifest(tmp_path, fisher_ce, fisher_mse)
        table = build_loss_comparison_table(manifest, "ce", "mse", results_dir=tmp_path)

        expected_p = mannwhitneyu(fisher_ce, fisher_mse).pvalue
        assert f"{expected_p:.4g}" in table
        for witness in (
            "qnc1_trace", "qnc1_hs", "qnc1_fisher", "qnc1_ratio",
            "overlap_offdiag", "purity_mean", "equinorm_cv",
            "equiangle_dev", "entropy_mean",
        ):
            assert witness in table

    def test_reports_tpt_fraction_and_e0_rows(self, tmp_path):
        manifest = self._manifest(tmp_path, [0.1, 0.2], [0.3, 0.4], e0_ce=10, e0_mse=None)
        table = build_loss_comparison_table(manifest, "ce", "mse", results_dir=tmp_path)
        assert "2/2" in table  # ce TPT fraction
        assert "0/2" in table  # mse TPT fraction
        assert "10" in table  # ce E0 mean
        assert "n/a" in table.lower()  # mse E0 with no reached seeds

    def test_raises_on_missing_label(self, tmp_path):
        manifest = self._manifest(tmp_path, [0.1, 0.2], [0.3, 0.4])
        with pytest.raises(KeyError):
            build_loss_comparison_table(manifest, "ce", "hinge", results_dir=tmp_path)


class TestBuildTransitionGroupStatistics:
    def test_pools_seeds_and_matches_scipy(self, tmp_path):
        low_labels = ["L2", "L4"]
        high_labels = ["L21"]
        low_vals_fisher = {"L2": [0.1, 0.2], "L4": [0.15, 0.25]}
        high_vals_fisher = {"L21": [0.8, 0.9, 0.7]}
        manifest: dict[str, dict[str, str]] = {}
        for label, seed_vals in {**low_vals_fisher, **high_vals_fisher}.items():
            manifest[label] = {}
            for seed, v in enumerate(seed_vals):
                run_id = f"{label}_s{seed}"
                _write_transition_run(
                    tmp_path, run_id, seed=seed, e0_epoch=5,
                    overrides={"qnc1_fisher": v, "equiangle_dev": v},
                )
                manifest[label][str(seed)] = run_id

        table = build_transition_group_statistics(manifest, low_labels, high_labels, results_dir=tmp_path)

        pooled_low = [v for vals in low_vals_fisher.values() for v in vals]
        pooled_high = [v for vals in high_vals_fisher.values() for v in vals]
        expected_p = mannwhitneyu(pooled_low, pooled_high).pvalue
        assert f"{expected_p:.4g}" in table
        assert "qnc1_fisher" in table
        assert "equiangle_dev" in table
        assert "n=4" in table  # pooled low group size (2+2)
        assert "n=3" in table  # pooled high group size


def _synthetic_stage13_records() -> dict[str, dict[str, dict]]:
    """Minimal synthetic PLAYBOOK3.md Stage 13 Task 2 records: 5 seeds per
    arm, with R3 given strictly stronger full-space witnesses than R0 so the
    Mann-Whitney U test has a clear direction to check."""
    out: dict[str, dict[str, dict]] = {}
    arm_bases = {"R0": 0.1, "R1": 0.2, "R2": 0.3, "R3": 0.9}
    m_params = {"R0": 0, "R1": 30, "R2": 57, "R3": 24}
    for label, base in arm_bases.items():
        out[label] = {}
        for seed in range(5):
            out[label][str(seed)] = {
                "m_params_measurement": m_params[label],
                "qnc1_fisher": base + 0.01 * seed,
                "purity_mean": base + 0.01 * seed,
                "overlap_offdiag": base + 0.01 * seed,
                "fiber_fraction": 0.99 - 0.01 * (base + 0.01 * seed),
                "feature_nc1_z": 1.0 - base,
            }
    return out


class TestBuildStage13ExpressivityTable:
    def test_table_has_all_arms_and_significant_r0_vs_r3(self):
        records = _synthetic_stage13_records()
        table = build_stage13_expressivity_table(records)
        for label in ("R0", "R1", "R2", "R3"):
            assert f"| {label} |" in table
        assert "qnc1_fisher" in table
        assert "fiber_fraction" in table
        assert "feature_nc1_z" in table
        # R0 (0.1-0.14) vs R3 (0.9-0.94) full-space witnesses are cleanly
        # separated -> exact Mann-Whitney U p-value at that separation.
        r0_fisher = [records["R0"][str(s)]["qnc1_fisher"] for s in range(5)]
        r3_fisher = [records["R3"][str(s)]["qnc1_fisher"] for s in range(5)]
        expected_p = mannwhitneyu(r0_fisher, r3_fisher).pvalue
        assert f"{expected_p:.4g}" in table


def _synthetic_stage13b_param_records() -> dict[str, dict[str, dict]]:
    """PLAYBOOK3.md Stage 13B parameter-accounting inputs: matched-M sweep's
    actual n_layers/n_params/m_params_measurement (SPEC3_ADDENDUM.md section 19
    header comment)."""
    layers = {"R0": 8, "R1": 5, "R2": 3, "R3": 6}
    n_params = {"R0": 96, "R1": 60, "R2": 36, "R3": 72}
    m_meas = {"R0": 0, "R1": 30, "R2": 57, "R3": 24}
    out: dict[str, dict[str, dict]] = {}
    for label in ("R0", "R1", "R2", "R3"):
        out[label] = {
            str(s): {"n_layers": layers[label], "n_params": n_params[label], "m_params_measurement": m_meas[label]}
            for s in range(5)
        }
    return out


class TestBuildParameterAccountingTable:
    def test_table_reports_total_m_and_brackets_stage9_depths(self):
        records = _synthetic_stage13b_param_records()
        table = build_parameter_accounting_table(records)
        assert "| R1 | 5 | 60 | 30 | 90 |" in table
        assert "| R2 | 3 | 36 | 57 | 93 |" in table
        # R1's L=5 sits strictly between Stage9's L4 and L8 references.
        assert "L4=0.9927" in table
        assert "L8=0.9934" in table
        # R2's L=3 sits strictly between Stage9's L2 and L4 references.
        assert "L2=0.9963" in table


class TestBuildSubspaceSplitTable:
    def test_isotropic_input_gives_concentration_near_one(self):
        # wrow_per_dim == complement_per_dim == meas_per_dim -> both ratios == 1.
        records_by_set = {
            "matched-M": {
                "R1": {
                    str(s): {"wrow_per_dim": 0.01, "complement_per_dim": 0.01, "meas_per_dim": 0.01}
                    for s in range(5)
                }
            }
        }
        table = build_subspace_split_table(records_by_set)
        assert "| matched-M | R1 |" in table
        assert "1 +/- 0" in table  # both ratio columns collapse to exactly 1

    def test_concentrated_wrow_gives_ratio_above_one(self):
        # wrow_per_dim half of meas_per_dim -> concentration factor 2.
        records_by_set = {
            "fixed-depth": {
                "R2": {
                    str(s): {"wrow_per_dim": 0.005, "complement_per_dim": 0.01, "meas_per_dim": 0.01}
                    for s in range(5)
                }
            }
        }
        table = build_subspace_split_table(records_by_set)
        assert "| fixed-depth | R2 |" in table
        assert "2 +/- 0" in table


class TestBuildArmComparisonTable:
    def test_reports_means_and_pvalues_against_baseline(self):
        records_by_arm = {
            "R0": {str(s): {"logit_nc1_z": 0.3 + 0.01 * s} for s in range(5)},
            "R1": {str(s): {"logit_nc1_z": 0.9 + 0.01 * s} for s in range(5)},
        }
        table = build_arm_comparison_table(records_by_arm, fields=("logit_nc1_z",), baseline="R0")
        expected_p = mannwhitneyu(
            [0.3 + 0.01 * s for s in range(5)], [0.9 + 0.01 * s for s in range(5)]
        ).pvalue
        assert "| R0 |" in table
        assert "| R1 |" in table
        assert f"{expected_p:.4g}" in table
