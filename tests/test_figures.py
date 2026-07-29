"""Unit tests for the multi-seed aggregation logic in src/qnc/figures.py (Stage 3,
SPEC.md §8: "seeds aggregated as mean line + shaded +/-1 std band").

Only the pure aggregation helper is checked numerically; make_aggregate_figures
itself is checked for producing the expected PNG files against synthetic
metrics.jsonl fixtures under tmp_path (no assertions on pixel content --
matplotlib output is not unit-tested elsewhere in this codebase either).
"""

import json
from pathlib import Path

from qnc.figures import (
    _aggregate_field,
    find_qfi_saturation_point,
    make_ablation_bound_figure,
    make_aggregate_figures,
    make_etf_formation_figure,
    make_expressivity_figure,
    make_fiber_dip_diagnostic_figure,
    make_loss_comparison_figure,
    make_nc1z_decomposition_figure,
    make_order_parameter_figure,
)

_BASE = {
    "loss": 0.1, "acc": 1.0, "beta": 10.0, "qnc1_rel": 1.0,
    "equiangle_std": 0.0, "gram_offdiag_cos": [-1.0],
    "class_mean_norms": [1.0, 1.0], "entropy_class": [0.5, 0.5],
    "entropy_classmean": [0.5, 0.5], "git_sha": "abc123", "wall_time_s": 1.0,
}


def _write_run(results_dir: Path, run_id: str, seed: int, e0_epoch: int, epochs: list[int], qnc1_trace_values: list[float]) -> None:
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True)
    with open(run_dir / "metrics.jsonl", "w") as f:
        for epoch, qnc1 in zip(epochs, qnc1_trace_values):
            rec = {
                **_BASE,
                "run_id": run_id,
                "epoch": epoch,
                "split": "train",
                "qnc1_trace": qnc1,
                "qnc1_hs": qnc1,
                "equinorm_cv": 0.1,
                "equiangle_dev": 0.1,
                "entropy_mean": 1.0,
                "tpt_reached": True,
                "e0_epoch": e0_epoch,
                "seed": seed,
            }
            f.write(json.dumps(rec) + "\n")


class TestAggregateField:
    def test_mean_and_std_at_common_epochs(self):
        run_a = [{"epoch": 0, "qnc1_trace": 1.0}, {"epoch": 1, "qnc1_trace": 0.5}]
        run_b = [{"epoch": 0, "qnc1_trace": 3.0}, {"epoch": 1, "qnc1_trace": 0.1}]
        epochs, mean, std = _aggregate_field([run_a, run_b], "qnc1_trace")
        assert epochs == [0, 1]
        assert mean[0] == 2.0  # (1.0 + 3.0) / 2
        assert mean[1] == 0.3  # (0.5 + 0.1) / 2
        assert std[0] == 1.0

    def test_intersects_misaligned_epoch_sets(self):
        run_a = [{"epoch": 0, "qnc1_trace": 1.0}, {"epoch": 1, "qnc1_trace": 2.0}, {"epoch": 5, "qnc1_trace": 3.0}]
        run_b = [{"epoch": 0, "qnc1_trace": 1.0}, {"epoch": 5, "qnc1_trace": 3.0}]
        epochs, mean, std = _aggregate_field([run_a, run_b], "qnc1_trace")
        assert epochs == [0, 5]  # epoch 1 dropped: absent from run_b


class TestMakeAggregateFigures:
    def test_writes_expected_files(self, tmp_path):
        _write_run(tmp_path, "exp_s0", seed=0, e0_epoch=2, epochs=[0, 1, 2, 3], qnc1_trace_values=[1.0, 0.5, 0.1, 0.05])
        _write_run(tmp_path, "exp_s1", seed=1, e0_epoch=3, epochs=[0, 1, 2, 3], qnc1_trace_values=[1.2, 0.6, 0.2, 0.06])

        make_aggregate_figures("exp", ["exp_s0", "exp_s1"], results_dir=tmp_path)

        fig_dir = tmp_path / "exp_aggregate" / "figures"
        assert (fig_dir / "training.png").exists()
        assert (fig_dir / "qnc1.png").exists()
        assert (fig_dir / "qnc2.png").exists()
        assert (fig_dir / "entropy.png").exists()


class TestFindQfiSaturationPoint:
    def test_returns_first_crossing_of_95_percent_of_max(self):
        m_over_mc = [0.1, 0.2, 0.3, 0.5]
        qfi_rank_means = [5.0, 10.0, 19.0, 20.0]
        # max=20, threshold=19 -> first m_over_mc with qfi_rank_mean >= 19 is 0.3
        assert find_qfi_saturation_point(m_over_mc, qfi_rank_means) == 0.3

    def test_flat_series_saturates_at_first_point(self):
        m_over_mc = [0.1, 0.2, 0.3]
        qfi_rank_means = [10.0, 10.0, 10.0]
        assert find_qfi_saturation_point(m_over_mc, qfi_rank_means) == 0.1

    def test_monotonic_series_saturates_at_last_point(self):
        m_over_mc = [0.1, 0.2, 0.3]
        qfi_rank_means = [1.0, 5.0, 20.0]
        assert find_qfi_saturation_point(m_over_mc, qfi_rank_means) == 0.3


_TRANSITION_BASE = {
    **_BASE,
    "qnc1_trace": 0.01,
    "qnc1_hs": 0.01,
    "qnc1_fisher": 0.5,
    "qnc1_ratio": 0.5,
    "overlap_offdiag": 0.1,
    "purity_class": [0.8, 0.8],
    "purity_mean": 0.8,
}


def _write_transition_run(
    results_dir: Path, run_id: str, seed: int, n_params: int, m_over_mc: float, qfi_rank: int, e0_epoch: int | None
) -> None:
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True)
    record = {
        **_TRANSITION_BASE,
        "run_id": run_id,
        "epoch": 100,
        "split": "train",
        "equinorm_cv": 0.1,
        "equiangle_dev": 0.1,
        "entropy_mean": 1.0,
        "tpt_reached": e0_epoch is not None,
        "e0_epoch": e0_epoch,
        "seed": seed,
        "n_params": n_params,
        "m_over_mc": m_over_mc,
        "qfi_rank": qfi_rank,
    }
    with open(run_dir / "metrics.jsonl", "w") as f:
        f.write(json.dumps(record) + "\n")


class TestMakeOrderParameterFigure:
    def test_writes_expected_file(self, tmp_path):
        manifest = {}
        for label, n_params, m_over_mc, qfi_rank in [
            ("L2", 24, 24 / 255, 10),
            ("L21", 252, 252 / 255, 200),
        ]:
            manifest[label] = {}
            for seed in (0, 1):
                run_id = f"{label}_s{seed}"
                _write_transition_run(
                    tmp_path, run_id, seed, n_params, m_over_mc, qfi_rank,
                    e0_epoch=5 if label == "L21" else None,
                )
                manifest[label][str(seed)] = run_id

        out_path = tmp_path / "order_parameter.png"
        make_order_parameter_figure(manifest, results_dir=tmp_path, out_path=out_path)
        assert out_path.exists()


class TestMakeLossComparisonFigure:
    """PLAYBOOK2.md Stage 10 CE-vs-MSE figure (SPEC2_ADDENDUM.md §14)."""

    def test_writes_expected_file(self, tmp_path):
        manifest = {"ce": {}, "mse": {}}
        for label in ("ce", "mse"):
            for seed in (0, 1, 2):
                run_id = f"loss_{label}_s{seed}"
                _write_transition_run(
                    tmp_path, run_id, seed, n_params=96, m_over_mc=96 / 255,
                    qfi_rank=90, e0_epoch=5,
                )
                manifest[label][str(seed)] = run_id

        out_path = tmp_path / "loss_comparison.png"
        make_loss_comparison_figure(manifest, results_dir=tmp_path, out_path=out_path)
        assert out_path.exists()


class TestMakeAblationBoundFigure:
    """PLAYBOOK2.md Stage 10 entanglement-x-transition figure: 4 arms
    (L2/L8 x entangling true/false), each annotated with its own logged
    m_over_mc -- the non-entangling arms' values sit against the corrected
    DLA bound 3n=12."""

    def test_writes_expected_file(self, tmp_path):
        arms = {}
        for label, n_params, m in [
            ("L2_ent", 24, 24 / 255),
            ("L2_noent", 24, 24 / 12),
            ("L8_ent", 96, 96 / 255),
            ("L8_noent", 96, 96 / 12),
        ]:
            arms[label] = {}
            for seed in (0, 1):
                run_id = f"abl_{label}_s{seed}"
                _write_transition_run(
                    tmp_path, run_id, seed, n_params=n_params, m_over_mc=m,
                    qfi_rank=12, e0_epoch=5 if "L8" in label else None,
                )
                arms[label][str(seed)] = run_id

        out_path = tmp_path / "ablation_bounds.png"
        make_ablation_bound_figure(arms, results_dir=tmp_path, out_path=out_path)
        assert out_path.exists()


def _synthetic_supplement_records(n_seeds: int = 2) -> dict[str, list[dict]]:
    """Minimal synthetic PLAYBOOK3.md Stage 13 Task 1 supplement records:
    two epochs per seed with every field the three supplement figures read."""
    records = {}
    for seed in range(n_seeds):
        records[f"seed{seed}"] = [
            {
                "epoch": 0,
                "equiangle_dev_z": 1.0,
                "pairwise_cos": [-0.2, -0.2, -0.2],
                "e0_epoch": 5,
                "trace_w": 10.0,
                "trace_b": 1.0,
                "v_fiber": 0.95,
            },
            {
                "epoch": 10,
                "equiangle_dev_z": 0.05,
                "pairwise_cos": [-0.49, -0.51, -0.5],
                "e0_epoch": 5,
                "trace_w": 0.1,
                "trace_b": 1.0,
                "v_fiber": 0.89,
            },
        ]
    return records


class TestMakeStage13SupplementFigures:
    """PLAYBOOK3.md Stage 13 Task 1: the three Stage 12 diagnostic supplements."""

    def test_etf_formation_writes_expected_file(self, tmp_path):
        out_path = tmp_path / "etf_formation.png"
        make_etf_formation_figure(_synthetic_supplement_records(), out_path=out_path)
        assert out_path.exists()

    def test_nc1z_decomposition_writes_expected_file(self, tmp_path):
        out_path = tmp_path / "nc1z_decomposition.png"
        make_nc1z_decomposition_figure(_synthetic_supplement_records(), out_path=out_path)
        assert out_path.exists()

    def test_fiber_dip_diagnostic_writes_expected_file(self, tmp_path):
        null_records = {f"null{i}": [{"v_fiber": 0.94}] for i in range(3)}
        out_path = tmp_path / "fiber_dip_diagnostic.png"
        make_fiber_dip_diagnostic_figure(_synthetic_supplement_records(), null_records, out_path=out_path)
        assert out_path.exists()


class TestMakeExpressivityFigure:
    """PLAYBOOK3.md Stage 13 Task 2 headline figure."""

    def test_writes_expected_file(self, tmp_path):
        stage13_records = {}
        for label in ("R0", "R1", "R2", "R3"):
            stage13_records[label] = {
                str(seed): {
                    "qnc1_fisher": 0.1,
                    "purity_mean": 0.5,
                    "overlap_offdiag": 0.1,
                    "fiber_fraction": 0.99,
                    "feature_nc1_z": 0.5,
                }
                for seed in range(5)
            }
        out_path = tmp_path / "expressivity.png"
        make_expressivity_figure(stage13_records, out_path=out_path)
        assert out_path.exists()

    def test_writes_expected_file_with_isotropic_null(self, tmp_path):
        # PLAYBOOK3.md Stage 13B: isotropic_fiber_null present -> null line
        # drawn on the fiber_fraction panel without erroring.
        stage13_records = {}
        for label in ("R0", "R1", "R2", "R3"):
            stage13_records[label] = {
                str(seed): {
                    "qnc1_fisher": 0.1,
                    "purity_mean": 0.5,
                    "overlap_offdiag": 0.1,
                    "fiber_fraction": 0.99,
                    "feature_nc1_z": 0.5,
                    "isotropic_fiber_null": 0.98,
                }
                for seed in range(5)
            }
        out_path = tmp_path / "expressivity_null.png"
        make_expressivity_figure(stage13_records, out_path=out_path)
        assert out_path.exists()
