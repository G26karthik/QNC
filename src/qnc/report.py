"""Cross-run analysis tables.

Stage 3 (PLAYBOOK.md): build_table reads the final logged epoch of each run's
metrics.jsonl (SPEC.md §7) and prints a per-seed + mean+/-std markdown table
of E0, qnc1_trace, qnc1_hs, equinorm_cv, equiangle_dev, entropy_mean.

Stage 4: build_cross_config_table reads a sweep manifest ({label: {seed:
run_id}}, see sweep.py) and prints one row per swept variant, aggregating the
same final-epoch metrics across that variant's seeds.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

_METRIC_FIELDS = ("qnc1_trace", "qnc1_hs", "equinorm_cv", "equiangle_dev", "entropy_mean")


def _load_final_record(run_id: str, results_dir: str | Path, split: str = "train") -> dict:
    path = Path(results_dir) / run_id / "metrics.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    split_records = [r for r in records if r["split"] == split]
    return split_records[-1]


def build_table(run_ids: list[str], results_dir: str | Path = "results") -> str:
    """Per-seed table for a single config's multi-seed campaign (Stage 3)."""
    if not run_ids:
        raise ValueError("build_table requires at least one run_id")

    finals = [_load_final_record(run_id, results_dir) for run_id in run_ids]

    header = "| seed | E0 | qnc1_trace | qnc1_hs | equinorm_cv | equiangle_dev | entropy_mean |"
    sep = "|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in finals:
        e0 = r["e0_epoch"] if r["e0_epoch"] is not None else "None (no TPT)"
        lines.append(
            f"| {r['seed']} | {e0} | {r['qnc1_trace']:.6g} | {r['qnc1_hs']:.6g} | "
            f"{r['equinorm_cv']:.6g} | {r['equiangle_dev']:.6g} | {r['entropy_mean']:.6g} |"
        )

    mean_cells = []
    for field in _METRIC_FIELDS:
        vals = np.array([r[field] for r in finals], dtype=float)
        mean_cells.append(f"{vals.mean():.6g} +/- {vals.std():.6g}")
    lines.append(f"| **mean+/-std** | -- | {' | '.join(mean_cells)} |")

    return "\n".join(lines)


def build_cross_config_table(
    manifest: dict[str, dict[str, str]], results_dir: str | Path = "results"
) -> str:
    """One row per swept variant, mean+/-std of final-epoch metrics across its seeds.

    `manifest` is `{variant_label: {seed_str: run_id}}` as written by sweep.py.
    """
    if not manifest:
        raise ValueError("build_cross_config_table requires a non-empty manifest")

    header = "| variant | n_seeds | tpt_reached | qnc1_trace | qnc1_hs | equinorm_cv | equiangle_dev | entropy_mean |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [header, sep]

    for label, seed_to_run in manifest.items():
        finals = [_load_final_record(run_id, results_dir) for run_id in seed_to_run.values()]
        n_tpt = sum(1 for r in finals if r["tpt_reached"])
        tpt_str = f"{n_tpt}/{len(finals)}"
        cells = []
        for field in _METRIC_FIELDS:
            vals = np.array([r[field] for r in finals], dtype=float)
            cells.append(f"{vals.mean():.6g} +/- {vals.std():.6g}")
        lines.append(f"| {label} | {len(finals)} | {tpt_str} | {' | '.join(cells)} |")

    return "\n".join(lines)


def build_statistical_comparison_table(
    manifest: dict[str, dict[str, str]],
    label_a: str,
    label_b: str,
    results_dir: str | Path = "results",
) -> str:
    """Stage 5 (PLAYBOOK.md): mean+/-std per side plus a Mann-Whitney U
    p-value per metric, comparing two variants of a manifest shaped like
    sweep.py's (`{variant_label: {seed_str: run_id}}`). Used for both
    real-vs-shuffled-labels and entangled-vs-not comparisons.
    """
    if label_a not in manifest:
        raise KeyError(f"label_a {label_a!r} not in manifest")
    if label_b not in manifest:
        raise KeyError(f"label_b {label_b!r} not in manifest")

    finals_a = [_load_final_record(run_id, results_dir) for run_id in manifest[label_a].values()]
    finals_b = [_load_final_record(run_id, results_dir) for run_id in manifest[label_b].values()]

    header = f"| metric | {label_a} (mean+/-std) | {label_b} (mean+/-std) | Mann-Whitney U p-value |"
    sep = "|---|---|---|---|"
    lines = [header, sep]

    for field in _METRIC_FIELDS:
        vals_a = np.array([r[field] for r in finals_a], dtype=float)
        vals_b = np.array([r[field] for r in finals_b], dtype=float)
        p_value = mannwhitneyu(vals_a, vals_b).pvalue
        lines.append(
            f"| {field} | {vals_a.mean():.6g} +/- {vals_a.std():.6g} | "
            f"{vals_b.mean():.6g} +/- {vals_b.std():.6g} | {p_value:.4g} |"
        )

    return "\n".join(lines)


_TRANSITION_FIELDS = (
    "qnc1_fisher",
    "qnc1_ratio",
    "overlap_offdiag",
    "purity_mean",
    "equiangle_dev",
    "entropy_mean",
)


def build_transition_table(manifest: dict[str, dict[str, str]], results_dir: str | Path = "results") -> str:
    """PLAYBOOK2.md Stage 9 per-L table (SPEC2_ADDENDUM.md section 12.3).

    One row per swept L (manifest label, in insertion order): n_params,
    m_over_mc, TPT fraction (over all seeds), E0 mean+/-std (over only the
    seeds that reached TPT -- "n/a" if none did), then mean+/-std across all
    seeds of qnc1_fisher, qnc1_ratio, overlap_offdiag, purity_mean,
    equiangle_dev, entropy_mean, qfi_rank (final epoch).
    """
    if not manifest:
        raise ValueError("build_transition_table requires a non-empty manifest")

    header = (
        "| L | n_params | m_over_mc | n_seeds | tpt_fraction | E0 (mean+/-std) | "
        + " | ".join(_TRANSITION_FIELDS)
        + " | qfi_rank |"
    )
    sep = "|" + "---|" * (7 + len(_TRANSITION_FIELDS))
    lines = [header, sep]

    for label, seed_to_run in manifest.items():
        finals = [_load_final_record(run_id, results_dir) for run_id in seed_to_run.values()]
        n_params = finals[0]["n_params"]
        m_over_mc = finals[0]["m_over_mc"]
        reached = [r for r in finals if r["tpt_reached"]]
        tpt_str = f"{len(reached)}/{len(finals)}"
        if reached:
            e0_vals = np.array([r["e0_epoch"] for r in reached], dtype=float)
            e0_str = f"{e0_vals.mean():.6g} +/- {e0_vals.std():.6g}"
        else:
            e0_str = "n/a"

        cells = []
        for field in _TRANSITION_FIELDS:
            vals = np.array([r[field] for r in finals], dtype=float)
            cells.append(f"{vals.mean():.6g} +/- {vals.std():.6g}")
        qfi_vals = np.array([r["qfi_rank"] for r in finals], dtype=float)
        qfi_str = f"{qfi_vals.mean():.6g} +/- {qfi_vals.std():.6g}"

        lines.append(
            f"| {label} | {n_params} | {m_over_mc:.6g} | {len(finals)} | {tpt_str} | {e0_str} | "
            + " | ".join(cells)
            + f" | {qfi_str} |"
        )

    return "\n".join(lines)


def build_transition_group_statistics(
    manifest: dict[str, dict[str, str]],
    low_labels: list[str],
    high_labels: list[str],
    results_dir: str | Path = "results",
) -> str:
    """PLAYBOOK2.md Stage 9: Mann-Whitney U on final qnc1_fisher and
    equiangle_dev, pooling seeds across `low_labels` (underparameterized, e.g.
    L<=12) vs `high_labels` (overparameterized, e.g. L>=21).
    """
    low_finals = [
        _load_final_record(run_id, results_dir)
        for label in low_labels
        for run_id in manifest[label].values()
    ]
    high_finals = [
        _load_final_record(run_id, results_dir)
        for label in high_labels
        for run_id in manifest[label].values()
    ]

    header = (
        f"| metric | underparam (n={len(low_finals)}, mean+/-std) | "
        f"overparam (n={len(high_finals)}, mean+/-std) | Mann-Whitney U p-value |"
    )
    sep = "|---|---|---|---|"
    lines = [header, sep]

    for field in ("qnc1_fisher", "equiangle_dev"):
        vals_low = np.array([r[field] for r in low_finals], dtype=float)
        vals_high = np.array([r[field] for r in high_finals], dtype=float)
        p_value = mannwhitneyu(vals_low, vals_high).pvalue
        lines.append(
            f"| {field} | {vals_low.mean():.6g} +/- {vals_low.std():.6g} | "
            f"{vals_high.mean():.6g} +/- {vals_high.std():.6g} | {p_value:.4g} |"
        )

    return "\n".join(lines)


_LOSS_COMPARISON_FIELDS = (
    "qnc1_trace",
    "qnc1_hs",
    "qnc1_fisher",
    "qnc1_ratio",
    "overlap_offdiag",
    "purity_mean",
    "equinorm_cv",
    "equiangle_dev",
    "entropy_mean",
)


def build_loss_comparison_table(
    manifest: dict[str, dict[str, str]],
    label_a: str = "ce",
    label_b: str = "mse",
    results_dir: str | Path = "results",
) -> str:
    """PLAYBOOK2.md Stage 10 CE-vs-MSE arm (SPEC2_ADDENDUM.md §14).

    MSE is the loss Du et al. (2023)'s theorem covers; this table puts every
    collapse witness side by side: TPT fraction and E0 rows first, then one
    row per witness with mean+/-std for each loss and a Mann-Whitney U
    p-value across the pooled seeds.
    """
    if label_a not in manifest:
        raise KeyError(f"label_a {label_a!r} not in manifest")
    if label_b not in manifest:
        raise KeyError(f"label_b {label_b!r} not in manifest")

    finals_a = [_load_final_record(run_id, results_dir) for run_id in manifest[label_a].values()]
    finals_b = [_load_final_record(run_id, results_dir) for run_id in manifest[label_b].values()]

    def _tpt_cell(finals: list[dict]) -> str:
        return f"{sum(1 for r in finals if r['tpt_reached'])}/{len(finals)}"

    def _e0_cell(finals: list[dict]) -> str:
        reached = [r["e0_epoch"] for r in finals if r["tpt_reached"]]
        if not reached:
            return "n/a"
        vals = np.array(reached, dtype=float)
        return f"{vals.mean():.6g} +/- {vals.std():.6g}"

    header = f"| witness | {label_a} (mean+/-std) | {label_b} (mean+/-std) | Mann-Whitney U p-value |"
    sep = "|---|---|---|---|"
    lines = [
        header,
        sep,
        f"| tpt_fraction | {_tpt_cell(finals_a)} | {_tpt_cell(finals_b)} | -- |",
        f"| E0 | {_e0_cell(finals_a)} | {_e0_cell(finals_b)} | -- |",
    ]

    for field in _LOSS_COMPARISON_FIELDS:
        vals_a = np.array([r[field] for r in finals_a], dtype=float)
        vals_b = np.array([r[field] for r in finals_b], dtype=float)
        p_value = mannwhitneyu(vals_a, vals_b).pvalue
        lines.append(
            f"| {field} | {vals_a.mean():.6g} +/- {vals_a.std():.6g} | "
            f"{vals_b.mean():.6g} +/- {vals_b.std():.6g} | {p_value:.4g} |"
        )

    return "\n".join(lines)


_STAGE13_TABLE_FIELDS = (
    "qnc1_fisher",
    "purity_mean",
    "overlap_offdiag",
    "fiber_fraction",
    "feature_nc1_z",
)


def build_stage13_expressivity_table(
    stage13_records: dict[str, dict[str, dict]],
    order: tuple[str, ...] = ("R0", "R1", "R2", "R3"),
) -> str:
    """PLAYBOOK3.md Stage 13 Task 2: per-arm table (mean+/-std across 5 seeds)
    of full-space witnesses (qnc1_fisher, purity_mean, overlap_offdiag),
    fiber_fraction, and feature-space NC1_z, plus Mann-Whitney U R0-vs-R3 on
    each full-space witness (SPEC3_ADDENDUM.md section 19's headline test:
    do full-space witnesses strengthen monotonically as measurement
    expressivity increases)."""
    header = "| arm | m_params_measurement | " + " | ".join(_STAGE13_TABLE_FIELDS) + " |"
    sep = "|---|---|" + "---|" * len(_STAGE13_TABLE_FIELDS)
    lines = [header, sep]

    per_arm_vals: dict[str, dict[str, np.ndarray]] = {}
    for label in order:
        recs = list(stage13_records[label].values())
        m_params = recs[0]["m_params_measurement"]
        per_arm_vals[label] = {
            field: np.array([r[field] for r in recs], dtype=float) for field in _STAGE13_TABLE_FIELDS
        }
        cells = [f"{per_arm_vals[label][field].mean():.6g} +/- {per_arm_vals[label][field].std():.6g}" for field in _STAGE13_TABLE_FIELDS]
        lines.append(f"| {label} | {m_params} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("Mann-Whitney U, R0 vs R1/R2/R3, full-space witnesses:")
    lines.append("")
    lines.append("| witness | R0 vs R1 | R0 vs R2 | R0 vs R3 |")
    lines.append("|---|---|---|---|")
    for field in ("qnc1_fisher", "purity_mean", "overlap_offdiag"):
        cells = [
            f"{mannwhitneyu(per_arm_vals['R0'][field], per_arm_vals[other][field]).pvalue:.4g}"
            for other in ("R1", "R2", "R3")
        ]
        lines.append(f"| {field} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


# Stage 9 L-sweep final-epoch fiber_fraction, per STAGE12_FINDINGS.md's
# "Stage 9 L-sweep, full-depth context" table -- Phase 2 reference values used
# to quantify the Stage 13 matched-M depth confound (PLAYBOOK3.md Stage 13B
# Task A).
STAGE9_FIBER_FRACTION_BY_L = {
    2: 0.9963,
    4: 0.9927,
    8: 0.9934,
    12: 0.9933,
    16: 0.9949,
    21: 0.9954,
    26: 0.9951,
    32: 0.9959,
}


def build_parameter_accounting_table(
    stage13_records: dict[str, dict[str, dict]], order: tuple[str, ...] = ("R0", "R1", "R2", "R3")
) -> str:
    """PLAYBOOK3.md Stage 13B Task A: circuit params (`n_params`) vs
    head/measurement params (`m_params_measurement`) per arm, with the
    nearest Stage 9 depth-sweep fiber_fraction values bracketing each arm's
    circuit depth -- quantifies how much of the matched-M sweep's R0>R1>R2
    weakening is attributable to depth alone versus readout family."""
    header = "| arm | n_layers | circuit params | head/measurement params | total M | nearest Stage9 depths (fiber_fraction) |"
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]
    depths = sorted(STAGE9_FIBER_FRACTION_BY_L)
    for label in order:
        recs = list(stage13_records[label].values())
        n_layers = recs[0]["n_layers"]
        n_params = recs[0]["n_params"]
        m_meas = recs[0]["m_params_measurement"]
        total_m = n_params + m_meas
        lower_candidates = [d for d in depths if d <= n_layers]
        upper_candidates = [d for d in depths if d >= n_layers]
        lower = max(lower_candidates) if lower_candidates else depths[0]
        upper = min(upper_candidates) if upper_candidates else depths[-1]
        if lower == upper:
            ref = f"L{lower}={STAGE9_FIBER_FRACTION_BY_L[lower]:.4g}"
        else:
            ref = f"L{lower}={STAGE9_FIBER_FRACTION_BY_L[lower]:.4g} / L{upper}={STAGE9_FIBER_FRACTION_BY_L[upper]:.4g}"
        lines.append(f"| {label} | {n_layers} | {n_params} | {m_meas} | {total_m} | {ref} |")
    return "\n".join(lines)


def build_subspace_split_table(records_by_set: dict[str, dict[str, dict[str, dict]]]) -> str:
    """PLAYBOOK3.md Stage 13B H1: per (checkpoint-set, arm) mean+/-std of the
    loss-visible W-row-space per-dim variance vs its in-span complement vs
    the isotropic (whole-measured-span) per-dim rate, plus the concentration
    factor (meas_per_dim / wrow_per_dim, >1 means the head-row directions are
    more collapsed than the measured-span average) and the isotropic-level
    factor (complement_per_dim / meas_per_dim, ~1 means the complement sits
    at the isotropic rate).

    `records_by_set`: {set_label: {arm_label: {seed: record}}}; records must
    carry wrow_per_dim/complement_per_dim/meas_per_dim (R1/R2 only, from
    `reanalyze._recompute_stage13_zspace_witnesses`)."""
    header = (
        "| set | arm | wrow_per_dim | complement_per_dim | meas_per_dim | "
        "concentration (meas/wrow) | isotropic-level (complement/meas) |"
    )
    sep = "|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for set_label, arms in records_by_set.items():
        for arm_label, seed_map in arms.items():
            recs = list(seed_map.values())
            wrow = np.array([r["wrow_per_dim"] for r in recs], dtype=float)
            comp = np.array([r["complement_per_dim"] for r in recs], dtype=float)
            meas = np.array([r["meas_per_dim"] for r in recs], dtype=float)
            conc = meas / wrow
            iso = comp / meas
            lines.append(
                f"| {set_label} | {arm_label} | {wrow.mean():.6g} +/- {wrow.std():.6g} | "
                f"{comp.mean():.6g} +/- {comp.std():.6g} | {meas.mean():.6g} +/- {meas.std():.6g} | "
                f"{conc.mean():.4g} +/- {conc.std():.4g} | {iso.mean():.4g} +/- {iso.std():.4g} |"
            )
    return "\n".join(lines)


def build_arm_comparison_table(
    records_by_arm: dict[str, dict[str, dict]], fields: tuple[str, ...], baseline: str = "R0"
) -> str:
    """Generic per-arm mean+/-std table for `fields`, plus Mann-Whitney U
    p-values comparing `baseline` against every other arm in
    `records_by_arm`. Used for PLAYBOOK3.md Stage 13B H2 (full-space
    witnesses + fiber_excess_above_null, fixed-depth R1/R2 vs R0) and H3
    (logit_nc1_z invariance, matched-M and fixed-depth sets)."""
    header = "| arm | " + " | ".join(fields) + " |"
    sep = "|---|" + "---|" * len(fields)
    lines = [header, sep]
    per_arm_vals: dict[str, dict[str, np.ndarray]] = {}
    for label, seed_map in records_by_arm.items():
        recs = list(seed_map.values())
        per_arm_vals[label] = {f: np.array([r[f] for r in recs], dtype=float) for f in fields}
        cells = [f"{per_arm_vals[label][f].mean():.6g} +/- {per_arm_vals[label][f].std():.6g}" for f in fields]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append(f"Mann-Whitney U, {baseline} vs each other arm:")
    lines.append("")
    lines.append("| arm | " + " | ".join(f"{f} p-value" for f in fields) + " |")
    lines.append("|---|" + "---|" * len(fields))
    for label in records_by_arm:
        if label == baseline:
            continue
        cells = [
            f"{mannwhitneyu(per_arm_vals[baseline][f], per_arm_vals[label][f]).pvalue:.4g}" for f in fields
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_tpt_fraction_table(tpt_fractions: dict[str, float], n_seeds: int) -> str:
    """PREDICTIONS.md Stage 14 arm 4 (transition localization, L in
    {3,5,6}): one row per arm with the fraction of `n_seeds` seeds that
    reached TPT (tpt_reached at the final logged epoch), in `tpt_fractions`
    insertion order (reanalyze.compute_tpt_fractions)."""
    header = "| arm | n_seeds | TPT fraction |"
    sep = "|---|---|---|"
    lines = [header, sep]
    for label, fraction in tpt_fractions.items():
        lines.append(f"| {label} | {n_seeds} | {fraction:.4g} |")
    return "\n".join(lines)


def main() -> None:
    """CLI: ``python -m qnc.report <run_id> [<run_id> ...]`` (Stage 3 per-seed table)."""
    import argparse

    parser = argparse.ArgumentParser(description="Print a per-seed markdown table from metrics.jsonl files.")
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    print(build_table(args.run_ids, args.results_dir))


if __name__ == "__main__":
    main()
