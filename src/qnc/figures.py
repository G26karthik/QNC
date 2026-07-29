"""Figure generation.

Implements SPEC.md §8: make_figures(run_id) reads results/<run_id>/metrics.jsonl
and writes results/<run_id>/figures/. Pure function of the JSONL; never plots
from in-memory training state (CLAUDE.md invariant 4).

Stage 1: figures 1-3 only (training, qnc1, qnc2); entropy/beta panels are
quantum-only and skipped when those fields are null.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_records(run_id: str, results_dir: str | Path = "results") -> list[dict]:
    path = Path(results_dir) / run_id / "metrics.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _training_figure(train_r: list[dict], test_r: list[dict], e0: int | None, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    ax1.plot([r["epoch"] for r in train_r], [r["loss"] for r in train_r], label="train loss")
    if test_r:
        ax1.plot([r["epoch"] for r in test_r], [r["loss"] for r in test_r], label="test loss")
    ax1.set_ylabel("loss")
    ax1.legend()

    ax2.plot([r["epoch"] for r in train_r], [r["acc"] for r in train_r], label="train acc")
    if test_r:
        ax2.plot([r["epoch"] for r in test_r], [r["acc"] for r in test_r], label="test acc")
    ax2.set_ylabel("accuracy")
    ax2.set_xlabel("epoch")
    ax2.legend()

    if e0 is not None:
        ax1.axvline(e0, color="black", linestyle="--", linewidth=1)
        ax2.axvline(e0, color="black", linestyle="--", linewidth=1)

    fig.suptitle("Training dynamics")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _qnc1_figure(train_r: list[dict], e0: int | None, out_path: Path) -> None:
    epochs = [r["epoch"] for r in train_r]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [r["qnc1_trace"] for r in train_r], marker="o", markersize=2, label="qnc1_trace")
    if train_r and train_r[0].get("qnc1_hs") is not None:
        ax.plot(epochs, [r["qnc1_hs"] for r in train_r], marker="o", markersize=2, label="qnc1_hs")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("QNC1 (log scale)")
    if e0 is not None:
        ax.axvline(e0, color="black", linestyle="--", linewidth=1, label="E0")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _qnc2_figure(train_r: list[dict], etf_target: float, out_path: Path) -> None:
    epochs = [r["epoch"] for r in train_r]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))

    ax1.plot(epochs, [r["equinorm_cv"] for r in train_r], label="equinorm_cv")
    ax1.plot(epochs, [r["equiangle_dev"] for r in train_r], label="equiangle_dev")
    ax1.set_yscale("symlog", linthresh=1e-3)
    ax1.axhline(0.0, color="gray", linestyle="-", linewidth=1)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("(symlog)")
    ax1.legend()

    num_pairs = len(train_r[0]["gram_offdiag_cos"]) if train_r else 0
    for k in range(num_pairs):
        ax2.plot(epochs, [r["gram_offdiag_cos"][k] for r in train_r], alpha=0.7)
    ax2.axhline(etf_target, color="black", linestyle="--", linewidth=1, label="etf_target")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("cos_theta (pairwise)")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _entropy_figure(train_r: list[dict], out_path: Path) -> None:
    epochs = [r["epoch"] for r in train_r]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [r["entropy_mean"] for r in train_r], color="black", linewidth=2, label="entropy_mean")
    num_classes = len(train_r[0]["entropy_class"]) if train_r else 0
    for c in range(num_classes):
        ax.plot(epochs, [r["entropy_class"][c] for r in train_r], alpha=0.6, label=f"class {c}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("entanglement entropy S(rho_A) [bits]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _beta_figure(train_r: list[dict], out_path: Path) -> None:
    epochs = [r["epoch"] for r in train_r]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [r["beta"] for r in train_r])
    ax.set_xlabel("epoch")
    ax.set_ylabel("beta (inverse temperature)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _cos_matrix_from_offdiag(offdiag: list[float], num_classes: int):
    import numpy as np

    mat = np.eye(num_classes)
    k = 0
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            mat[i, j] = offdiag[k]
            mat[j, i] = offdiag[k]
            k += 1
    return mat


def _gram_triptych_figure(train_r: list[dict], e0: int | None, num_classes: int, out_path: Path) -> None:
    if not train_r:
        return
    init_r = train_r[0]
    e0_r = next((r for r in train_r if e0 is not None and r["epoch"] == e0), None)
    final_r = train_r[-1]
    panels = [("init", init_r), ("E0", e0_r), ("final", final_r)]
    panels = [(name, r) for name, r in panels if r is not None]

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (name, r) in zip(axes, panels):
        mat = _cos_matrix_from_offdiag(r["gram_offdiag_cos"], num_classes)
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(f"{name} (epoch {r['epoch']})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_figures(run_id: str, results_dir: str | Path = "results") -> None:
    """Implements SPEC.md §8 figures, regenerated purely from metrics.jsonl.

    Figures 1-3 (training, qnc1, qnc2) apply to both classical and quantum
    runs. Figures 4-6 (entropy, gram_triptych, beta) are quantum-only and
    skipped when those fields are null (classical runs).
    """
    records = _load_records(run_id, results_dir)
    train_r = [r for r in records if r["split"] == "train"]
    test_r = [r for r in records if r["split"] == "test"]

    e0 = next((r["e0_epoch"] for r in train_r if r["e0_epoch"] is not None), None)
    num_classes = len(train_r[0]["class_mean_norms"]) if train_r else 0
    etf_target = -1.0 / (num_classes - 1) if num_classes > 1 else None

    fig_dir = Path(results_dir) / run_id / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    _training_figure(train_r, test_r, e0, fig_dir / "training.png")
    _qnc1_figure(train_r, e0, fig_dir / "qnc1.png")
    if etf_target is not None:
        _qnc2_figure(train_r, etf_target, fig_dir / "qnc2.png")

    is_quantum = bool(train_r) and train_r[0].get("entropy_mean") is not None
    if is_quantum:
        _entropy_figure(train_r, fig_dir / "entropy.png")
        _beta_figure(train_r, fig_dir / "beta.png")
        _gram_triptych_figure(train_r, e0, num_classes, fig_dir / "gram_triptych.png")


def _aggregate_field(runs_train_records: list[list[dict]], field: str) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Mean +/- std of `field` across seeds, aligned on epochs common to every run
    (SPEC.md §8: "seeds aggregated as mean line + shaded +/-1 std band")."""
    per_run_maps = [{r["epoch"]: r[field] for r in recs} for recs in runs_train_records]
    common_epochs = sorted(set.intersection(*(set(m) for m in per_run_maps)))
    values = np.array([[m[e] for m in per_run_maps] for e in common_epochs])
    return common_epochs, values.mean(axis=1), values.std(axis=1)


def make_aggregate_figures(experiment: str, run_ids: list[str], results_dir: str | Path = "results") -> None:
    """Multi-seed campaign figures (Stage 3, SPEC.md §8): mean line + shaded
    +/-1 std band per metric, aligned across seeds on common epochs. Writes
    results/<experiment>_aggregate/figures/*.png. Per-seed figures (including
    gram_triptych) are produced separately by make_figures for each run_id.
    """
    runs_train = []
    e0s = []
    for run_id in run_ids:
        records = _load_records(run_id, results_dir)
        train_r = [r for r in records if r["split"] == "train"]
        runs_train.append(train_r)
        e0 = next((r["e0_epoch"] for r in train_r if r["e0_epoch"] is not None), None)
        if e0 is not None:
            e0s.append(e0)
    e0_mean = round(sum(e0s) / len(e0s)) if e0s else None

    fig_dir = Path(results_dir) / f"{experiment}_aggregate" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    n_seeds = len(run_ids)

    epochs, loss_mean, loss_std = _aggregate_field(runs_train, "loss")
    _, acc_mean, acc_std = _aggregate_field(runs_train, "acc")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    ax1.plot(epochs, loss_mean, label="train loss (mean)")
    ax1.fill_between(epochs, loss_mean - loss_std, loss_mean + loss_std, alpha=0.3)
    ax1.set_ylabel("loss")
    ax1.legend()
    ax2.plot(epochs, acc_mean, label="train acc (mean)")
    ax2.fill_between(epochs, acc_mean - acc_std, acc_mean + acc_std, alpha=0.3)
    ax2.set_ylabel("accuracy")
    ax2.set_xlabel("epoch")
    ax2.legend()
    if e0_mean is not None:
        ax1.axvline(e0_mean, color="black", linestyle="--", linewidth=1)
        ax2.axvline(e0_mean, color="black", linestyle="--", linewidth=1)
    fig.suptitle(f"{experiment}: training dynamics (mean +/- 1 std, n={n_seeds} seeds)")
    fig.tight_layout()
    fig.savefig(fig_dir / "training.png")
    plt.close(fig)

    epochs, trace_mean, trace_std = _aggregate_field(runs_train, "qnc1_trace")
    _, hs_mean, hs_std = _aggregate_field(runs_train, "qnc1_hs")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, trace_mean, label="qnc1_trace (mean)")
    ax.fill_between(epochs, np.clip(trace_mean - trace_std, 1e-12, None), trace_mean + trace_std, alpha=0.3)
    ax.plot(epochs, hs_mean, label="qnc1_hs (mean)")
    ax.fill_between(epochs, np.clip(hs_mean - hs_std, 1e-12, None), hs_mean + hs_std, alpha=0.3)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("QNC1 (log scale)")
    if e0_mean is not None:
        ax.axvline(e0_mean, color="black", linestyle="--", linewidth=1, label="mean E0")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "qnc1.png")
    plt.close(fig)

    epochs, cv_mean, cv_std = _aggregate_field(runs_train, "equinorm_cv")
    _, dev_mean, dev_std = _aggregate_field(runs_train, "equiangle_dev")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, cv_mean, label="equinorm_cv (mean)")
    ax.fill_between(epochs, cv_mean - cv_std, cv_mean + cv_std, alpha=0.3)
    ax.plot(epochs, dev_mean, label="equiangle_dev (mean)")
    ax.fill_between(epochs, dev_mean - dev_std, dev_mean + dev_std, alpha=0.3)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.axhline(0.0, color="gray", linewidth=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("(symlog)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "qnc2.png")
    plt.close(fig)

    if runs_train[0] and runs_train[0][0].get("entropy_mean") is not None:
        epochs, ent_mean, ent_std = _aggregate_field(runs_train, "entropy_mean")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, ent_mean, color="black", label="entropy_mean (mean)")
        ax.fill_between(epochs, ent_mean - ent_std, ent_mean + ent_std, alpha=0.3, color="black")
        ax.set_xlabel("epoch")
        ax.set_ylabel("entanglement entropy S(rho_A) [bits]")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "entropy.png")
        plt.close(fig)


def make_sweep_comparison_figure(
    sweep_config: dict,
    manifest: dict[str, dict[str, str]],
    out_path: str | Path,
    results_dir: str | Path = "results",
) -> None:
    """Stage 4 (PLAYBOOK.md): one panel overlaying final-epoch equiangle_dev
    and qnc1_trace (mean +/- std across seeds) across the swept values.

    `sweep_config["variants"]` supplies the x-axis: each variant's optional
    `value` field (falling back to its `label`) in variant-list order.
    `manifest` is `{label: {seed: run_id}}` as written by sweep.py.
    """
    variants = sweep_config["variants"]
    labels = [v["label"] for v in variants]
    swept_values = [v.get("value", v["label"]) for v in variants]
    xlabel = sweep_config.get("xlabel", "swept value")

    dev_means, dev_stds, trace_means, trace_stds = [], [], [], []
    for label in labels:
        finals = []
        for run_id in manifest[label].values():
            records = _load_records(run_id, results_dir)
            train_r = [r for r in records if r["split"] == "train"]
            finals.append(train_r[-1])
        dev_vals = np.array([r["equiangle_dev"] for r in finals])
        trace_vals = np.array([r["qnc1_trace"] for r in finals])
        dev_means.append(dev_vals.mean())
        dev_stds.append(dev_vals.std())
        trace_means.append(trace_vals.mean())
        trace_stds.append(trace_vals.std())

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.errorbar(swept_values, dev_means, yerr=dev_stds, marker="o", color="C0", label="equiangle_dev (final)")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("equiangle_dev (final)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")

    ax2 = ax1.twinx()
    ax2.errorbar(swept_values, trace_means, yerr=trace_stds, marker="s", color="C1", label="qnc1_trace (final)")
    ax2.set_ylabel("qnc1_trace (final, log scale)", color="C1")
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor="C1")

    fig.suptitle(sweep_config.get("name", "sweep comparison"))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _aggregate_records_field(runs_records: list[list[dict]], field: str) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Mean +/- std of `field` across seeds' Stage 7 reanalysis records
    (shape: list of {"epoch": int, field: float, ...} per run), aligned on
    epochs common to every run. Mirrors `_aggregate_field` but for the
    reanalyze.py record shape (no "split" key)."""
    per_run_maps = [{r["epoch"]: r[field] for r in recs} for recs in runs_records]
    common_epochs = sorted(set.intersection(*(set(m) for m in per_run_maps)))
    values = np.array([[m[e] for m in per_run_maps] for e in common_epochs])
    return common_epochs, values.mean(axis=1), values.std(axis=1)


def make_normalized_witness_trajectory_figure(
    c2_records: dict[str, list[dict]],
    out_path: str | Path = "results/stage2_vqc_c2_diagnostic_aggregate/figures/normalized_witnesses.png",
) -> None:
    """PLAYBOOK2.md Stage 7: qnc1_fisher, qnc1_ratio, purity_mean vs epoch,
    mean +/- 1 std across the 5-seed C=2 anchor campaign, aligned on epochs
    common to every seed's Stage 7 reanalysis records."""
    runs_records = list(c2_records.values())
    n_seeds = len(runs_records)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    epochs, fisher_mean, fisher_std = _aggregate_records_field(runs_records, "qnc1_fisher")
    ax1.plot(epochs, fisher_mean, label="qnc1_fisher (mean)")
    ax1.fill_between(epochs, fisher_mean - fisher_std, fisher_mean + fisher_std, alpha=0.3)
    _, ratio_mean, ratio_std = _aggregate_records_field(runs_records, "qnc1_ratio")
    ax1.plot(epochs, ratio_mean, label="qnc1_ratio (mean)")
    ax1.fill_between(epochs, ratio_mean - ratio_std, ratio_mean + ratio_std, alpha=0.3)
    ax1.set_ylabel("S_W / S_B  or  D_within / D_between")
    ax1.legend()

    _, purity_mean_line, purity_std = _aggregate_records_field(runs_records, "purity_mean")
    ax2.plot(epochs, purity_mean_line, color="black", label="purity_mean (mean)")
    ax2.fill_between(epochs, purity_mean_line - purity_std, purity_mean_line + purity_std, alpha=0.3, color="black")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("mean class purity Tr(rho_bar_c^2)")
    ax2.legend()

    fig.suptitle(f"Stage 7: normalized collapse witnesses (mean +/- 1 std, n={n_seeds} seeds)")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def make_four_witness_comparison_figure(
    c2_records: dict[str, list[dict]],
    null_records: dict[str, list[dict]],
    out_path: str | Path = "results/stage7_reanalysis/figures/four_witness_comparison.png",
) -> None:
    """PLAYBOOK2.md Stage 7: bar comparison of qnc1_trace, qnc1_fisher,
    qnc1_ratio, purity_mean between the trained (final epoch, n=5) and
    random-init null (n=10) distributions."""
    fields = ("qnc1_trace", "qnc1_fisher", "qnc1_ratio", "purity_mean")
    trained_finals = [recs[-1] for recs in c2_records.values()]
    null_finals = [recs[-1] for recs in null_records.values()]

    trained_means = [np.mean([r[f] for r in trained_finals]) for f in fields]
    trained_stds = [np.std([r[f] for r in trained_finals]) for f in fields]
    null_means = [np.mean([r[f] for r in null_finals]) for f in fields]
    null_stds = [np.std([r[f] for r in null_finals]) for f in fields]

    x = np.arange(len(fields))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, trained_means, width, yerr=trained_stds, label="trained (n=5)", color="C0")
    ax.bar(x + width / 2, null_means, width, yerr=null_stds, label="null (n=10)", color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels(fields, rotation=15)
    ax.set_ylabel("value")
    ax.set_title("Stage 7: trained vs. random-init null, four normalized witnesses")
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def find_qfi_saturation_point(m_over_mc_values: list[float], qfi_rank_means: list[float]) -> float:
    """PLAYBOOK2.md Stage 9 / SPEC2_ADDENDUM.md section 12.2: the empirical M_c
    marker for the order-parameter figure.

    Saturation-detection rule (a modeling choice -- section 12.2 specifies how
    to compute qfi_rank but not how to locate its saturation point): the
    smallest m_over_mc at which qfi_rank_mean first reaches 95% of its
    maximum across the sweep.
    """
    threshold = 0.95 * max(qfi_rank_means)
    for m, q in zip(m_over_mc_values, qfi_rank_means):
        if q >= threshold:
            return m
    return m_over_mc_values[-1]


_ORDER_PARAMETER_FIELDS = (
    ("qnc1_fisher", "qnc1_fisher (final)"),
    ("qnc1_ratio", "qnc1_ratio (final)"),
    ("overlap_offdiag", "overlap_offdiag (final)"),
    ("purity_mean", "purity_mean (final)"),
    ("equiangle_dev", "equiangle_dev (final)"),
    ("entropy_mean", "entropy_mean (final)"),
    ("qfi_rank", "qfi_rank (final)"),
)


def make_order_parameter_figure(
    manifest: dict[str, dict[str, str]],
    results_dir: str | Path,
    out_path: str | Path,
) -> None:
    """PLAYBOOK2.md Stage 9 headline figure (SPEC2_ADDENDUM.md section 12.3):
    all outcome variables vs M/M_c_bound on a shared x-axis, with the
    qfi_rank saturation point (find_qfi_saturation_point) marked on every
    panel. One point per swept L (manifest label); mean +/- std across seeds
    except tpt_fraction and e0_mean (the latter averaged only over seeds that
    reached TPT).
    """
    labels = list(manifest.keys())
    finals_by_label = {}
    for label in labels:
        finals = []
        for run_id in manifest[label].values():
            records = _load_records(run_id, results_dir)
            train_r = [r for r in records if r["split"] == "train"]
            finals.append(train_r[-1])
        finals_by_label[label] = finals

    m_over_mc = [finals_by_label[label][0]["m_over_mc"] for label in labels]

    tpt_fraction = [
        sum(1 for r in finals_by_label[label] if r["tpt_reached"]) / len(finals_by_label[label])
        for label in labels
    ]

    e0_mean, e0_std = [], []
    for label in labels:
        reached = [r["e0_epoch"] for r in finals_by_label[label] if r["tpt_reached"]]
        if reached:
            arr = np.array(reached, dtype=float)
            e0_mean.append(arr.mean())
            e0_std.append(arr.std())
        else:
            e0_mean.append(np.nan)
            e0_std.append(0.0)

    qfi_rank_means = [
        np.array([r["qfi_rank"] for r in finals_by_label[label]], dtype=float).mean() for label in labels
    ]
    saturation_x = find_qfi_saturation_point(m_over_mc, qfi_rank_means)

    fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True)
    axes_flat = axes.flatten()

    ax = axes_flat[0]
    ax.plot(m_over_mc, tpt_fraction, marker="o")
    ax.set_ylabel("tpt_fraction")
    ax.set_ylim(-0.05, 1.05)

    ax = axes_flat[1]
    ax.errorbar(m_over_mc, e0_mean, yerr=e0_std, marker="o")
    ax.set_ylabel("E0 (mean+/-std)")

    for i, (field, ylabel) in enumerate(_ORDER_PARAMETER_FIELDS):
        ax = axes_flat[2 + i]
        means, stds = [], []
        for label in labels:
            vals = np.array([r[field] for r in finals_by_label[label]], dtype=float)
            means.append(vals.mean())
            stds.append(vals.std())
        ax.errorbar(m_over_mc, means, yerr=stds, marker="o")
        ax.set_ylabel(ylabel)

    for ax in axes_flat:
        ax.axvline(saturation_x, color="black", linestyle="--", linewidth=1)

    for ax in axes[-1]:
        ax.set_xlabel("M / M_c_bound")

    fig.suptitle("Stage 9: collapse as an order parameter of the overparameterization transition")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _final_train_records(manifest_entry: dict[str, str], results_dir: str | Path) -> list[dict]:
    """Final train-split record per seed for one manifest arm."""
    finals = []
    for run_id in manifest_entry.values():
        records = _load_records(run_id, results_dir)
        train_r = [r for r in records if r["split"] == "train"]
        finals.append(train_r[-1])
    return finals


_LOSS_FIGURE_FIELDS = (
    "qnc1_fisher",
    "qnc1_ratio",
    "overlap_offdiag",
    "purity_mean",
    "equiangle_dev",
    "entropy_mean",
)


def make_loss_comparison_figure(
    manifest: dict[str, dict[str, str]],
    results_dir: str | Path,
    out_path: str | Path,
    label_a: str = "ce",
    label_b: str = "mse",
) -> None:
    """PLAYBOOK2.md Stage 10 CE-vs-MSE figure (SPEC2_ADDENDUM.md §14): one
    panel per collapse witness, final-epoch mean +/- std bars for the matched
    CE and MSE arms. MSE is the loss covered by Du et al. (2023)'s theorem."""
    finals_a = _final_train_records(manifest[label_a], results_dir)
    finals_b = _final_train_records(manifest[label_b], results_dir)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, field in zip(axes.flatten(), _LOSS_FIGURE_FIELDS):
        vals_a = np.array([r[field] for r in finals_a], dtype=float)
        vals_b = np.array([r[field] for r in finals_b], dtype=float)
        ax.bar([0, 1], [vals_a.mean(), vals_b.mean()],
               yerr=[vals_a.std(), vals_b.std()], color=["C0", "C1"], width=0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"{label_a} (n={len(vals_a)})", f"{label_b} (n={len(vals_b)})"])
        ax.set_ylabel(field)

    fig.suptitle("Stage 10: matched CE vs MSE collapse witnesses (final epoch)")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


_ABLATION_FIGURE_FIELDS = (
    "qnc1_fisher",
    "qnc1_ratio",
    "purity_mean",
    "equiangle_dev",
    "entropy_mean",
)

_ABLATION_CAPTION = (
    "m_over_mc per arm uses each ansatz's own DLA capacity bound: 4^n-1=255 (entangling)\n"
    "vs 3n=12 (non-entangling: DLA is su(2)^{tensor n} regardless of depth, so those arms are\n"
    "always effectively saturated and can never exceed su(2)^{tensor n} capacity)."
)


def make_ablation_bound_figure(
    arms: dict[str, dict[str, str]],
    results_dir: str | Path,
    out_path: str | Path,
) -> None:
    """PLAYBOOK2.md Stage 10 entanglement-x-transition figure.

    One bar group per arm (e.g. L2/L8 x entangling true/false), per-witness
    panels plus a TPT-fraction panel. Each arm's x-tick is annotated with its
    logged m_over_mc, which for non-entangling arms is computed against the
    corrected DLA bound 3n=12 (see _ABLATION_CAPTION, rendered under the
    figure per the Stage 10 prompt)."""
    labels = list(arms.keys())
    finals_by_arm = {label: _final_train_records(arms[label], results_dir) for label in labels}
    x = np.arange(len(labels))
    tick_labels = [
        f"{label}\nM/M_c={finals_by_arm[label][0]['m_over_mc']:.3g}" for label in labels
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.flatten()

    ax = axes_flat[0]
    tpt_fracs = [
        sum(1 for r in finals_by_arm[label] if r["tpt_reached"]) / len(finals_by_arm[label])
        for label in labels
    ]
    ax.bar(x, tpt_fracs, color="C2", width=0.6)
    ax.set_ylabel("tpt_fraction")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=8)

    for ax, field in zip(axes_flat[1:], _ABLATION_FIGURE_FIELDS):
        means = [np.mean([r[field] for r in finals_by_arm[label]]) for label in labels]
        stds = [np.std([r[field] for r in finals_by_arm[label]]) for label in labels]
        ax.bar(x, means, yerr=stds, color="C0", width=0.6)
        ax.set_ylabel(field)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=8)

    fig.suptitle("Stage 10: entanglement x transition interaction (final epoch)")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.text(0.5, 0.01, _ABLATION_CAPTION, ha="center", fontsize=8)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def make_zspace_trajectory_figure(
    l8_records: dict[str, list[dict]],
    out_path: str | Path = "results/transition_L_L8_aggregate/figures/zspace_trajectory.png",
) -> None:
    """PLAYBOOK3.md Stage 12: NC1_z, r_z, beta*mean_margin vs epoch, mean +/-
    1 std across the 5-seed L=8 anchor (SPEC3_ADDENDUM.md sections 18.1-18.2)."""
    runs_records = list(l8_records.values())
    n_seeds = len(runs_records)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    epochs, nc1z_mean, nc1z_std = _aggregate_records_field(runs_records, "nc1_z")
    ax1.plot(epochs, nc1z_mean, label="NC1_z (mean)")
    ax1.fill_between(epochs, np.clip(nc1z_mean - nc1z_std, 1e-12, None), nc1z_mean + nc1z_std, alpha=0.3)
    _, rz_mean, rz_std = _aggregate_records_field(runs_records, "r_z")
    ax1.plot(epochs, rz_mean, label="r_z (mean)")
    ax1.fill_between(epochs, np.clip(rz_mean - rz_std, 1e-12, None), rz_mean + rz_std, alpha=0.3)
    ax1.set_yscale("log")
    ax1.set_ylabel("z-space NC1 (log scale)")
    ax1.legend()

    _, bm_mean, bm_std = _aggregate_records_field(runs_records, "beta_mean_margin")
    ax2.plot(epochs, bm_mean, color="black", label="beta * mean_margin (mean)")
    ax2.fill_between(epochs, bm_mean - bm_std, bm_mean + bm_std, alpha=0.3, color="black")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("effective CE margin")
    ax2.legend()

    fig.suptitle(f"Stage 12: z-space collapse trajectory (mean +/- 1 std, n={n_seeds} seeds)")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def make_box_vs_etf_figure(
    l8_records: dict[str, list[dict]],
    out_path: str | Path = "results/transition_L_L8_aggregate/figures/box_vs_etf.png",
) -> None:
    """PLAYBOOK3.md Stage 12: normalized d_boxn = d_box/2 vs d_etf =
    equiangle_dev_z trajectories, mean +/- 1 std across the 5-seed L=8 anchor
    (SPEC3_ADDENDUM.md section 18.3 box-vs-ETF model comparison)."""
    runs_records = list(l8_records.values())
    n_seeds = len(runs_records)

    epochs, dbox_mean, dbox_std = _aggregate_records_field(runs_records, "d_box")
    dboxn_mean, dboxn_std = dbox_mean / 2.0, dbox_std / 2.0
    _, detf_mean, detf_std = _aggregate_records_field(runs_records, "equiangle_dev_z")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, dboxn_mean, label="d_boxn = d_box/2 (mean)")
    ax.fill_between(epochs, dboxn_mean - dboxn_std, dboxn_mean + dboxn_std, alpha=0.3)
    ax.plot(epochs, detf_mean, label="d_etf = equiangle_dev_z (mean)")
    ax.fill_between(epochs, detf_mean - detf_std, detf_mean + detf_std, alpha=0.3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("distance to geometry target")
    ax.legend()
    fig.suptitle(f"Stage 12: box vertices vs simplex ETF (mean +/- 1 std, n={n_seeds} seeds)")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def make_fiber_headline_figure(
    l8_records: dict[str, list[dict]],
    out_path: str | Path = "results/transition_L_L8_aggregate/figures/fiber_headline.png",
) -> None:
    """PLAYBOOK3.md Stage 12 headline figure: V_meas vs V_fiber over training
    (SPEC3_ADDENDUM.md section 18.4), mean +/- 1 std across the 5-seed L=8
    anchor, plus fiber_fraction in the second panel."""
    runs_records = list(l8_records.values())
    n_seeds = len(runs_records)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    epochs, vmeas_mean, vmeas_std = _aggregate_records_field(runs_records, "v_meas")
    ax1.plot(epochs, vmeas_mean, label="V_meas (mean)")
    ax1.fill_between(epochs, np.clip(vmeas_mean - vmeas_std, 0, None), vmeas_mean + vmeas_std, alpha=0.3)
    _, vfiber_mean, vfiber_std = _aggregate_records_field(runs_records, "v_fiber")
    ax1.plot(epochs, vfiber_mean, label="V_fiber (mean)")
    ax1.fill_between(epochs, np.clip(vfiber_mean - vfiber_std, 0, None), vfiber_mean + vfiber_std, alpha=0.3)
    ax1.set_ylabel("within-class HS variance")
    ax1.legend()

    _, ff_mean, ff_std = _aggregate_records_field(runs_records, "fiber_fraction")
    ax2.plot(epochs, ff_mean, color="black", label="fiber_fraction (mean)")
    ax2.fill_between(epochs, ff_mean - ff_std, ff_mean + ff_std, alpha=0.3, color="black")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("V_fiber / (V_meas + V_fiber)")
    ax2.legend()

    fig.suptitle(f"Stage 12 headline: measurement-subspace vs fiber decomposition (mean +/- 1 std, n={n_seeds} seeds)")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def make_etf_formation_figure(
    supplement_records: dict[str, list[dict]],
    out_path: str | Path = "results/transition_L_L8_aggregate/figures/etf_formation.png",
) -> None:
    """PLAYBOOK3.md Stage 13 Task 1: d_etf (equiangle_dev_z) and mean pairwise
    z-space cosine vs epoch, E0 marked per seed, L=8 anchor (5 seeds) --
    shows when ETF geometry forms relative to TPT onset."""
    runs_records = []
    e0_epochs = []
    for recs in supplement_records.values():
        annotated = [{**r, "mean_pairwise_cos": float(np.mean(r["pairwise_cos"]))} for r in recs]
        runs_records.append(annotated)
        e0_epochs.append(recs[0]["e0_epoch"])
    n_seeds = len(runs_records)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    epochs, detf_mean, detf_std = _aggregate_records_field(runs_records, "equiangle_dev_z")
    ax1.plot(epochs, detf_mean, label="d_etf = equiangle_dev_z (mean)")
    ax1.fill_between(epochs, detf_mean - detf_std, detf_mean + detf_std, alpha=0.3)
    ax1.set_ylabel("ETF deviation")
    ax1.legend()

    _, cos_mean, cos_std = _aggregate_records_field(runs_records, "mean_pairwise_cos")
    ax2.plot(epochs, cos_mean, color="black", label="mean pairwise z-space cosine")
    ax2.fill_between(epochs, cos_mean - cos_std, cos_mean + cos_std, alpha=0.3, color="black")
    ax2.axhline(-0.5, color="red", linestyle=":", label="ETF target (-1/(C-1))")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("cosine")
    ax2.legend()

    for e0 in e0_epochs:
        if e0 is not None:
            ax1.axvline(e0, color="gray", linestyle="--", alpha=0.4)
            ax2.axvline(e0, color="gray", linestyle="--", alpha=0.4)

    fig.suptitle(f"Stage 13 supplement: ETF-formation dynamics (mean +/- 1 std, n={n_seeds} seeds, E0 per seed marked)")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def make_nc1z_decomposition_figure(
    supplement_records: dict[str, list[dict]],
    out_path: str | Path = "results/transition_L_L8_aggregate/figures/nc1z_decomposition.png",
) -> None:
    """PLAYBOOK3.md Stage 13 Task 1: tr(Sigma_W^z) and tr(Sigma_B^z) vs epoch
    separately (within-class shrink vs between-class growth), L=8 anchor
    (5 seeds)."""
    runs_records = list(supplement_records.values())
    n_seeds = len(runs_records)

    fig, ax = plt.subplots(figsize=(7, 5))
    epochs, tw_mean, tw_std = _aggregate_records_field(runs_records, "trace_w")
    ax.plot(epochs, tw_mean, label="tr(Sigma_W^z) (mean)")
    ax.fill_between(epochs, np.clip(tw_mean - tw_std, 1e-12, None), tw_mean + tw_std, alpha=0.3)
    _, tb_mean, tb_std = _aggregate_records_field(runs_records, "trace_b")
    ax.plot(epochs, tb_mean, label="tr(Sigma_B^z) (mean)")
    ax.fill_between(epochs, np.clip(tb_mean - tb_std, 1e-12, None), tb_mean + tb_std, alpha=0.3)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("trace (log scale)")
    ax.legend()
    fig.suptitle(f"Stage 13 supplement: NC1_z decomposition -- within-shrink vs between-growth (mean +/- 1 std, n={n_seeds} seeds)")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def make_fiber_dip_diagnostic_figure(
    supplement_records: dict[str, list[dict]],
    null_records: dict[str, list[dict]],
    out_path: str | Path = "results/transition_L_L8_aggregate/figures/fiber_dip_diagnostic.png",
) -> None:
    """PLAYBOOK3.md Stage 13 Task 1: V_fiber vs epoch against the untrained
    null band, L=8 anchor (5 seeds), showing when the ~6% dip (STAGE12_FINDINGS
    P4) develops."""
    runs_records = list(supplement_records.values())
    n_seeds = len(runs_records)
    null_finals = [recs[-1]["v_fiber"] for recs in null_records.values()]
    null_mean = float(np.mean(null_finals))
    null_std = float(np.std(null_finals))

    fig, ax = plt.subplots(figsize=(7, 5))
    epochs, vf_mean, vf_std = _aggregate_records_field(runs_records, "v_fiber")
    ax.plot(epochs, vf_mean, color="black", label="V_fiber (mean)")
    ax.fill_between(epochs, vf_mean - vf_std, vf_mean + vf_std, alpha=0.3, color="black")
    ax.axhline(null_mean, color="red", linestyle="--", label="untrained-null V_fiber (mean)")
    ax.fill_between(epochs, null_mean - null_std, null_mean + null_std, color="red", alpha=0.15)
    ax.set_xlabel("epoch")
    ax.set_ylabel("V_fiber")
    ax.legend()
    fig.suptitle(f"Stage 13 supplement: V_fiber dip vs untrained-null band (mean +/- 1 std, n={n_seeds} seeds)")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


_EXPRESSIVITY_FIELDS = (
    ("qnc1_fisher", "qnc1_fisher (full-space)"),
    ("purity_mean", "purity_mean (full-space)"),
    ("overlap_offdiag", "overlap_offdiag (full-space)"),
    ("fiber_fraction", "fiber_fraction"),
    ("feature_nc1_z", "feature-space NC1_z"),
)


def make_expressivity_figure(
    stage13_records: dict[str, dict[str, dict]],
    out_path: str | Path = "results/sweeps/stage13_expressivity_aggregate/expressivity.png",
    order: tuple[str, ...] = ("R0", "R1", "R2", "R3"),
) -> None:
    """PLAYBOOK3.md Stage 13 Task 2 headline figure: full-space witnesses
    (qnc1_fisher, purity_mean, overlap_offdiag) AND fiber_fraction, plus
    feature-space NC1_z, vs readout family (mean +/- 1 std across the 5
    seeds per arm). PLAYBOOK3.md Stage 13B Task A adds the isotropic
    dimension-counting null (1 - k/(4^n-1), per-arm `isotropic_fiber_null`
    field) to the fiber_fraction panel, when present in `stage13_records`."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    x = np.arange(len(order))
    for ax, (field, ylabel) in zip(axes.flatten(), _EXPRESSIVITY_FIELDS):
        means = []
        stds = []
        for label in order:
            recs = list(stage13_records[label].values())
            vals = np.array([r[field] for r in recs], dtype=float)
            means.append(vals.mean())
            stds.append(vals.std())
        ax.errorbar(x, means, yerr=stds, marker="o", capsize=4, label="fiber_fraction" if field == "fiber_fraction" else None)
        if field == "fiber_fraction" and all("isotropic_fiber_null" in r for recs in stage13_records.values() for r in recs.values()):
            null_vals = [
                np.mean([r["isotropic_fiber_null"] for r in stage13_records[label].values()]) for label in order
            ]
            ax.plot(x, null_vals, marker="x", linestyle="--", color="grey", label="isotropic null (1-k/(4^n-1))")
            ax.legend(fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(order)
        ax.set_ylabel(ylabel)
    axes.flatten()[-1].axis("off")
    fig.suptitle("Stage 13: expressivity sweep -- full-space + feature-space witnesses vs readout family")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def make_stage14_long_horizon_drift_figure(
    records: list[dict],
    e0_epoch: int | None,
    out_path: str | Path = "results/stage14_long_horizon_aggregate/figures/trace_w_drift.png",
) -> None:
    """PREDICTIONS.md Stage 14 arm 2 (long horizon, 5000 epochs, 1 seed):
    tr(Sigma_W^z) vs log(epoch), E0 marked, for the post-TPT Spearman drift
    test's underlying trajectory."""
    epochs = np.array([r["epoch"] for r in records if r["epoch"] > 0])
    trace_w = np.array([r["trace_w"] for r in records if r["epoch"] > 0])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, trace_w, marker=".", color="C0")
    ax.set_xscale("log")
    ax.set_yscale("log")
    if e0_epoch is not None:
        ax.axvline(e0_epoch, color="red", linestyle="--", label=f"E0 (epoch {e0_epoch})")
        ax.legend()
    ax.set_xlabel("epoch (log scale)")
    ax.set_ylabel("tr(Sigma_W^z) (log scale)")
    fig.suptitle("Stage 14 arm 2: within-class trace drift over 5000 epochs (1 seed)")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def make_stage14_tpt_fraction_figure(
    tpt_fractions: dict[str, float],
    variant_values: dict[str, float],
    out_path: str | Path = "results/sweeps/stage14_transition_localization_aggregate/figures/tpt_fraction.png",
) -> None:
    """PREDICTIONS.md Stage 14 arm 4 (transition localization): TPT-reached
    fraction (10 seeds/arm) vs L, in `tpt_fractions` insertion order.
    `variant_values` maps each label to its numeric L for the x-axis
    (configs/sweeps/stage14_transition_localization.yaml's `value` field)."""
    labels = list(tpt_fractions.keys())
    x = [variant_values[label] for label in labels]
    y = [tpt_fractions[label] for label in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, marker="o", color="C0")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("layers (L)")
    ax.set_ylabel("TPT-reached fraction (n=10 seeds)")
    fig.suptitle("Stage 14 arm 4: TPT-fraction vs transition localization")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def make_stage14b_rotation_variance_figure(
    reupload_true_records: dict[str, list[dict]],
    reupload_false_records: dict[str, list[dict]],
    out_path: str | Path = "results/stage14b_reanalysis/figures/total_variance_rotation.png",
) -> None:
    """PREDICTIONS.md Stage 14B (HR1/HR2, "collapse by rotation"): total
    within-class Hilbert-Schmidt variance (v_meas + v_fiber, `total_var`)
    vs epoch, mean +/- 1 std across seeds, reupload=true vs reupload=false
    overlaid on the same axes. reupload=false is predicted flat (rotation
    only, HR2); reupload=true is predicted to decline only slightly (HR1)."""
    true_runs = list(reupload_true_records.values())
    false_runs = list(reupload_false_records.values())

    fig, ax = plt.subplots(figsize=(7, 5))

    epochs_t, mean_t, std_t = _aggregate_records_field(true_runs, "total_var")
    ax.plot(epochs_t, mean_t, color="C0", label=f"reupload=true (n={len(true_runs)})")
    ax.fill_between(epochs_t, mean_t - std_t, mean_t + std_t, alpha=0.3, color="C0")

    epochs_f, mean_f, std_f = _aggregate_records_field(false_runs, "total_var")
    ax.plot(epochs_f, mean_f, color="C1", label=f"reupload=false (n={len(false_runs)})")
    ax.fill_between(epochs_f, mean_f - std_f, mean_f + std_f, alpha=0.3, color="C1")

    ax.set_xlabel("epoch")
    ax.set_ylabel("total within-class HS variance (v_meas + v_fiber)")
    ax.legend()
    fig.suptitle("Stage 14B: collapse-by-rotation -- total variance, reupload true vs false")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def make_stage15_n4_vs_n6_dichotomy_figure(
    n4_records: dict[str, list[dict]],
    n6_records: dict[str, list[dict]],
    out_path: str | Path = "results/stage15_n6_reanalysis/figures/n4_vs_n6_dichotomy.png",
) -> None:
    """PLAYBOOK3.md Stage 15 (SPEC3_ADDENDUM.md section 21): fiber_fraction
    vs epoch, mean +/- 1 std, n=4 L=8 anchor (Stage 12) side by side with
    n=6 L=8 campaign (Stage 15), same y-axis -- tests whether the
    measurement-subspace-collapse / fiber-no-collapse dichotomy strengthens
    (fiber_fraction closer to 1) as n grows (HN2)."""
    n4_runs = list(n4_records.values())
    n6_runs = list(n6_records.values())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    epochs4, ff4_mean, ff4_std = _aggregate_records_field(n4_runs, "fiber_fraction")
    ax1.plot(epochs4, ff4_mean, color="C0")
    ax1.fill_between(epochs4, ff4_mean - ff4_std, ff4_mean + ff4_std, alpha=0.3, color="C0")
    ax1.set_title(f"n=4 (L=8 anchor, n={len(n4_runs)} seeds)")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("fiber_fraction = V_fiber / (V_meas + V_fiber)")
    ax1.set_ylim(-0.05, 1.05)

    epochs6, ff6_mean, ff6_std = _aggregate_records_field(n6_runs, "fiber_fraction")
    ax2.plot(epochs6, ff6_mean, color="C1")
    ax2.fill_between(epochs6, ff6_mean - ff6_std, ff6_mean + ff6_std, alpha=0.3, color="C1")
    ax2.set_title(f"n=6 (L=8 campaign, n={len(n6_runs)} seeds)")
    ax2.set_xlabel("epoch")

    fig.suptitle("Stage 15: measurement-subspace-collapse / fiber-no-collapse dichotomy, n=4 vs n=6")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def make_stage15_n6_total_var_overlay_figure(
    reupload_true_records: dict[str, list[dict]],
    reupload_false_records: dict[str, list[dict]],
    out_path: str | Path = "results/stage15_n6_reanalysis/figures/n6_total_variance_rotation.png",
) -> None:
    """PLAYBOOK3.md Stage 15 (HN1): total within-class Hilbert-Schmidt
    variance (v_meas + v_fiber, `total_var`) vs epoch, mean +/- 1 std, n=6
    L=8 campaign, reupload=true vs reupload=false overlaid -- same rotation-
    mechanism check as make_stage14b_rotation_variance_figure (HR1/HR2),
    replicated at n=6."""
    true_runs = list(reupload_true_records.values())
    false_runs = list(reupload_false_records.values())

    fig, ax = plt.subplots(figsize=(7, 5))

    epochs_t, mean_t, std_t = _aggregate_records_field(true_runs, "total_var")
    ax.plot(epochs_t, mean_t, color="C0", label=f"reupload=true (n={len(true_runs)})")
    ax.fill_between(epochs_t, mean_t - std_t, mean_t + std_t, alpha=0.3, color="C0")

    epochs_f, mean_f, std_f = _aggregate_records_field(false_runs, "total_var")
    ax.plot(epochs_f, mean_f, color="C1", label=f"reupload=false (n={len(false_runs)})")
    ax.fill_between(epochs_f, mean_f - std_f, mean_f + std_f, alpha=0.3, color="C1")

    ax.set_xlabel("epoch")
    ax.set_ylabel("total within-class HS variance (v_meas + v_fiber)")
    ax.legend()
    fig.suptitle("Stage 15: n=6 collapse-by-rotation replication -- total variance, reupload true vs false")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def make_stage20_w2_contraction_vs_rotation_figure(
    classical_records: dict[str, list[list[dict]]],
    vqc_records: dict[str, list[list[dict]]],
    out_path: str | Path = "figs/fig19_contraction_vs_rotation.png",
) -> None:
    """PREDICTIONS.md Stage 20 Arm W2: classical MLP's absolute within-class
    scatter trace (tr(Sigma_W), unbounded -- ReLU activations have no norm
    constraint) beside the matched VQC's total within-class HS variance
    (total_var, pinned to ~1e-15 by unitarity, T5). Both panels share a
    log-y axis so the classical trajectory's multi-order-of-magnitude change
    and the VQC's dead-flat machine-precision line read on the same visual
    scale. classical_records/vqc_records: {dataset_label: [seed run records]}.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    for i, (label, runs) in enumerate(classical_records.items()):
        color = f"C{i}"
        epochs, mean, std = _aggregate_records_field(runs, "within_class_scatter_trace")
        ax1.plot(epochs, mean, color=color, label=f"{label} (n={len(runs)})")
        ax1.fill_between(epochs, np.clip(mean - std, 1e-12, None), mean + std, alpha=0.3, color=color)
    ax1.set_yscale("log")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("tr(Sigma_W), absolute (classical MLP)")
    ax1.set_title("Classical MLP: absolute within-class scatter")
    ax1.legend()

    for i, (label, runs) in enumerate(vqc_records.items()):
        color = f"C{i}"
        epochs, mean, std = _aggregate_records_field(runs, "total_var")
        ax2.plot(epochs, mean, color=color, label=f"{label} (n={len(runs)})")
        ax2.fill_between(epochs, np.clip(mean - std, 1e-18, None), mean + std, alpha=0.3, color=color)
    ax2.set_yscale("log")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("total_var = v_meas + v_fiber (VQC, no-reupload)")
    ax2.set_title("VQC (no-reupload): total within-class HS variance")
    ax2.legend()

    fig.suptitle("Stage 20 Arm W2: contraction (classical) vs rotation (VQC), matched PCA-4 inputs")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def make_hardware_zspace_comparison_figure(
    zbar_c_hw: np.ndarray,
    zbar_c_hw_err: np.ndarray,
    zbar_c_sim: np.ndarray,
    out_path: str | Path,
) -> None:
    """Stage 16 Task C (SPEC3_ADDENDUM.md section 22): z-space class means,
    hardware (with binomial shot-noise error bars) vs exact simulator, one
    grouped point per (class, coordinate)."""
    num_classes = zbar_c_sim.shape[0]
    labels = [f"c={c},z[{k}]" for c in range(num_classes) for k in range(num_classes)]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.errorbar(
        x, zbar_c_hw.flatten(), yerr=zbar_c_hw_err.flatten(),
        fmt="o", color="C0", label="hardware", capsize=3,
    )
    ax.scatter(x, zbar_c_sim.flatten(), marker="x", color="C1", label="simulator (exact)", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("zbar_c value")
    ax.legend()
    fig.suptitle("Stage 16: z-space class means, hardware vs simulator")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def make_hardware_nc1z_bar_figure(
    nc1z_hw: float,
    nc1z_sim: float,
    out_path: str | Path,
) -> None:
    """Stage 16 Task C (SPEC3_ADDENDUM.md section 22): NC1_z, hardware vs
    exact simulator, single comparison bar figure."""
    fig, ax = plt.subplots(figsize=(4, 4.5))
    ax.bar(["hardware", "simulator\n(exact)"], [nc1z_hw, nc1z_sim], color=["C0", "C1"])
    ax.set_ylabel("NC1_z")
    fig.suptitle("Stage 16: NC1_z, hardware vs simulator")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def make_hardware_zspace_rescaled_figure(
    zbar_c_hw_rescaled: np.ndarray,
    zbar_c_hw_rescaled_err: np.ndarray,
    zbar_c_sim: np.ndarray,
    out_path: str | Path,
) -> None:
    """Stage 16b HW1 (PREDICTIONS.md): z-space class means, hardware
    RESCALED by the fitted contraction factor 1/lambda, overlaid on the
    exact simulator -- visualizes whether a pure scaling accounts for the
    hardware-vs-simulator gap."""
    num_classes = zbar_c_sim.shape[0]
    labels = [f"c={c},z[{k}]" for c in range(num_classes) for k in range(num_classes)]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.errorbar(
        x, zbar_c_hw_rescaled.flatten(), yerr=zbar_c_hw_rescaled_err.flatten(),
        fmt="o", color="C0", label="hardware / lambda (rescaled)", capsize=3,
    )
    ax.scatter(x, zbar_c_sim.flatten(), marker="x", color="C1", label="simulator (exact)", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("zbar_c value")
    ax.legend()
    fig.suptitle("Stage 16b HW1: z-space class means, hardware/lambda vs simulator")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    """CLI entry: ``python -m qnc.figures <run_id>`` (CLAUDE.md commands)."""
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate SPEC §8 figures from metrics.jsonl.")
    parser.add_argument("run_id")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    make_figures(args.run_id, args.results_dir)
    print(f"figures written to {Path(args.results_dir) / args.run_id / 'figures'}")


if __name__ == "__main__":
    main()
