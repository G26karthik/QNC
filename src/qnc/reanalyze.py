"""Stage 7 reanalysis of Phase 1 data with SPEC2_ADDENDUM.md §11 normalized
collapse witnesses.

Runs NO new training (PLAYBOOK2.md Stage 7): every state below comes either
from re-reading logged JSONL fields, from reloading a saved checkpoint's
state_dict, or from a seeded re-construction of the pre-training (epoch 0)
model -- never from a gradient step. Never rewrites existing metrics.jsonl
files (CLAUDE.md invariant 3, append-only logging).
"""

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import mannwhitneyu, spearmanr

from qnc.metrics import (
    centered_means,
    class_means,
    d_box,
    equiangle_dev_z,
    fiber_decomposition,
    hs_norm,
    isotropic_fiber_null,
    loss_visible_subspace_decomposition,
    margins,
    measured_subspace_basis,
    measured_subspace_basis_r1,
    measured_subspace_basis_r2,
    measured_subspace_basis_r3,
    nc1_z_metrics,
    overlap_offdiag,
    purity,
    qfi_matrix,
    qfi_rank,
    qnc1_fisher,
    qnc1_metrics,
    qnc1_ratio,
    z_features,
)
from qnc.data import load_blobs3, load_fashion_mnist_subset, load_mnist_subset, pca_encode, pca_encode_amplitude
from qnc.models import VQC
from qnc.sweep import merge_overrides
from qnc.train import assert_physical_rhos, statevectors_to_rhos

RESULTS_DIR = Path("results")
CONFIGS_DIR = Path("configs")

C2_ANCHOR_CONFIG = "stage2_vqc_c2_diagnostic.yaml"
C2_ANCHOR_PREFIX = "stage2_vqc_c2_diagnostic_s"
RANDOM_INIT_CONFIG = "stage5_random_init.yaml"
RANDOM_INIT_PREFIX = "stage5_random_init_s"

STAGE9_BASE_CONFIG = "stage9_transition.yaml"
STAGE9_SWEEP_CONFIG = Path("configs/sweeps/transition_L.yaml")
STAGE9_MANIFEST_PATH = Path("results/sweeps/transition_L_manifest.json")

STAGE13_BASE_CONFIG = "stage13_expressivity.yaml"
STAGE13_SWEEP_CONFIG = Path("configs/sweeps/stage13_expressivity.yaml")

STAGE14_MECHANISM_BASE_CONFIG = "stage14_mechanism_base.yaml"
STAGE14_MECHANISM_SWEEP_CONFIG = Path("configs/sweeps/stage14_mechanism_arms.yaml")
STAGE14_LONG_HORIZON_CONFIG = "stage14_long_horizon.yaml"
STAGE14_TRANSITION_SWEEP_CONFIG = Path("configs/sweeps/stage14_transition_localization.yaml")
STAGE14_QFI_ARM_LABELS = ("L8", "L16", "L21", "L26", "L32")

STAGE14B_BLOB_STD_SWEEP_CONFIG = Path("configs/sweeps/stage14b_blob_std.yaml")
STAGE14B_BLOB_STD_MANIFEST_PATH = Path("results/sweeps/stage14b_blob_std_assembled/manifest.json")
STAGE14B_REUPLOAD_FALSE_SWEEP_CONFIG = Path("configs/sweeps/stage14b_reupload_false.yaml")
STAGE14B_REUPLOAD_FALSE_RUN_IDS = [
    "stage14b_reupload_false_reupload_false_s0_20260715-213042",
    "stage14b_reupload_false_reupload_false_s1_20260715-213528",
    "stage14b_reupload_false_reupload_false_s2_20260715-213936",
    "stage14b_reupload_false_reupload_false_s3_20260715-214338",
    "stage14b_reupload_false_reupload_false_s4_20260715-214807",
]

FOUR_WITNESS_FIELDS = ("qnc1_trace", "qnc1_fisher", "qnc1_ratio", "purity_mean")


def discover_run_ids(results_dir: Path = RESULTS_DIR) -> list[str]:
    """Every Phase 1 run directory that has its own metrics.jsonl (excludes
    *_aggregate dirs, sweeps/, and top-level .md files)."""
    return sorted(
        p.name
        for p in results_dir.iterdir()
        if p.is_dir() and (p / "metrics.jsonl").exists()
    )


def _discover_seeded_run_ids(prefix: str, expected_count: int, results_dir: Path) -> list[str]:
    run_ids = sorted(p.name for p in results_dir.iterdir() if p.is_dir() and p.name.startswith(prefix))
    assert len(run_ids) == expected_count, (
        f"expected {expected_count} run dirs with prefix {prefix!r}, found {len(run_ids)}: {run_ids}"
    )
    return run_ids


def discover_c2_anchor_run_ids(results_dir: Path = RESULTS_DIR) -> list[str]:
    return _discover_seeded_run_ids(C2_ANCHOR_PREFIX, 5, results_dir)


def discover_random_init_run_ids(results_dir: Path = RESULTS_DIR) -> list[str]:
    return _discover_seeded_run_ids(RANDOM_INIT_PREFIX, 10, results_dir)


def _parse_seed(run_id: str) -> int:
    # run_id = <experiment>_s<seed>_<timestamp>
    marker = "_s"
    idx = run_id.rindex(marker, 0, run_id.rindex("_"))
    tail = run_id[idx + len(marker) :]
    return int(tail.split("_")[0])


def recompute_fisher_from_jsonl(results_dir: Path = RESULTS_DIR) -> dict[str, list[dict]]:
    """SPEC2 §11.1 task (a): qnc1_fisher per logged epoch for every Phase 1
    run, using only already-logged qnc1_hs / class_mean_norms fields.
    Classical (Stage 1) runs, where qnc1_hs is null, are skipped."""
    out: dict[str, list[dict]] = {}
    for run_id in discover_run_ids(results_dir):
        path = results_dir / run_id / "metrics.jsonl"
        recs = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("qnc1_hs") is None or r.get("class_mean_norms") is None:
                    continue
                fisher = qnc1_fisher(r["qnc1_hs"], np.array(r["class_mean_norms"]))
                recs.append({"epoch": r["epoch"], "split": r["split"], "qnc1_fisher": fisher})
        if recs:
            out[run_id] = recs
    return out


def _build_model_and_data(config: dict, seed: int) -> tuple[VQC, torch.Tensor, torch.Tensor]:
    """Seeded reconstruction of train_vqc's pre-training state: same seed ->
    identical data split, PCA encoding, and initial VQC weights (SPEC.md §2).
    No optimizer, no gradient step."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_cfg = config["data"]
    dataset_name = data_cfg["dataset"]
    if dataset_name == "mnist":
        x_train_raw, y_train, x_test_raw, _y_test = load_mnist_subset(
            classes=data_cfg["classes"],
            samples_per_class=data_cfg["samples_per_class"],
            test_fraction=data_cfg["test_fraction"],
            seed=seed,
        )
        num_classes = len(data_cfg["classes"])
    elif dataset_name == "fashion_mnist":
        x_train_raw, y_train, x_test_raw, _y_test = load_fashion_mnist_subset(
            classes=data_cfg["classes"],
            samples_per_class=data_cfg["samples_per_class"],
            test_fraction=data_cfg["test_fraction"],
            seed=seed,
        )
        num_classes = len(data_cfg["classes"])
    elif dataset_name == "blobs3":
        x_train_raw, y_train, x_test_raw, _y_test = load_blobs3(
            samples_per_class=data_cfg["samples_per_class"],
            blob_dist=data_cfg["blob_dist"],
            blob_std=data_cfg["blob_std"],
            test_fraction=data_cfg["test_fraction"],
            seed=seed,
            ambient_dim=data_cfg.get("ambient_dim", 4),
        )
        num_classes = 3
    else:
        raise ValueError(f"_build_model_and_data: unsupported dataset {dataset_name!r}")

    model_cfg = config["model"]
    n_qubits = model_cfg["n_qubits"]
    pca_components = data_cfg.get("pca_components", n_qubits)
    encoding = model_cfg.get("encoding", "angle")
    if encoding == "amplitude":
        if dataset_name in ("mnist", "fashion_mnist"):
            # Mirrors train.py's amplitude branch (Stage 20 Arm W4): PCA down
            # to pca_components WITHOUT angle-rescaling, then let
            # AmplitudeEmbedding normalize internally.
            x_train, _x_test = pca_encode_amplitude(x_train_raw, x_test_raw, n_components=pca_components, seed=seed)
        else:
            x_train = x_train_raw
    else:
        x_train, _x_test = pca_encode(x_train_raw, x_test_raw, n_qubits=pca_components, seed=seed)
    x_train = x_train.to(torch.float64)
    model = VQC(
        n_qubits=n_qubits,
        num_classes=num_classes,
        n_layers=model_cfg["layers"],
        reupload=model_cfg.get("reupload", False),
        entangling=model_cfg.get("entangling", True),
        beta_0=model_cfg.get("beta0", 5.0),
        beta_learnable=model_cfg.get("beta_learnable", True),
        ansatz=model_cfg.get("ansatz", "strongly_entangling"),
        encoding=encoding,
        readout_family=model_cfg.get("readout_family", "R0"),
        measurement_layers=model_cfg.get("measurement_layers", 2),
        shots=config["train"].get("shots"),
    )
    return model, x_train, y_train


def _recompute_epoch_witnesses(model: VQC, x_train: torch.Tensor, y_train: torch.Tensor) -> dict:
    """Regenerates states from the model's CURRENT weights (caller has either
    just constructed it, at epoch 0, or loaded a checkpoint state_dict) and
    computes all SPEC2 §11 witnesses plus the recomputed qnc1_trace/qnc1_hs
    for a consistency check against the original JSONL log."""
    model.eval()
    with torch.no_grad():
        psi = model.state(x_train).detach().cpu().numpy()
    rhos = statevectors_to_rhos(psi)
    assert_physical_rhos(rhos)

    labels = y_train.detach().cpu().numpy()
    means, global_mean = class_means(rhos, labels)
    m = centered_means(means, global_mean)
    class_mean_norms = np.array([hs_norm(m[c]) for c in range(means.shape[0])])

    qnc1 = qnc1_metrics(rhos, labels, means, global_mean)
    fisher = qnc1_fisher(qnc1["qnc1_hs"], class_mean_norms)
    ratio = qnc1_ratio(qnc1["qnc1_trace"], means)
    overlap = overlap_offdiag(means)
    purity_mean = float(np.mean([purity(means[c]) for c in range(means.shape[0])]))

    return {
        "qnc1_trace": qnc1["qnc1_trace"],
        "qnc1_hs": qnc1["qnc1_hs"],
        "qnc1_fisher": fisher,
        "qnc1_ratio": ratio,
        "overlap_offdiag": overlap,
        "purity_mean": purity_mean,
    }


def reanalyze_c2_anchor(
    results_dir: Path = RESULTS_DIR, configs_dir: Path = CONFIGS_DIR
) -> dict[str, list[dict]]:
    """SPEC2 §11.2/11.3 task (b), 5-seed C=2 anchor campaign: recompute at
    epoch 0 (seeded reconstruction) and at every saved checkpoint."""
    config = yaml.safe_load(open(configs_dir / C2_ANCHOR_CONFIG))
    out: dict[str, list[dict]] = {}
    for run_id in discover_c2_anchor_run_ids(results_dir):
        seed = _parse_seed(run_id)
        model, x_train, y_train = _build_model_and_data(config, seed)

        records = [{"epoch": 0, **_recompute_epoch_witnesses(model, x_train, y_train)}]

        ckpt_dir = results_dir / run_id / "checkpoints"
        for ckpt_path in sorted(ckpt_dir.glob("epoch_*.pt")):
            epoch = int(ckpt_path.stem.split("_")[1])
            model.load_state_dict(torch.load(ckpt_path, weights_only=True))
            records.append({"epoch": epoch, **_recompute_epoch_witnesses(model, x_train, y_train)})

        out[run_id] = records
    return out


def reanalyze_random_init_null(
    results_dir: Path = RESULTS_DIR, configs_dir: Path = CONFIGS_DIR
) -> dict[str, list[dict]]:
    """SPEC2 §11.2/11.3 task (b), 10-seed random-init null: epochs=0 and
    checkpoint=false in this config, so the only state is the seeded
    pre-training reconstruction (identical procedure to the c2 anchor's
    epoch 0, no checkpoint file needed)."""
    config = yaml.safe_load(open(configs_dir / RANDOM_INIT_CONFIG))
    out: dict[str, list[dict]] = {}
    for run_id in discover_random_init_run_ids(results_dir):
        seed = _parse_seed(run_id)
        model, x_train, y_train = _build_model_and_data(config, seed)
        out[run_id] = [{"epoch": 0, **_recompute_epoch_witnesses(model, x_train, y_train)}]
    return out


def _recompute_zspace_witnesses(model: VQC, x_train: torch.Tensor, y_train: torch.Tensor, include_fiber: bool = True) -> dict:
    """SPEC3_ADDENDUM.md section 18 witnesses, regenerated from the model's
    CURRENT weights (same no-gradient contract as _recompute_epoch_witnesses).

    nc1_z, r_z, mean_margin, min_margin, beta_mean_margin, d_box always
    computed. equiangle_dev_z only at num_classes == 3 (section 18.3).
    Fiber decomposition (section 18.4) computed unless include_fiber=False.
    """
    model.eval()
    with torch.no_grad():
        psi = model.state(x_train).detach().cpu().numpy()
    rhos = statevectors_to_rhos(psi)
    assert_physical_rhos(rhos)

    labels = y_train.detach().cpu().numpy()
    z = z_features(rhos, model.n_qubits, model.num_classes)
    nc1z = nc1_z_metrics(z, labels)
    m = margins(z, labels)
    beta = float(model.beta.detach().cpu())

    out = {
        "nc1_z": nc1z["nc1_z"],
        "r_z": nc1z["r_z"],
        "trace_w": nc1z["trace_w"],
        "trace_b": nc1z["trace_b"],
        "mean_margin": m["mean_margin"],
        "min_margin": m["min_margin"],
        "beta_mean_margin": beta * m["mean_margin"],
        "d_box": d_box(nc1z["zbar_c"]),
        "zbar_c": nc1z["zbar_c"].tolist(),
    }
    if model.num_classes == 3:
        out["equiangle_dev_z"] = equiangle_dev_z(z, labels)

    if include_fiber:
        classes = np.unique(labels)
        class_idx = {c: idx for idx, c in enumerate(classes)}
        means, _global_mean = class_means(rhos, labels)
        deviations = np.stack([rhos[i] - means[class_idx[labels[i]]] for i in range(len(labels))])
        basis = measured_subspace_basis(model.n_qubits, model.num_classes)
        fiber = fiber_decomposition(deviations, basis)
        out.update(fiber)
        out["total_var"] = fiber["v_meas"] + fiber["v_fiber"]
        # PREDICTIONS.md Stage 17b (HR6/HN5): per-class-mean purity and
        # pairwise overlap, the same rho_bar_c's already built above for the
        # fiber decomposition -- no new formula, reuses SPEC2 §11.3's purity
        # and overlap_offdiag applied to class-mean (not per-sample) states.
        out["mean_class_purity"] = float(np.mean([purity(means[c]) for c in range(means.shape[0])]))
        out["class_mean_overlap"] = overlap_offdiag(means)

    return out


def load_stage9_manifest(manifest_path: Path = STAGE9_MANIFEST_PATH) -> dict[str, dict[str, str]]:
    with open(manifest_path) as f:
        return json.load(f)


def _stage9_variant_overrides(sweep_config_path: Path = STAGE9_SWEEP_CONFIG) -> dict[str, dict]:
    sweep_config = yaml.safe_load(open(sweep_config_path))
    return {v["label"]: v.get("overrides", {}) for v in sweep_config["variants"]}


def reanalyze_l8_anchor_zspace(
    results_dir: Path = RESULTS_DIR, configs_dir: Path = CONFIGS_DIR
) -> dict[str, list[dict]]:
    """SPEC3_ADDENDUM.md section 18, Phase 2 L=8 anchor (5 seeds, PLAYBOOK3.md
    Stage 12 Task C(i)): section 18 witnesses at epoch 0 (seeded
    reconstruction) and every checkpointed epoch."""
    base_config = yaml.safe_load(open(configs_dir / STAGE9_BASE_CONFIG))
    overrides = _stage9_variant_overrides()["L8"]
    config = merge_overrides(base_config, overrides)

    manifest = load_stage9_manifest()
    out: dict[str, list[dict]] = {}
    for seed_str, run_id in sorted(manifest["L8"].items(), key=lambda kv: int(kv[0])):
        seed = int(seed_str)
        model, x_train, y_train = _build_model_and_data(config, seed)

        records = [{"epoch": 0, **_recompute_zspace_witnesses(model, x_train, y_train)}]
        ckpt_dir = results_dir / run_id / "checkpoints"
        for ckpt_path in sorted(ckpt_dir.glob("epoch_*.pt")):
            epoch = int(ckpt_path.stem.split("_")[1])
            model.load_state_dict(torch.load(ckpt_path, weights_only=True))
            records.append({"epoch": epoch, **_recompute_zspace_witnesses(model, x_train, y_train)})

        out[run_id] = records
    return out


STAGE15_N6_BASE_CONFIG = "stage15_n6_base.yaml"
STAGE15_N6_DEPTH_PROBE_SWEEP_CONFIG = Path("configs/sweeps/stage15_n6_depth_probe.yaml")
STAGE15_N6_CAMPAIGN_SWEEP_CONFIG = Path("configs/sweeps/stage15_n6_campaign.yaml")


def _zspace_checkpoint_walk(
    config: dict, seed: int, run_id: str, results_dir: Path = RESULTS_DIR, include_fiber: bool = True
) -> list[dict]:
    """SPEC3_ADDENDUM.md section 18 witnesses at epoch 0 (seeded
    reconstruction) and every checkpointed epoch, for one (config, seed,
    run_id) -- the same no-new-training contract as
    `reanalyze_l8_anchor_zspace`, generalized so it isn't tied to the Stage
    9/12 L8 anchor's specific config/manifest (Stage 15's n=6 campaign)."""
    model, x_train, y_train = _build_model_and_data(config, seed)
    records = [{"epoch": 0, **_recompute_zspace_witnesses(model, x_train, y_train, include_fiber=include_fiber)}]
    ckpt_dir = results_dir / run_id / "checkpoints"
    for ckpt_path in sorted(ckpt_dir.glob("epoch_*.pt")):
        epoch = int(ckpt_path.stem.split("_")[1])
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        records.append(
            {"epoch": epoch, **_recompute_zspace_witnesses(model, x_train, y_train, include_fiber=include_fiber)}
        )
    return records


def reanalyze_n6_sweep_zspace(
    manifest: dict[str, dict[str, str]],
    variant_overrides: dict[str, dict],
    results_dir: Path = RESULTS_DIR,
    configs_dir: Path = CONFIGS_DIR,
) -> dict[str, dict[str, list[dict]]]:
    """PLAYBOOK3.md Stage 15 Task B: SPEC3 section 18 witnesses at epoch 0
    and every checkpointed epoch, for every (variant label, seed) in
    `manifest` (a stage15_n6_* sweep manifest.json), off
    `configs/stage15_n6_base.yaml` + each variant's overrides. Used for both
    the depth-probe TPT check (HN4) and the 5-seed full-witness campaign
    (HN1-HN3)."""
    base_config = yaml.safe_load(open(configs_dir / STAGE15_N6_BASE_CONFIG))
    out: dict[str, dict[str, list[dict]]] = {}
    for label, seed_map in manifest.items():
        config = merge_overrides(base_config, variant_overrides[label])
        out[label] = {}
        for seed_str, run_id in sorted(seed_map.items(), key=lambda kv: int(kv[0])):
            seed = int(seed_str)
            out[label][seed_str] = _zspace_checkpoint_walk(config, seed, run_id, results_dir)
    return out


def select_n6_campaign_depths(
    manifest: dict[str, dict[str, str]], results_dir: Path = RESULTS_DIR
) -> dict[str, int | None]:
    """PLAYBOOK3.md Stage 15 Task B: given the n=6 depth-probe manifest
    (labels `L{depth}_reupload_{true,false}`, 3 seeds each), returns the
    smallest depth per reupload arm with 3/3 probe seeds reaching TPT (per
    `compute_tpt_fractions`), for the 5-seed full-witness campaign. An arm
    with no fully-interpolating depth among the probed values maps to
    `None` (reported factually, not silently defaulted to the largest
    depth)."""
    tpt_fractions = compute_tpt_fractions(manifest, results_dir)
    depths_by_label = {
        v["label"]: v["value"]
        for v in yaml.safe_load(open(STAGE15_N6_DEPTH_PROBE_SWEEP_CONFIG))["variants"]
    }

    out: dict[str, int | None] = {}
    for arm in ("reupload_true", "reupload_false"):
        candidates = sorted(
            (depths_by_label[label], label) for label in manifest if label.endswith(f"_{arm}")
        )
        selected = next((depth for depth, label in candidates if tpt_fractions[label] == 1.0), None)
        out[arm] = selected
    return out


def reanalyze_stage9_sweep_zspace_final(
    results_dir: Path = RESULTS_DIR, configs_dir: Path = CONFIGS_DIR
) -> dict[str, dict[str, dict]]:
    """SPEC3_ADDENDUM.md section 18, full Stage 9 L sweep (PLAYBOOK3.md
    Stage 12 Task C(ii)): section 18 witnesses at the FINAL checkpointed
    epoch only, per (variant label, seed)."""
    base_config = yaml.safe_load(open(configs_dir / STAGE9_BASE_CONFIG))
    variant_overrides = _stage9_variant_overrides()
    manifest = load_stage9_manifest()

    out: dict[str, dict[str, dict]] = {}
    for label, seed_map in manifest.items():
        config = merge_overrides(base_config, variant_overrides[label])
        out[label] = {}
        for seed_str, run_id in sorted(seed_map.items(), key=lambda kv: int(kv[0])):
            seed = int(seed_str)
            model, x_train, y_train = _build_model_and_data(config, seed)
            ckpt_dir = results_dir / run_id / "checkpoints"
            final_ckpt = sorted(ckpt_dir.glob("epoch_*.pt"))[-1]
            model.load_state_dict(torch.load(final_ckpt, weights_only=True))
            epoch = int(final_ckpt.stem.split("_")[1])
            out[label][seed_str] = {
                "epoch": epoch,
                "run_id": run_id,
                **_recompute_zspace_witnesses(model, x_train, y_train),
            }
    return out


def reanalyze_c2_anchor_zspace(
    results_dir: Path = RESULTS_DIR, configs_dir: Path = CONFIGS_DIR
) -> dict[str, list[dict]]:
    """SPEC3_ADDENDUM.md section 18, Phase 1 C=2 anchor (PLAYBOOK3.md Stage 12
    Task C(iii)): d_box, NC1_z, margins ONLY -- no equiangle_dev_z (C=2 is
    degenerate for angle witnesses per section 18.3) and no fiber
    decomposition (task scope excludes it for this arm)."""
    config = yaml.safe_load(open(configs_dir / C2_ANCHOR_CONFIG))
    out: dict[str, list[dict]] = {}
    for run_id in discover_c2_anchor_run_ids(results_dir):
        seed = _parse_seed(run_id)
        model, x_train, y_train = _build_model_and_data(config, seed)

        records = [
            {"epoch": 0, **_recompute_zspace_witnesses(model, x_train, y_train, include_fiber=False)}
        ]
        ckpt_dir = results_dir / run_id / "checkpoints"
        for ckpt_path in sorted(ckpt_dir.glob("epoch_*.pt")):
            epoch = int(ckpt_path.stem.split("_")[1])
            model.load_state_dict(torch.load(ckpt_path, weights_only=True))
            records.append(
                {"epoch": epoch, **_recompute_zspace_witnesses(model, x_train, y_train, include_fiber=False)}
            )

        out[run_id] = records
    return out


def reanalyze_random_init_null_zspace(
    results_dir: Path = RESULTS_DIR, configs_dir: Path = CONFIGS_DIR
) -> dict[str, list[dict]]:
    """SPEC3_ADDENDUM.md section 18, 10-seed untrained null (PLAYBOOK3.md
    Stage 12 Task C(iv)): ALL section 18 metrics (including fiber
    decomposition) at the single seeded-reconstruction epoch 0 (this config
    trains 0 epochs and saves no checkpoints, identical procedure to
    reanalyze_random_init_null)."""
    config = yaml.safe_load(open(configs_dir / RANDOM_INIT_CONFIG))
    out: dict[str, list[dict]] = {}
    for run_id in discover_random_init_run_ids(results_dir):
        seed = _parse_seed(run_id)
        model, x_train, y_train = _build_model_and_data(config, seed)
        out[run_id] = [{"epoch": 0, **_recompute_zspace_witnesses(model, x_train, y_train)}]
    return out


def build_stage12_supplement_records(
    l8_records: dict[str, list[dict]], results_dir: Path = RESULTS_DIR
) -> dict[str, list[dict]]:
    """PLAYBOOK3.md Stage 13 Task 1: annotates `l8_records` (already computed
    by `reanalyze_l8_anchor_zspace`, no new checkpoint reload) with the two
    fields its three diagnostic figures need beyond what's already there:

    - e0_epoch: read from each seed's original metrics.jsonl (TPT onset).
    - pairwise_cos: pairwise z-space cosines per epoch, recomputed from the
      already-stored zbar_c (same formula equiangle_dev_z uses internally).

    trace_w/trace_b/v_fiber/fiber_fraction are already present in
    l8_records (SPEC3 §18.1/18.4); no new training or simulation here.
    """
    out: dict[str, list[dict]] = {}
    for run_id, records in l8_records.items():
        with open(results_dir / run_id / "metrics.jsonl") as f:
            e0_epoch = None
            for line in f:
                r = json.loads(line)
                if r.get("e0_epoch") is not None:
                    e0_epoch = r["e0_epoch"]
                    break

        annotated = []
        for rec in records:
            zbar_c = np.array(rec["zbar_c"])
            zbar_g = zbar_c.mean(axis=0)
            m = zbar_c - zbar_g
            norms = np.linalg.norm(m, axis=1)
            cosines = [
                float(np.dot(m[i], m[j]) / (norms[i] * norms[j]))
                for i in range(len(zbar_c))
                for j in range(i + 1, len(zbar_c))
                if norms[i] >= 1e-12 and norms[j] >= 1e-12
            ]
            annotated.append({**rec, "e0_epoch": e0_epoch, "pairwise_cos": cosines})
        out[run_id] = annotated

    return out


def append_stage12_supplements(findings_path: Path = Path("STAGE12_FINDINGS.md")) -> None:
    """Appends a factual "## Supplements" section (no re-scoring of P1-P5)
    describing the three PLAYBOOK3.md Stage 13 Task 1 figures."""
    section = """
## Supplements (PLAYBOOK3.md Stage 13 Task 1, existing checkpoints only)

Three diagnostic figures added to the L=8 anchor (5 seeds) reanalysis, no new
training or checkpoints:

- `results/transition_L_L8_aggregate/figures/etf_formation.png`: d_etf
  (equiangle_dev_z) and pairwise z-space cosines vs epoch, E0 marked per seed
  -- shows when ETF geometry forms relative to TPT onset.
- `results/transition_L_L8_aggregate/figures/nc1z_decomposition.png`:
  tr(Sigma_W^z) and tr(Sigma_B^z) vs epoch separately (within-class shrink vs
  between-class growth), same runs.
- `results/transition_L_L8_aggregate/figures/fiber_dip_diagnostic.png`:
  V_fiber vs epoch against the untrained-null band, showing when the ~6% dip
  (P4 supplement) develops.
"""
    with open(findings_path, "a") as f:
        f.write(section)


def _load_final_train_record(run_id: str, results_dir: Path) -> dict:
    """Final logged train-split record for one run -- already contains the
    full-space witnesses (qnc1_fisher, purity_mean, overlap_offdiag, ...)
    computed live by train.py from model.state(), unaffected by readout
    family. No recomputation needed for these."""
    path = results_dir / run_id / "metrics.jsonl"
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    train_records = [r for r in records if r["split"] == "train"]
    return train_records[-1]


def _raw_features_from_basis(rhos: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    """Physical expectation values <rho_i, unnormalized basis operator>,
    recovered from an orthonormalized measured-subspace basis (each
    basis[j] = op_j / 2^(n/2), metrics.py measured_subspace_basis*): feature
    = 2^(n/2) * Tr(rho @ basis[j])."""
    n = rhos.shape[0]
    dim = rhos.shape[-1]
    scale = float(np.sqrt(dim))
    feats = np.zeros((n, len(basis)))
    for i in range(n):
        for j, b in enumerate(basis):
            feats[i, j] = float((np.trace(rhos[i] @ b) * scale).real)
    return feats


def _recompute_stage13_zspace_witnesses(model: VQC, x_train: torch.Tensor, y_train: torch.Tensor) -> dict:
    """SPEC3_ADDENDUM.md section 19 (Stage 13): per-arm section 18 z/fiber
    witnesses, generalizing the measured-subspace basis to each readout
    family (metrics.py measured_subspace_basis_r1/r2/r3). Feature-space NC1
    is computed on the arm's NATURAL measured-subspace feature vector
    (dimension C for R0/R3, 3C for R1, 3n+n(n-1)/2 for R2) -- the same
    operator basis the fiber decomposition below is built from, so it tests
    collapse of the measured subspace itself rather than the R1/R2 head's
    output.

    Also computes PREDICTIONS.md's Stage 13B H1 (loss-visible subspace split,
    R1/R2 only -- R0/R3 have no trainable head, wrow_dim == measured_dim
    trivially) and H3 (logit_nc1_z: NC1_z on the arm's actual model output,
    the true CE-loss input, distinct from feature_nc1_z above).
    """
    model.eval()
    with torch.no_grad():
        psi = model.state(x_train).detach().cpu().numpy()
        logits = model(x_train).detach().cpu().numpy()
    rhos = statevectors_to_rhos(psi)
    assert_physical_rhos(rhos)
    labels = y_train.detach().cpu().numpy()

    if model.readout_family == "R0":
        basis = measured_subspace_basis(model.n_qubits, model.num_classes)
    elif model.readout_family == "R1":
        basis = measured_subspace_basis_r1(model.n_qubits, model.num_classes)
    elif model.readout_family == "R2":
        basis = measured_subspace_basis_r2(model.n_qubits)
    else:
        assert model.measurement_weights is not None
        measurement_weights = model.measurement_weights.detach().cpu().numpy()
        basis = measured_subspace_basis_r3(model.n_qubits, model.num_classes, measurement_weights)

    features = _raw_features_from_basis(rhos, basis)
    nc1z = nc1_z_metrics(features, labels)
    logit_nc1z = nc1_z_metrics(logits, labels)

    classes = np.unique(labels)
    class_idx = {c: idx for idx, c in enumerate(classes)}
    means, _global_mean = class_means(rhos, labels)
    deviations = np.stack([rhos[i] - means[class_idx[labels[i]]] for i in range(len(labels))])
    fiber = fiber_decomposition(deviations, basis)
    isotropic_null = isotropic_fiber_null(model.n_qubits, len(basis))

    out = {
        "feature_nc1_z": nc1z["nc1_z"],
        "feature_r_z": nc1z["r_z"],
        "logit_nc1_z": logit_nc1z["nc1_z"],
        "isotropic_fiber_null": isotropic_null,
        "fiber_excess_above_null": fiber["fiber_fraction"] - isotropic_null,
        **fiber,
    }

    if model.readout_family in ("R1", "R2"):
        head_weight = model.head.weight.detach().cpu().numpy()
        split = loss_visible_subspace_decomposition(deviations, basis, head_weight)
        out["wrow_dim"] = split["wrow_dim"]
        out["complement_dim"] = split["complement_dim"]
        out["v_wrow"] = split["v_wrow"]
        out["v_complement"] = split["v_complement"]
        out["wrow_per_dim"] = split["v_wrow"] / split["wrow_dim"] if split["wrow_dim"] > 0 else float("nan")
        out["complement_per_dim"] = (
            split["v_complement"] / split["complement_dim"] if split["complement_dim"] > 0 else float("nan")
        )
        out["meas_per_dim"] = fiber["v_meas"] / len(basis)

    return out


def reanalyze_stage13_expressivity(
    manifest: dict[str, dict[str, str]],
    results_dir: Path = RESULTS_DIR,
    configs_dir: Path = CONFIGS_DIR,
    sweep_config_path: Path = STAGE13_SWEEP_CONFIG,
) -> dict[str, dict[str, dict]]:
    """SPEC3_ADDENDUM.md section 19 (Stage 13 Task 2), 4 readout families x 5
    seeds: full-space witnesses read straight from the final logged epoch
    (train.py already computes them from model.state(), unaffected by
    readout family), plus section 18 z/fiber witnesses recomputed from the
    final checkpoint with the arm's generalized measured-subspace basis. No
    new training.

    `sweep_config_path` selects which sweep YAML's per-arm overrides
    (layers/readout_family/measurement_layers) to apply -- defaults to the
    original matched-M Stage 13 sweep; PLAYBOOK3.md Stage 13B's fixed-depth
    rerun passes `configs/sweeps/stage13b_fixed_depth.yaml` instead.
    """
    base_config = yaml.safe_load(open(configs_dir / STAGE13_BASE_CONFIG))
    variant_overrides = _stage9_variant_overrides(sweep_config_path)

    out: dict[str, dict[str, dict]] = {}
    for label, seed_map in manifest.items():
        config = merge_overrides(base_config, variant_overrides[label])
        out[label] = {}
        for seed_str, run_id in sorted(seed_map.items(), key=lambda kv: int(kv[0])):
            seed = int(seed_str)
            model, x_train, y_train = _build_model_and_data(config, seed)
            ckpt_dir = results_dir / run_id / "checkpoints"
            final_ckpt = sorted(ckpt_dir.glob("epoch_*.pt"))[-1]
            model.load_state_dict(torch.load(final_ckpt, weights_only=True))
            epoch = int(final_ckpt.stem.split("_")[1])

            full_space_record = _load_final_train_record(run_id, results_dir)
            zspace = _recompute_stage13_zspace_witnesses(model, x_train, y_train)

            out[label][seed_str] = {
                "epoch": epoch,
                "run_id": run_id,
                "readout_family": label,
                "n_layers": model.n_layers,
                "n_params": model.n_params,
                "m_params_measurement": model.m_params_measurement,
                "qnc1_fisher": full_space_record["qnc1_fisher"],
                "purity_mean": full_space_record["purity_mean"],
                "overlap_offdiag": full_space_record["overlap_offdiag"],
                **zspace,
            }
    return out


def _r0_concentration_factor(model: VQC) -> float:
    """PREDICTIONS.md Stage 14 witness-set note: R0's fixed-Z_c readout has
    no trainable head, so the measured subspace IS the loss-visible subspace
    (wrow_dim == meas_dim == num_classes, complement_dim == 0) by
    construction -- concentration_factor is trivially 1.0, preregistered as
    a pipeline sanity check rather than a real hypothesis test. Asserts the
    construction holds instead of calling loss_visible_subspace_decomposition
    (which requires model.head, absent for R0)."""
    assert model.readout_family == "R0", model.readout_family
    assert model.head is None
    return 1.0


def _recompute_stage14_r0_witnesses(model: VQC, x_train: torch.Tensor, y_train: torch.Tensor) -> dict:
    """PREDICTIONS.md Stage 14 witness set (arms 1, 2, 3, 6, baseline -- all
    R0 readout): trace_w, trace_b, nc1_z (via the existing SPEC3 section 18.1
    `_recompute_zspace_witnesses`, fiber decomposition not needed here) plus
    the concentration_factor sanity check."""
    z = _recompute_zspace_witnesses(model, x_train, y_train, include_fiber=False)
    return {
        "trace_w": z["trace_w"],
        "trace_b": z["trace_b"],
        "nc1_z": z["nc1_z"],
        "concentration_factor": _r0_concentration_factor(model),
    }


def reanalyze_stage14_baseline(
    manifest_path: Path = Path("results/sweeps/stage13_expressivity_20260714-111146/manifest.json"),
    results_dir: Path = RESULTS_DIR,
    configs_dir: Path = CONFIGS_DIR,
) -> dict[str, dict]:
    """PREDICTIONS.md Stage 14 "Baseline": the L=8 R0 arm of the existing
    Stage 13 matched-M sweep, recomputed fresh here (not reused from
    STAGE12_FINDINGS.md, which used a different base config,
    stage9_transition.yaml vs stage13_expressivity.yaml) for an apples-to-
    apples comparison against the new mechanism arms. No new training."""
    base_config = yaml.safe_load(open(configs_dir / STAGE13_BASE_CONFIG))
    overrides = _stage9_variant_overrides(STAGE13_SWEEP_CONFIG)["R0"]
    config = merge_overrides(base_config, overrides)
    with open(manifest_path) as f:
        manifest = json.load(f)

    out: dict[str, dict] = {}
    for seed_str, run_id in sorted(manifest["R0"].items(), key=lambda kv: int(kv[0])):
        seed = int(seed_str)
        model, x_train, y_train = _build_model_and_data(config, seed)
        epoch0 = _recompute_stage14_r0_witnesses(model, x_train, y_train)
        ckpt_dir = results_dir / run_id / "checkpoints"
        final_ckpt = sorted(ckpt_dir.glob("epoch_*.pt"))[-1]
        model.load_state_dict(torch.load(final_ckpt, weights_only=True))
        out[seed_str] = {
            "run_id": run_id,
            "epoch0": epoch0,
            **_recompute_stage14_r0_witnesses(model, x_train, y_train),
        }
    return out


def reanalyze_stage14_mechanism_arms(
    manifest: dict[str, dict[str, str]],
    results_dir: Path = RESULTS_DIR,
    configs_dir: Path = CONFIGS_DIR,
    sweep_config_path: Path = STAGE14_MECHANISM_SWEEP_CONFIG,
) -> dict[str, dict[str, dict]]:
    """PREDICTIONS.md Stage 14 arms 1 (beta_max500), 3 (label_smoothing), 6
    (frozen_beta): final-checkpoint witness set per (arm label, seed). No new
    training -- `manifest` comes from a prior `python -m qnc.sweep` run of
    configs/sweeps/stage14_mechanism_arms.yaml."""
    base_config = yaml.safe_load(open(configs_dir / STAGE14_MECHANISM_BASE_CONFIG))
    variant_overrides = _stage9_variant_overrides(sweep_config_path)

    out: dict[str, dict[str, dict]] = {}
    for label, seed_map in manifest.items():
        config = merge_overrides(base_config, variant_overrides[label])
        out[label] = {}
        for seed_str, run_id in sorted(seed_map.items(), key=lambda kv: int(kv[0])):
            seed = int(seed_str)
            model, x_train, y_train = _build_model_and_data(config, seed)
            epoch0 = _recompute_stage14_r0_witnesses(model, x_train, y_train)
            ckpt_dir = results_dir / run_id / "checkpoints"
            final_ckpt = sorted(ckpt_dir.glob("epoch_*.pt"))[-1]
            model.load_state_dict(torch.load(final_ckpt, weights_only=True))
            out[label][seed_str] = {
                "run_id": run_id,
                "epoch0": epoch0,
                **_recompute_stage14_r0_witnesses(model, x_train, y_train),
            }
    return out


def reanalyze_stage14_long_horizon(
    run_id: str,
    results_dir: Path = RESULTS_DIR,
    configs_dir: Path = CONFIGS_DIR,
    seed: int = 0,
) -> list[dict]:
    """PREDICTIONS.md Stage 14 arm 2 (long horizon, 5000 epochs, 1 seed):
    trace_w/trace_b/nc1_z at every checkpointed epoch (dense-logged every 50
    epochs throughout, per configs/stage14_long_horizon.yaml). No new
    training -- `run_id` comes from a prior `python -m qnc.run` invocation."""
    config = yaml.safe_load(open(configs_dir / STAGE14_LONG_HORIZON_CONFIG))
    model, x_train, y_train = _build_model_and_data(config, seed)

    records = [{"epoch": 0, **_recompute_stage14_r0_witnesses(model, x_train, y_train)}]
    ckpt_dir = results_dir / run_id / "checkpoints"
    for ckpt_path in sorted(ckpt_dir.glob("epoch_*.pt")):
        epoch = int(ckpt_path.stem.split("_")[1])
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        records.append({"epoch": epoch, **_recompute_stage14_r0_witnesses(model, x_train, y_train)})
    return records


def spearman_trace_w_drift(records: list[dict], e0_epoch: int) -> dict:
    """PREDICTIONS.md Stage 14 arm 2: Spearman(trace_w, log(epoch)) over
    checkpoints strictly after e0_epoch (post-TPT), testing for a downward
    drift the scaling-escape hypothesis predicts should NOT be present."""
    post_tpt = [r for r in records if r["epoch"] > e0_epoch]
    log_epochs = np.log(np.array([r["epoch"] for r in post_tpt], dtype=float))
    trace_w = np.array([r["trace_w"] for r in post_tpt], dtype=float)
    rho, p_value = spearmanr(log_epochs, trace_w)
    return {"rho": float(rho), "p_value": float(p_value), "n": len(post_tpt)}


def spearman_trace_w_vs_layers(
    sweep_final: dict[str, dict[str, dict]], sweep_config_path: Path = STAGE9_SWEEP_CONFIG
) -> dict:
    """PREDICTIONS.md Stage 14B HR4: Spearman(trace_w, L) across the Stage 9
    L sweep's final-epoch trace_w (SPEC3 section 18.1), using
    `reanalyze_stage9_sweep_zspace_final`'s output (no new training). Tests
    whether deeper reuploading circuits contract more in z-space
    within-class variance."""
    sweep_config = yaml.safe_load(open(sweep_config_path))
    value_by_label = {v["label"]: v["value"] for v in sweep_config["variants"]}
    layers: list[float] = []
    trace_w: list[float] = []
    for label, seed_map in sweep_final.items():
        for rec in seed_map.values():
            layers.append(value_by_label[label])
            trace_w.append(rec["trace_w"])
    rho, p_value = spearmanr(layers, trace_w)
    return {"rho": float(rho), "p_value": float(p_value), "n": len(layers)}


def _epoch0_and_final_trace_w(
    config: dict, seed: int, run_id: str, results_dir: Path = RESULTS_DIR
) -> dict:
    """Checkpoints-only epoch0 (seeded reconstruction) + final-checkpoint
    (state_dict reload) SPEC3 section 18 witnesses for one run, no gradient
    steps. Used by HR5b (epoch0 total_var/trace_w) and HR5c (epoch0 vs final
    trace_w inheritance)."""
    model, x_train, y_train = _build_model_and_data(config, seed)
    epoch0 = _recompute_zspace_witnesses(model, x_train, y_train, include_fiber=True)

    ckpt_dir = results_dir / run_id / "checkpoints"
    final_ckpt = sorted(ckpt_dir.glob("epoch_*.pt"))[-1]
    model.load_state_dict(torch.load(final_ckpt, weights_only=True))
    final = _recompute_zspace_witnesses(model, x_train, y_train, include_fiber=True)

    return {
        "run_id": run_id,
        "epoch0_total_var": epoch0["total_var"],
        "epoch0_trace_w": epoch0["trace_w"],
        "final_total_var": final["total_var"],
        "final_trace_w": final["trace_w"],
    }


def reanalyze_hr5_diagnosis(
    results_dir: Path = RESULTS_DIR, configs_dir: Path = CONFIGS_DIR
) -> dict[str, dict]:
    """PREDICTIONS.md Stage 15 "HR5 diagnosis" (checkpoints only, no new
    training): epoch0 + final trace_w/total_var for the 15 blob_std runs
    (HR5b + HR5c) and the 5 reupload_false runs (HR5c), by blob_std arm
    label / seed. The L8 anchor's epoch0/final trace_w for HR5c's pooled
    n=25 is read separately from `reanalyze_l8_anchor_zspace`'s output
    (already computed, no new checkpoint reload needed here)."""
    base_config = yaml.safe_load(open(configs_dir / STAGE14_MECHANISM_BASE_CONFIG))
    blob_std_overrides = {
        v["label"]: v["overrides"] for v in yaml.safe_load(open(STAGE14B_BLOB_STD_SWEEP_CONFIG))["variants"]
    }
    manifest = json.load(open(STAGE14B_BLOB_STD_MANIFEST_PATH))

    blob_std_records: dict[str, list[dict]] = {}
    for label, seed_map in manifest.items():
        config = merge_overrides(base_config, blob_std_overrides[label])
        blob_std_records[label] = [
            _epoch0_and_final_trace_w(config, int(seed_str), run_id, results_dir)
            for seed_str, run_id in sorted(seed_map.items(), key=lambda kv: int(kv[0]))
        ]

    reupload_false_overrides = yaml.safe_load(open(STAGE14B_REUPLOAD_FALSE_SWEEP_CONFIG))["variants"][0][
        "overrides"
    ]
    reupload_false_config = merge_overrides(base_config, reupload_false_overrides)
    reupload_false_records = [
        _epoch0_and_final_trace_w(reupload_false_config, seed, run_id, results_dir)
        for seed, run_id in enumerate(STAGE14B_REUPLOAD_FALSE_RUN_IDS)
    ]

    return {"blob_std": blob_std_records, "reupload_false": reupload_false_records}


def compute_hr5b(blob_std_records: dict[str, list[dict]]) -> dict:
    """PREDICTIONS.md HR5b: relative range of arm-mean epoch0 total_var and
    epoch0 trace_w across the three blob_std arms, `(max-min)/mean(3 arm
    means)`. CONFIRMED if both < 20%, PARTIAL if one, REFUTED if neither."""
    arm_means_total_var = {
        label: float(np.mean([r["epoch0_total_var"] for r in records])) for label, records in blob_std_records.items()
    }
    arm_means_trace_w = {
        label: float(np.mean([r["epoch0_trace_w"] for r in records])) for label, records in blob_std_records.items()
    }

    def relative_range(arm_means: dict[str, float]) -> float:
        values = list(arm_means.values())
        return (max(values) - min(values)) / np.mean(values)

    rel_range_total_var = float(relative_range(arm_means_total_var))
    rel_range_trace_w = float(relative_range(arm_means_trace_w))
    pass_total_var = rel_range_total_var < 0.20
    pass_trace_w = rel_range_trace_w < 0.20

    if pass_total_var and pass_trace_w:
        verdict = "CONFIRMED"
    elif pass_total_var or pass_trace_w:
        verdict = "PARTIAL"
    else:
        verdict = "REFUTED"

    return {
        "arm_mean_epoch0_total_var": arm_means_total_var,
        "arm_mean_epoch0_trace_w": arm_means_trace_w,
        "relative_range_total_var": rel_range_total_var,
        "relative_range_trace_w": rel_range_trace_w,
        "verdict": verdict,
    }


def compute_hr5c(
    blob_std_records: dict[str, list[dict]],
    reupload_false_records: list[dict],
    l8_anchor_records: dict[str, list[dict]],
) -> dict:
    # l8_anchor_records: reanalyze_l8_anchor_zspace() output, {run_id: [epoch-ordered records]}
    """PREDICTIONS.md HR5c: pooled Spearman correlation of final trace_w vs
    that SAME run's epoch0 trace_w, n=25 (15 blob_std + 5 reupload_false + 5
    L8-anchor). CONFIRMED if rho>0 and p<0.05, PARTIAL if rho>0 and p>=0.05,
    REFUTED if rho<=0."""
    epoch0: list[float] = []
    final: list[float] = []
    run_ids: list[str] = []

    for records in blob_std_records.values():
        for r in records:
            epoch0.append(r["epoch0_trace_w"])
            final.append(r["final_trace_w"])
            run_ids.append(r["run_id"])

    for r in reupload_false_records:
        epoch0.append(r["epoch0_trace_w"])
        final.append(r["final_trace_w"])
        run_ids.append(r["run_id"])

    for run_id, records in l8_anchor_records.items():
        epoch0.append(records[0]["trace_w"])
        final.append(records[-1]["trace_w"])
        run_ids.append(run_id)

    rho, p_value = spearmanr(epoch0, final)
    rho = float(rho)
    p_value = float(p_value)

    if rho > 0 and p_value < 0.05:
        verdict = "CONFIRMED"
    elif rho > 0:
        verdict = "PARTIAL"
    else:
        verdict = "REFUTED"

    return {
        "n": len(epoch0),
        "run_ids": run_ids,
        "rho": rho,
        "p_value": p_value,
        "verdict": verdict,
    }


def compute_tpt_fractions(
    manifest: dict[str, dict[str, str]], results_dir: Path = RESULTS_DIR
) -> dict[str, float]:
    """PREDICTIONS.md Stage 14 arm 4 (transition localization, L in
    {3,5,6}): fraction of seeds reaching TPT per arm, read straight from
    each run's final logged `tpt_reached` field -- no checkpoint reload, no
    new metrics."""
    out: dict[str, float] = {}
    for label, seed_map in manifest.items():
        reached = [_load_final_train_record(run_id, results_dir)["tpt_reached"] for run_id in seed_map.values()]
        out[label] = float(np.mean(reached))
    return out


def reanalyze_stage14_single_input_qfi(
    l_values: tuple[str, ...] = STAGE14_QFI_ARM_LABELS,
    seed: int = 0,
    configs_dir: Path = CONFIGS_DIR,
    n_inputs: int = 4,
) -> dict[str, dict]:
    """PREDICTIONS.md Stage 14 arm 5 (single-input QFI rank fix, original
    SPEC3_ADDENDUM.md section 20 arm 5, unchanged): epoch-0 seeded
    reconstruction (no training, no checkpoint) off the existing Stage 9
    `transition_L` configs, QFI rank recomputed per SINGLE input (batch=1)
    for `n_inputs` separate training examples, at L in `l_values`."""
    base_config = yaml.safe_load(open(configs_dir / STAGE9_BASE_CONFIG))
    variant_overrides = _stage9_variant_overrides()

    out: dict[str, dict] = {}
    for label in l_values:
        config = merge_overrides(base_config, variant_overrides[label])
        model, x_train, _y_train = _build_model_and_data(config, seed)
        model.eval()
        ranks = []
        for i in range(n_inputs):
            x_single = x_train[i : i + 1]
            psi_fn = lambda x=x_single: model.state(x)
            g = qfi_matrix(psi_fn, model.weights)
            ranks.append(qfi_rank(g))
        out[label] = {"ranks": ranks, "mean_rank": float(np.mean(ranks))}
    return out


def build_four_witness_comparison(
    c2_records: dict[str, list[dict]], null_records: dict[str, list[dict]]
) -> str:
    """SPEC2 §11 task (c): Mann-Whitney U, trained (final checkpointed epoch,
    n=5) vs null (epoch 0, n=10), for all four witnesses."""
    trained_finals = [recs[-1] for recs in c2_records.values()]
    null_finals = [recs[-1] for recs in null_records.values()]

    header = "| metric | trained (C=2 anchor, final epoch, n=5) | null (random-init, n=10) | Mann-Whitney U p-value |"
    sep = "|---|---|---|---|"
    lines = [header, sep]
    for field in FOUR_WITNESS_FIELDS:
        a = np.array([r[field] for r in trained_finals], dtype=float)
        b = np.array([r[field] for r in null_finals], dtype=float)
        p_value = mannwhitneyu(a, b).pvalue
        lines.append(
            f"| {field} | {a.mean():.6g} +/- {a.std():.6g} | {b.mean():.6g} +/- {b.std():.6g} | {p_value:.4g} |"
        )
    return "\n".join(lines)


def _verdict(condition: bool, confirmed: str, refuted: str) -> str:
    return confirmed if condition else refuted


def build_stage12_findings(
    l8_records: dict[str, list[dict]],
    c2_records: dict[str, list[dict]],
    null_records: dict[str, list[dict]],
    stage9_final: dict[str, dict[str, dict]],
) -> str:
    """PLAYBOOK3.md Stage 12 Task D: score every PREDICTIONS.md prediction
    (P1-P5) CONFIRMED / REFUTED / PARTIAL against the recomputed section 18
    numbers."""
    l8_finals = [recs[-1] for recs in l8_records.values()]
    l8_inits = [recs[0] for recs in l8_records.values()]
    c2_finals = [recs[-1] for recs in c2_records.values()]
    c2_inits = [recs[0] for recs in c2_records.values()]

    def _ratio(field, finals, inits):
        f = np.array([r[field] for r in finals], dtype=float)
        i = np.array([r[field] for r in inits], dtype=float)
        return f, i, float(np.mean(f)) / float(np.mean(i)) if np.mean(i) != 0 else float("nan")

    l8_nc1z_f, l8_nc1z_i, l8_nc1z_ratio = _ratio("nc1_z", l8_finals, l8_inits)
    l8_rz_f, l8_rz_i, l8_rz_ratio = _ratio("r_z", l8_finals, l8_inits)
    c2_nc1z_f, c2_nc1z_i, c2_nc1z_ratio = _ratio("nc1_z", c2_finals, c2_inits)
    c2_rz_f, c2_rz_i, c2_rz_ratio = _ratio("r_z", c2_finals, c2_inits)

    p1_hits = sum(r <= 0.1 for r in (l8_nc1z_ratio, l8_rz_ratio, c2_nc1z_ratio, c2_rz_ratio))
    p1_verdict = "CONFIRMED" if p1_hits == 4 else ("REFUTED" if p1_hits == 0 else "PARTIAL")

    l8_bm_f = np.mean([r["beta_mean_margin"] for r in l8_finals])
    l8_bm_i = np.mean([r["beta_mean_margin"] for r in l8_inits])
    c2_bm_f = np.mean([r["beta_mean_margin"] for r in c2_finals])
    c2_bm_i = np.mean([r["beta_mean_margin"] for r in c2_inits])
    p2_hits = sum([l8_bm_f > l8_bm_i, c2_bm_f > c2_bm_i])
    p2_verdict = "CONFIRMED" if p2_hits == 2 else ("REFUTED" if p2_hits == 0 else "PARTIAL")

    dboxn_mean = float(np.mean([r["d_box"] / 2.0 for r in l8_finals]))
    dboxn_std = float(np.std([r["d_box"] / 2.0 for r in l8_finals]))
    detf_mean = float(np.mean([r["equiangle_dev_z"] for r in l8_finals]))
    detf_std = float(np.std([r["equiangle_dev_z"] for r in l8_finals]))
    overlap = (dboxn_mean - dboxn_std <= detf_mean + detf_std) and (detf_mean - detf_std <= dboxn_mean + dboxn_std)
    p3_verdict = "PARTIAL" if overlap else ("CONFIRMED" if dboxn_mean < detf_mean else "REFUTED")

    l8_vmeas_f = float(np.mean([r["v_meas"] for r in l8_finals]))
    l8_vmeas_i = float(np.mean([r["v_meas"] for r in l8_inits]))
    l8_vfiber_f = float(np.mean([r["v_fiber"] for r in l8_finals]))
    l8_vfiber_std_f = float(np.std([r["v_fiber"] for r in l8_finals]))
    null_finals = [recs[-1] for recs in null_records.values()]
    null_vfiber_mean = float(np.mean([r["v_fiber"] for r in null_finals]))
    null_vfiber_std = float(np.std([r["v_fiber"] for r in null_finals]))
    l8_ff_f = float(np.mean([r["fiber_fraction"] for r in l8_finals]))

    p4_sub1 = l8_vmeas_f < l8_vmeas_i
    p4_sub2 = abs(l8_vfiber_f - null_vfiber_mean) <= 2 * (null_vfiber_std + l8_vfiber_std_f + 1e-12)
    p4_sub3 = l8_ff_f >= 0.9
    p4_hits = sum([p4_sub1, p4_sub2, p4_sub3])
    p4_verdict = "CONFIRMED" if p4_hits == 3 else ("REFUTED" if p4_hits <= 1 else "PARTIAL")

    null_nc1z = np.array([recs[0]["nc1_z"] for recs in null_records.values()], dtype=float)
    l8_final_nc1z = np.array([r["nc1_z"] for r in l8_finals], dtype=float)
    p5_p = float(mannwhitneyu(l8_final_nc1z, null_nc1z).pvalue)
    p5_verdict = _verdict(p5_p < 0.05, "CONFIRMED", "REFUTED")

    stage9_rows = []
    for label, seed_map in stage9_final.items():
        recs = list(seed_map.values())
        ff = np.array([r["fiber_fraction"] for r in recs], dtype=float)
        nc1z = np.array([r["nc1_z"] for r in recs], dtype=float)
        stage9_rows.append(
            f"| {label} | {nc1z.mean():.4g} +/- {nc1z.std():.4g} | {ff.mean():.4g} +/- {ff.std():.4g} |"
        )

    body = f"""# STAGE12_FINDINGS.md -- Preregistered reanalysis: measurement-subspace collapse

Implements PLAYBOOK3.md Stage 12 / SPEC3_ADDENDUM.md sections 17-18. Scored
strictly against PREDICTIONS.md (committed before any metric below was
computed). No new training runs -- every number is recomputed from existing
checkpoints via seeded reconstruction (epoch 0) and state_dict reload.

Mandatory citations (CLAUDE.md): Papyan, Han, Donoho (2020), PNAS. Du, Yang,
Tao, Hsieh (2023), PRL 131, 140601. San Sebastian, Canizo, Orus (2026),
arXiv:2602.08485.

## P1 -- NC1_z / r_z decay post-TPT ({p1_verdict})

| arm | metric | epoch-0 | final | final/init |
|---|---|---|---|---|
| L=8 anchor | NC1_z | {l8_nc1z_i.mean():.4g} +/- {l8_nc1z_i.std():.4g} | {l8_nc1z_f.mean():.4g} +/- {l8_nc1z_f.std():.4g} | {l8_nc1z_ratio:.4g} |
| L=8 anchor | r_z | {l8_rz_i.mean():.4g} +/- {l8_rz_i.std():.4g} | {l8_rz_f.mean():.4g} +/- {l8_rz_f.std():.4g} | {l8_rz_ratio:.4g} |
| C=2 anchor | NC1_z | {c2_nc1z_i.mean():.4g} +/- {c2_nc1z_i.std():.4g} | {c2_nc1z_f.mean():.4g} +/- {c2_nc1z_f.std():.4g} | {c2_nc1z_ratio:.4g} |
| C=2 anchor | r_z | {c2_rz_i.mean():.4g} +/- {c2_rz_i.std():.4g} | {c2_rz_f.mean():.4g} +/- {c2_rz_f.std():.4g} | {c2_rz_ratio:.4g} |

Threshold: final/init <= 0.1 (>= 1 order of magnitude decay) on all 4 rows
for CONFIRMED; 0 rows for REFUTED; otherwise PARTIAL.

## P2 -- beta * mean_margin grows post-TPT ({p2_verdict})

| arm | epoch-0 | final |
|---|---|---|
| L=8 anchor | {l8_bm_i:.4g} | {l8_bm_f:.4g} |
| C=2 anchor | {c2_bm_i:.4g} | {c2_bm_f:.4g} |

## P3 -- box vertices vs simplex ETF, C=3, L=8 anchor final epoch ({p3_verdict})

d_boxn (normalized d_box) = {dboxn_mean:.4g} +/- {dboxn_std:.4g}
d_etf (equiangle_dev_z)    = {detf_mean:.4g} +/- {detf_std:.4g}

## P4 -- fiber decomposition dichotomy, L=8 anchor ({p4_verdict})

- V_meas: epoch-0 {l8_vmeas_i:.4g} -> final {l8_vmeas_f:.4g} (decays: {p4_sub1})
- V_fiber: final {l8_vfiber_f:.4g} +/- {l8_vfiber_std_f:.4g} vs untrained-null
  {null_vfiber_mean:.4g} +/- {null_vfiber_std:.4g} (flat within 2 std: {p4_sub2})
- fiber_fraction: final {l8_ff_f:.4g} (>= 0.9: {p4_sub3})

## P5 -- untrained null shows no z-space collapse ({p5_verdict})

Mann-Whitney U, L=8-anchor final NC1_z (n=5, mean {l8_final_nc1z.mean():.4g})
vs untrained-null NC1_z (n=10, mean {null_nc1z.mean():.4g}): p = {p5_p:.4g}

## Stage 9 L-sweep, full-depth context (final epoch, all 8 depths)

| L | NC1_z (final) | fiber_fraction (final) |
|---|---|---|
{chr(10).join(stage9_rows)}

## Interpretation

Scored outcome per PLAYBOOK3.md Stage 12 gate: P1={p1_verdict}, P2={p2_verdict},
P3={p3_verdict}, P4={p4_verdict}, P5={p5_verdict}. Reported factually; no
softening of REFUTED/PARTIAL results (CLAUDE.md).
"""
    return body


def _max_relative_change(values: list[float]) -> float:
    """max_t abs(v[t] - v[0]) / abs(v[0]), the same "how far does this
    series drift from its own starting value" check used for HR2/HN1's
    total_var (PREDICTIONS.md), generalized to any scalar series."""
    v0 = values[0]
    if abs(v0) < 1e-12:
        return float("nan")
    return float(max(abs(v - v0) / abs(v0) for v in values))


def reanalyze_hr6_hn5_invariance(
    results_dir: Path = RESULTS_DIR, configs_dir: Path = CONFIGS_DIR
) -> dict[str, dict]:
    """PREDICTIONS.md Stage 17b (HR6/HN5): per-class-mean purity and pairwise
    overlap, recomputed at every checkpoint of the SAME reupload=false runs
    already used for HR2 (n=4) and HN1 (n=6) -- no new training, no new
    checkpoints. `mean_class_purity`/`class_mean_overlap` are already emitted
    per-epoch by `_recompute_zspace_witnesses` (reusing `purity`/
    `overlap_offdiag` on the class-mean states built for the fiber
    decomposition); this function just walks the checkpoints and reports the
    max relative change from each run's own epoch-0 value."""
    base_config = yaml.safe_load(open(configs_dir / STAGE14_MECHANISM_BASE_CONFIG))
    reupload_false_overrides = yaml.safe_load(open(STAGE14B_REUPLOAD_FALSE_SWEEP_CONFIG))["variants"][0][
        "overrides"
    ]
    n4_config = merge_overrides(base_config, reupload_false_overrides)

    n4_records: dict[str, dict] = {}
    for seed, run_id in enumerate(STAGE14B_REUPLOAD_FALSE_RUN_IDS):
        records = _zspace_checkpoint_walk(n4_config, seed, run_id, results_dir, include_fiber=True)
        purities = [r["mean_class_purity"] for r in records]
        overlaps = [r["class_mean_overlap"] for r in records]
        n4_records[run_id] = {
            "n_checkpoints": len(records),
            "max_rel_change_purity": _max_relative_change(purities),
            "max_rel_change_overlap": _max_relative_change(overlaps),
        }

    n6_base_config = yaml.safe_load(open(configs_dir / STAGE15_N6_BASE_CONFIG))
    n6_campaign_overrides = {
        v["label"]: v["overrides"] for v in yaml.safe_load(open(STAGE15_N6_CAMPAIGN_SWEEP_CONFIG))["variants"]
    }["L8_reupload_false"]
    n6_config = merge_overrides(n6_base_config, n6_campaign_overrides)
    n6_manifest = json.load(open(Path("results/sweeps/stage15_n6_campaign_20260716-163857/manifest.json")))[
        "L8_reupload_false"
    ]

    n6_records: dict[str, dict] = {}
    for seed_str, run_id in sorted(n6_manifest.items(), key=lambda kv: int(kv[0])):
        records = _zspace_checkpoint_walk(n6_config, int(seed_str), run_id, results_dir, include_fiber=True)
        purities = [r["mean_class_purity"] for r in records]
        overlaps = [r["class_mean_overlap"] for r in records]
        n6_records[run_id] = {
            "n_checkpoints": len(records),
            "max_rel_change_purity": _max_relative_change(purities),
            "max_rel_change_overlap": _max_relative_change(overlaps),
        }

    def _verdict(records: dict[str, dict]) -> str:
        purity_pass = [r["max_rel_change_purity"] < 1e-6 for r in records.values()]
        overlap_pass = [r["max_rel_change_overlap"] < 1e-6 for r in records.values()]
        if all(purity_pass) and all(overlap_pass):
            return "CONFIRMED"
        if (sum(purity_pass) >= 3 and sum(overlap_pass) >= 3) or (
            all(purity_pass) != all(overlap_pass)
        ):
            return "PARTIAL"
        return "REFUTED"

    return {
        "hr6_n4": n4_records,
        "hr6_verdict": _verdict(n4_records),
        "hn5_n6": n6_records,
        "hn5_verdict": _verdict(n6_records),
    }


def main() -> None:
    """Runs the full Stage 7 reanalysis and writes REANALYSIS.md."""
    fisher_by_run = recompute_fisher_from_jsonl()
    print(f"recomputed qnc1_fisher (JSONL-only) for {len(fisher_by_run)} runs")

    c2_records = reanalyze_c2_anchor()
    null_records = reanalyze_random_init_null()
    comparison_table = build_four_witness_comparison(c2_records, null_records)

    any_significant = False
    for field in FOUR_WITNESS_FIELDS:
        a = np.array([recs[-1][field] for recs in c2_records.values()], dtype=float)
        b = np.array([recs[-1][field] for recs in null_records.values()], dtype=float)
        if mannwhitneyu(a, b).pvalue < 0.05:
            any_significant = True

    verdict = (
        "At least one normalized witness separates trained from untrained "
        "states (p < 0.05) where raw qnc1_trace did not."
        if any_significant
        else "No normalized witness separates trained from untrained states "
        "(all p >= 0.05) at this C=2 anchor; the Phase 1 null result is not "
        "a metric artifact at this configuration."
    )

    body = f"""# REANALYSIS.md -- Stage 7: normalized-witness reanalysis

Implements PLAYBOOK2.md Stage 7 / SPEC2_ADDENDUM.md section 11. No new
training runs. See CLAUDE.md for mandatory citations (Papyan/Han/Donoho 2020;
Du/Yang/Tao/Hsieh 2023; San Sebastian/Canizo/Orus 2026) plus the Phase 2
additions (Larocca et al. 2023; Bowles et al. 2024; Han/Papyan/Donoho 2022).

## Method

- `qnc1_fisher` (SPEC2 section 11.1) recomputed for every Phase 1 run
  directly from already-logged `qnc1_hs` and `class_mean_norms` fields --
  no checkpoint reload needed. Note: this is algebraically identical to the
  `qnc1_rel` field already present in every Phase 1 JSONL line
  (`qnc1_rel = qnc1_hs / mean_c ||M_c||^2_HS`, the same S_W/S_B ratio);
  Stage 7 gives that existing quantity its SPEC2 name and headline role.
- `qnc1_ratio`, `overlap_offdiag`, `purity_mean` (SPEC2 sections 11.2-11.3)
  recomputed by reloading checkpoint `state_dict`s (5-seed C=2 anchor,
  `configs/stage2_vqc_c2_diagnostic.yaml`) or reconstructing the seeded
  pre-training state (10-seed random-init null,
  `configs/stage5_random_init.yaml`, which trains 0 epochs and saves no
  checkpoints) and regenerating states via `VQC.state()`. No optimizer step
  is ever taken.

## Four-witness trained-vs-null comparison

{comparison_table}

## Verdict

{verdict}
"""
    Path("REANALYSIS.md").write_text(body)
    print("wrote REANALYSIS.md")

    from qnc.figures import make_normalized_witness_trajectory_figure, make_four_witness_comparison_figure

    make_normalized_witness_trajectory_figure(c2_records)
    make_four_witness_comparison_figure(c2_records, null_records)
    print("wrote Stage 7 figures")

    l8_records = reanalyze_l8_anchor_zspace()
    print(f"recomputed section 18 z-space witnesses for {len(l8_records)} L=8 anchor seeds")
    stage9_final = reanalyze_stage9_sweep_zspace_final()
    print(f"recomputed section 18 z-space witnesses for {sum(len(v) for v in stage9_final.values())} Stage 9 sweep runs (final epoch)")
    c2_zspace_records = reanalyze_c2_anchor_zspace()
    null_zspace_records = reanalyze_random_init_null_zspace()

    findings = build_stage12_findings(l8_records, c2_zspace_records, null_zspace_records, stage9_final)
    Path("STAGE12_FINDINGS.md").write_text(findings)
    print("wrote STAGE12_FINDINGS.md")

    from qnc.figures import make_zspace_trajectory_figure, make_box_vs_etf_figure, make_fiber_headline_figure

    make_zspace_trajectory_figure(l8_records)
    make_box_vs_etf_figure(l8_records)
    make_fiber_headline_figure(l8_records)
    print("wrote Stage 12 figures")

    supplement_records = build_stage12_supplement_records(l8_records)
    append_stage12_supplements()
    print("appended Stage 12 supplements section to STAGE12_FINDINGS.md")

    from qnc.figures import (
        make_etf_formation_figure,
        make_nc1z_decomposition_figure,
        make_fiber_dip_diagnostic_figure,
    )

    make_etf_formation_figure(supplement_records)
    make_nc1z_decomposition_figure(supplement_records)
    make_fiber_dip_diagnostic_figure(supplement_records, null_zspace_records)
    print("wrote Stage 12 supplement figures")


if __name__ == "__main__":
    main()
