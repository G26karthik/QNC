"""Training loop.

Stage 1: classical training loop with per-epoch classical NC hook (SPEC.md §5).
Stage 2: VQC training loop with per-epoch quantum NC hook (SPEC.md §4),
implementing the Terminal Phase of Training rule in SPEC.md §3.
"""

import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from qnc.data import (
    load_bars_stripes_4x4,
    load_blobs3,
    load_fashion_mnist_subset,
    load_linearly_separable_4d,
    load_mnist_subset,
    load_two_moons,
    pca_encode,
    pca_encode_amplitude,
)
from qnc.logging_utils import JSONLLogger, generate_run_id
from qnc.metrics import (
    class_means,
    classical_nc1,
    classical_nc2,
    within_class_scatter_trace,
    overlap_offdiag,
    purity,
    qfi_matrix,
    qfi_rank,
    qnc1_fisher,
    qnc1_metrics,
    qnc1_ratio,
    qnc2_metrics,
    reduced_density,
    von_neumann_entropy,
)
from qnc.models import ClassicalMLP, VQC

E0_TOLERANCE = 1e-9


def is_e0(train_acc: float, tol: float = E0_TOLERANCE) -> bool:
    """True if train_acc counts as 100% (train error = 0), SPEC §3."""
    return train_acc >= 1.0 - tol


def compute_target_epochs(e0: int | None, total_epochs: int, tpt_multiple: int) -> int:
    """Training must continue to max(total_epochs, tpt_multiple * E0), SPEC §3."""
    if e0 is None:
        return total_epochs
    return max(total_epochs, tpt_multiple * e0)


def m_over_mc(n_params: int, n_qubits: int, entangling: bool = True) -> float:
    """M / M_c_bound, M_c_bound = 4^n - 1 (DLA dimension upper bound),
    SPEC2_ADDENDUM.md §12.1.

    PLAYBOOK2.md Stage 10: with entangling gates removed the DLA is
    su(2)^{tensor n} with dimension 3n regardless of depth, so the corrected
    capacity bound for the ablation arms is 3n (=12 at n=4)."""
    bound = (4**n_qubits - 1) if entangling else 3 * n_qubits
    return n_params / bound


def should_log(epoch: int, dense_until: int, log_interval_dense: int, log_interval_sparse: int) -> bool:
    """Log every epoch until dense_until, then every log_interval_sparse epochs (SPEC §7)."""
    if epoch <= dense_until:
        return epoch % log_interval_dense == 0
    return epoch % log_interval_sparse == 0


def _build_criterion(loss_type: str, label_smoothing: float, num_classes: int):
    """Builds the loss callable for train_vqc's loss_type/label_smoothing config
    (SPEC3_ADDENDUM.md §20 arm 3, §23 label_smoothing field). Returns a
    callable(logits, y) -> scalar tensor, matching nn.Module.__call__'s signature
    so it can substitute directly for nn.CrossEntropyLoss()."""
    if loss_type == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    elif loss_type == "mse":
        mse_base = nn.MSELoss()

        def criterion(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            y_onehot = torch.nn.functional.one_hot(y, num_classes=num_classes).to(logits.dtype)
            return mse_base(logits, y_onehot)

        return criterion
    else:
        raise ValueError(f"unknown loss: {loss_type}")


def get_git_sha() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _nc_record_fields(features: np.ndarray, labels: np.ndarray, num_classes: int) -> dict:
    """Classical NC1/NC2 mapped onto the shared SPEC §7 schema fields."""
    nc1 = classical_nc1(features, labels)
    nc2 = classical_nc2(features, labels)
    gram_offdiag_cos = [
        float(nc2["cos"][i, j])
        for i in range(num_classes)
        for j in range(i + 1, num_classes)
    ]
    return {
        "qnc1_trace": nc1,
        "qnc1_hs": None,
        "qnc1_rel": None,
        "equinorm_cv": nc2["equinorm_cv"],
        "equiangle_dev": nc2["equiangle_dev"],
        "equiangle_std": nc2["equiangle_std"],
        "gram_offdiag_cos": gram_offdiag_cos,
        "class_mean_norms": [float(v) for v in nc2["class_mean_norms"]],
        "entropy_mean": None,
        "entropy_class": None,
        "entropy_classmean": None,
        "beta": None,
        "qnc1_fisher": None,
        "qnc1_ratio": None,
        "overlap_offdiag": None,
        "purity_class": None,
        "purity_mean": None,
        "qfi_rank": None,
        "within_class_scatter_trace": within_class_scatter_trace(features, labels),
    }


def train(config: dict, seed_override: int | None = None) -> str:
    """Runs one classical-control training run per SPEC §5, logs per SPEC §7.

    Returns the run_id.
    """
    seed = seed_override if seed_override is not None else config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_cfg = config["train"]
    device_cfg = train_cfg.get("device", "auto")
    use_cuda = device_cfg == "cuda" or (device_cfg == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")

    data_cfg = config["data"]
    dataset_name = data_cfg.get("dataset", "mnist")
    if dataset_name == "mnist":
        x_train_raw, y_train, x_test_raw, y_test = load_mnist_subset(
            classes=data_cfg["classes"],
            samples_per_class=data_cfg["samples_per_class"],
            test_fraction=data_cfg["test_fraction"],
            seed=seed,
        )
    elif dataset_name == "fashion_mnist":
        x_train_raw, y_train, x_test_raw, y_test = load_fashion_mnist_subset(
            classes=data_cfg["classes"],
            samples_per_class=data_cfg["samples_per_class"],
            test_fraction=data_cfg["test_fraction"],
            seed=seed,
        )
    else:
        raise ValueError(f"unknown dataset: {dataset_name}")

    pca_components = data_cfg.get("pca_components")
    if pca_components is not None:
        # Stage 19 Arm V2(a): capacity-matched classical arm on the SAME
        # PCA-4 features the VQC sees, via the shared pca_encode pipeline.
        x_train, x_test = pca_encode(x_train_raw, x_test_raw, n_qubits=pca_components, seed=seed)
        encoding = "pca"
        input_dim = pca_components
    else:
        x_train, x_test = x_train_raw, x_test_raw
        encoding = None
        input_dim = x_train.shape[1]
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    num_classes = len(data_cfg["classes"])
    model_cfg = config.get("model", {})
    hidden_dims = tuple(model_cfg.get("hidden_dims", (512, 512, 64)))
    model = ClassicalMLP(num_classes=num_classes, input_dim=input_dim, hidden_dims=hidden_dims).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train_cfg["lr"],
        momentum=train_cfg["momentum"],
        weight_decay=train_cfg["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    lr_schedule = train_cfg.get("lr_schedule", "constant")
    scheduler = None
    if lr_schedule == "step_decay_thirds":
        milestones = [train_cfg["epochs"] // 3, 2 * train_cfg["epochs"] // 3]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif lr_schedule == "multistep":
        milestones = train_cfg["lr_milestones"]
        gamma = train_cfg.get("lr_gamma", 0.1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    log_cfg = config["logging"]
    compute_on = config.get("metrics", {}).get("compute_on", ["train"])

    run_id = generate_run_id(config["experiment"], seed)
    logger = JSONLLogger(run_id)
    git_sha = get_git_sha()

    total_epochs_config = train_cfg["epochs"]
    tpt_multiple = train_cfg.get("tpt_multiple", 5)
    batch_size = train_cfg["batch_size"]

    n_train = x_train.shape[0]
    e0: int | None = None
    target_epochs = total_epochs_config
    epoch = 0
    start_time = time.time()

    # Epoch 0 (pre-training) baseline. Captures the true early-training NC peak
    # at random init, before the first gradient step collapses it: on a tiny
    # subset (~12 batches/epoch) one epoch already separates the classes, so the
    # "early-training peak" (PLAYBOOK Stage 1 / SPEC §8) lives at epoch 0.
    model.eval()
    with torch.no_grad():
        init_train_logits, init_train_features = model(x_train, return_features=True)
        init_train_acc = (init_train_logits.argmax(dim=1) == y_train).float().mean().item()
        init_train_loss = criterion(init_train_logits, y_train).item()
    base_record0 = {
        "run_id": run_id,
        "epoch": 0,
        "tpt_reached": False,
        "e0_epoch": None,
        "seed": seed,
        "git_sha": git_sha,
        "wall_time_s": 0.0,
        "n_params": n_params,
        "m_over_mc": None,
        "loss_type": "ce",
        "dataset": dataset_name,
        "encoding": encoding,
    }
    if "train" in compute_on:
        logger.log({
            **base_record0,
            "split": "train",
            "loss": init_train_loss,
            "acc": init_train_acc,
            **_nc_record_fields(
                init_train_features.detach().cpu().numpy().astype(np.float64),
                y_train.detach().cpu().numpy(),
                num_classes,
            ),
        })
    if "test" in compute_on:
        with torch.no_grad():
            init_test_logits, init_test_features = model(x_test, return_features=True)
            init_test_acc = (init_test_logits.argmax(dim=1) == y_test).float().mean().item()
            init_test_loss = criterion(init_test_logits, y_test).item()
        logger.log({
            **base_record0,
            "split": "test",
            "loss": init_test_loss,
            "acc": init_test_acc,
            **_nc_record_fields(
                init_test_features.detach().cpu().numpy().astype(np.float64),
                y_test.detach().cpu().numpy(),
                num_classes,
            ),
        })

    while epoch < target_epochs:
        epoch += 1
        model.train()
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = x_train[idx], y_train[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            train_logits, train_features = model(x_train, return_features=True)
            train_acc = (train_logits.argmax(dim=1) == y_train).float().mean().item()
            train_loss = criterion(train_logits, y_train).item()

        if e0 is None and is_e0(train_acc):
            e0 = epoch
        target_epochs = compute_target_epochs(e0, total_epochs_config, tpt_multiple)

        is_final = epoch == target_epochs
        if should_log(epoch, log_cfg["dense_until"], log_cfg["log_interval_dense"], log_cfg["log_interval_sparse"]) or is_final:
            base_record = {
                "run_id": run_id,
                "epoch": epoch,
                "tpt_reached": e0 is not None,
                "e0_epoch": e0,
                "seed": seed,
                "git_sha": git_sha,
                "wall_time_s": time.time() - start_time,
                "n_params": n_params,
                "m_over_mc": None,
                "loss_type": "ce",
                "dataset": dataset_name,
                "encoding": encoding,
            }

            if "train" in compute_on:
                features_np = train_features.detach().cpu().numpy().astype(np.float64)
                labels_np = y_train.detach().cpu().numpy()
                record = {
                    **base_record,
                    "split": "train",
                    "loss": train_loss,
                    "acc": train_acc,
                    **_nc_record_fields(features_np, labels_np, num_classes),
                }
                logger.log(record)

            if "test" in compute_on:
                with torch.no_grad():
                    test_logits, test_features = model(x_test, return_features=True)
                    test_acc = (test_logits.argmax(dim=1) == y_test).float().mean().item()
                    test_loss = criterion(test_logits, y_test).item()
                test_features_np = test_features.detach().cpu().numpy().astype(np.float64)
                test_labels_np = y_test.detach().cpu().numpy()
                record = {
                    **base_record,
                    "split": "test",
                    "loss": test_loss,
                    "acc": test_acc,
                    **_nc_record_fields(test_features_np, test_labels_np, num_classes),
                }
                logger.log(record)

            if log_cfg.get("checkpoint", True):
                ckpt_dir = logger.run_dir / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:04d}.pt")

    return run_id


def shuffle_labels(y: torch.Tensor, seed: int) -> torch.Tensor:
    """Seeded label permutation for the Stage 5 shuffled-labels control
    (SPEC/PLAYBOOK Stage 5): reassigns each sample's label from a random
    permutation of the same label multiset, decorrelating x from y while
    leaving class balance and x unchanged."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    return y[torch.from_numpy(perm)]


def statevectors_to_rhos(psi: np.ndarray) -> np.ndarray:
    """rho_i = |psi_i><psi_i| for a batch of pure statevectors, SPEC §1.

    psi: complex array (N, d). Returns rho: complex array (N, d, d).
    """
    return np.einsum("ni,nj->nij", psi, psi.conj())


def assert_physical_rhos(rhos: np.ndarray, tol: float = 1e-8) -> None:
    """Runtime assertions per SPEC §4.1/§9: Hermitian, Tr=1, PSD (eigenvalues >= -1e-10)."""
    hermitian_gap = float(np.max(np.abs(rhos - rhos.conj().transpose(0, 2, 1))))
    assert hermitian_gap < tol, f"rho not Hermitian: max gap {hermitian_gap}"
    traces = np.trace(rhos, axis1=1, axis2=2)
    trace_gap = float(np.max(np.abs(traces - 1.0)))
    assert trace_gap < tol, f"rho trace != 1: max deviation {trace_gap}"
    eigvals = np.linalg.eigvalsh(rhos)
    assert np.all(eigvals >= -1e-10), f"rho not PSD: min eigenvalue {eigvals.min()}"


def _qnc_record_fields(rhos: np.ndarray, labels: np.ndarray, n_qubits: int) -> dict:
    """Full SPEC §4 quantum NC + entanglement entropy fields, mapped onto the
    shared SPEC §7 schema. Uses only the already-tested metrics.py functions."""
    classes = np.unique(labels)
    means, global_mean = class_means(rhos, labels)
    qnc1 = qnc1_metrics(rhos, labels, means, global_mean)
    qnc2 = qnc2_metrics(means, global_mean)
    num_classes = len(classes)
    gram_offdiag_cos = [
        float(qnc2["cos"][i, j]) for i in range(num_classes) for j in range(i + 1, num_classes)
    ]

    num_keep = math.ceil(n_qubits / 2)
    entropy_per_sample = np.array(
        [von_neumann_entropy(reduced_density(rho, n_qubits, num_keep)) for rho in rhos]
    )
    entropy_mean = float(entropy_per_sample.mean())
    entropy_class = [float(entropy_per_sample[labels == c].mean()) for c in classes]
    entropy_classmean = [
        float(von_neumann_entropy(reduced_density(means[idx], n_qubits, num_keep)))
        for idx in range(num_classes)
    ]

    return {
        "qnc1_trace": qnc1["qnc1_trace"],
        "qnc1_hs": qnc1["qnc1_hs"],
        "qnc1_rel": qnc1["qnc1_rel"],
        "equinorm_cv": qnc2["equinorm_cv"],
        "equiangle_dev": qnc2["equiangle_dev"],
        "equiangle_std": qnc2["equiangle_std"],
        "gram_offdiag_cos": gram_offdiag_cos,
        "class_mean_norms": [float(v) for v in qnc2["class_mean_norms"]],
        "entropy_mean": entropy_mean,
        "entropy_class": entropy_class,
        "entropy_classmean": entropy_classmean,
        "qnc1_fisher": qnc1_fisher(qnc1["qnc1_hs"], qnc2["class_mean_norms"]),
        "qnc1_ratio": qnc1_ratio(qnc1["qnc1_trace"], means),
        "overlap_offdiag": overlap_offdiag(means),
        "purity_class": [purity(means[idx]) for idx in range(num_classes)],
        "purity_mean": float(np.mean([purity(means[idx]) for idx in range(num_classes)])),
        "within_class_scatter_trace": None,
    }


def _lr_for_epoch(epoch: int, total_epochs: int, base_lr: float, schedule: str) -> float:
    """Optional lr schedules (Playbook Stage 2 research knobs), applied once per
    epoch. `cosine_last_half` decays from `base_lr` to a 0.5% floor over the
    second half of `total_epochs`; the floor (not zero) keeps gradient flow
    alive during any TPT-rule epoch extension past `total_epochs`."""
    if schedule == "constant":
        return base_lr
    if schedule == "cosine_last_half":
        half = total_epochs / 2
        if epoch <= half:
            return base_lr
        progress = min(1.0, (epoch - half) / max(total_epochs - half, 1e-9))
        floor = 0.005
        return base_lr * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))
    raise ValueError(f"unknown lr_schedule: {schedule}")


def train_vqc(config: dict, seed_override: int | None = None) -> str:
    """Runs one VQC training run per SPEC §2/§3, logs per SPEC §7.

    Returns the run_id.
    """
    seed = seed_override if seed_override is not None else config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_cfg = config["train"]
    # PennyLane default.qubit (this version) initializes its internal statevector
    # on CPU regardless of input tensor placement, so a cuda model/input causes a
    # cuda/cpu device-mismatch crash inside the simulator. Force CPU for the VQC
    # path; n_qubits here (4-8) is trivial CPU work anyway, unlike the classical
    # MLP path which does benefit from GPU.
    device = torch.device("cpu")

    data_cfg = config["data"]
    dataset_name = data_cfg["dataset"]
    if dataset_name == "two_moons":
        x_train_raw, y_train, x_test_raw, y_test = load_two_moons(
            n_samples=data_cfg["n_samples"],
            noise=data_cfg["noise"],
            test_fraction=data_cfg["test_fraction"],
            seed=seed,
        )
    elif dataset_name == "mnist":
        x_train_raw, y_train, x_test_raw, y_test = load_mnist_subset(
            classes=data_cfg["classes"],
            samples_per_class=data_cfg["samples_per_class"],
            test_fraction=data_cfg["test_fraction"],
            seed=seed,
        )
    elif dataset_name == "fashion_mnist":
        x_train_raw, y_train, x_test_raw, y_test = load_fashion_mnist_subset(
            classes=data_cfg["classes"],
            samples_per_class=data_cfg["samples_per_class"],
            test_fraction=data_cfg["test_fraction"],
            seed=seed,
        )
    elif dataset_name == "blobs3":
        x_train_raw, y_train, x_test_raw, y_test = load_blobs3(
            samples_per_class=data_cfg["samples_per_class"],
            blob_dist=data_cfg["blob_dist"],
            blob_std=data_cfg["blob_std"],
            test_fraction=data_cfg["test_fraction"],
            seed=seed,
            ambient_dim=data_cfg.get("ambient_dim", 4),
        )
    elif dataset_name == "bars_stripes_4x4":
        x_train_raw, y_train, x_test_raw, y_test = load_bars_stripes_4x4(
            samples_per_class=data_cfg["samples_per_class"],
            noise_std=data_cfg.get("noise_std", 0.5),
            test_fraction=data_cfg["test_fraction"],
            seed=seed,
        )
    elif dataset_name == "linearly_separable_4d":
        x_train_raw, y_train, x_test_raw, y_test = load_linearly_separable_4d(
            samples_per_class=data_cfg["samples_per_class"],
            margin=data_cfg.get("margin", 0.1),
            test_fraction=data_cfg["test_fraction"],
            seed=seed,
        )
    else:
        raise ValueError(f"unknown dataset: {dataset_name}")

    if data_cfg.get("shuffle_labels", False):
        # Stage 5 falsification control: decorrelate x from y (train split
        # only -- test labels stay real so test acc remains a meaningful
        # chance-level check) so a converging model alone can't trivially
        # reproduce the NC signal.
        y_train = shuffle_labels(y_train, seed)

    if dataset_name in ("mnist", "fashion_mnist"):
        num_classes = len(data_cfg["classes"])
    elif dataset_name == "blobs3":
        num_classes = 3
    else:  # two_moons, bars_stripes_4x4, linearly_separable_4d: all binary
        num_classes = 2

    model_cfg = config["model"]
    n_qubits = model_cfg["n_qubits"]
    pca_components = data_cfg.get("pca_components", n_qubits)
    encoding = model_cfg.get("encoding", "angle")

    if encoding == "amplitude":
        if dataset_name in ("mnist", "fashion_mnist"):
            # Stage 20 Arm W4: raw pixels are 784-d, far above the 2^n_qubits
            # amplitude space, so PCA down to `pca_components` first (NOT the
            # angle-rescaling pca_encode -- AmplitudeEmbedding normalizes the
            # raw vector itself, SPEC3_ADDENDUM.md section 24).
            x_train, x_test = pca_encode_amplitude(
                x_train_raw, x_test_raw, n_components=pca_components, seed=seed
            )
            x_train, x_test = x_train.to(torch.float64), x_test.to(torch.float64)
        else:
            # AmplitudeEmbedding normalizes internally; skip PCA/angle-scaling
            # so the model sees the raw feature vector (e.g. the 16
            # bars-stripes pixels mapping exactly onto the 4-qubit amplitude
            # space).
            x_train = x_train_raw.to(torch.float64)
            x_test = x_test_raw.to(torch.float64)
    else:
        x_train, x_test = pca_encode(x_train_raw, x_test_raw, n_qubits=pca_components, seed=seed)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    model = VQC(
        n_qubits=n_qubits,
        num_classes=num_classes,
        n_layers=model_cfg["layers"],
        reupload=model_cfg.get("reupload", False),
        entangling=model_cfg.get("entangling", True),
        beta_0=model_cfg.get("beta0", 5.0),
        beta_learnable=model_cfg.get("beta_learnable", True),
        ansatz=model_cfg.get("ansatz", "strongly_entangling"),
        shots=train_cfg.get("shots"),
        encoding=encoding,
        readout_family=model_cfg.get("readout_family", "R0"),
        measurement_layers=model_cfg.get("measurement_layers", 2),
    ).to(device)
    n_params = model.n_params
    m_over_mc_value = m_over_mc(n_params, n_qubits, entangling=model_cfg.get("entangling", True))

    optimizer_name = train_cfg.get("optimizer", "adam")
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    elif optimizer_name == "sgd":
        # Stage 18 arm B1 (PREDICTIONS.md "Stage 18 preregistration"):
        # tests whether the collapse phenomenology is Adam-specific.
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=train_cfg["lr"],
            momentum=train_cfg.get("momentum", 0.0),
        )
    elif optimizer_name == "lbfgs":
        # Full-batch quasi-Newton: appropriate here since the VQC has only
        # O(n_layers * n_qubits * 3) parameters (dozens, not thousands).
        # Strong-Wolfe line search lets a single optimizer.step(closure) call
        # perform up to lbfgs_max_iter internal L-BFGS iterations per epoch.
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=train_cfg["lr"],
            max_iter=train_cfg.get("lbfgs_max_iter", 20),
            history_size=train_cfg.get("lbfgs_history_size", 10),
            line_search_fn="strong_wolfe",
        )
    else:
        raise ValueError(f"unknown optimizer: {optimizer_name}")

    loss_type = train_cfg.get("loss", "ce")
    # SPEC2_ADDENDUM.md §14 (mse path): L = mean_i ||beta*z_i - y_i||^2 (+
    # weight decay, added separately below as in the CE path). model(x)
    # already returns beta*z (VQC.forward); y is one-hot per SPEC2 §14.
    label_smoothing = train_cfg.get("label_smoothing", 0.0)
    criterion = _build_criterion(loss_type, label_smoothing, num_classes)

    weight_decay = train_cfg.get("weight_decay", 0.0)
    beta_clamp = model_cfg.get("beta_max", 50.0)
    lr_schedule = train_cfg.get("lr_schedule", "constant")

    log_cfg = config["logging"]
    compute_on = config.get("metrics", {}).get("compute_on", ["train"])

    run_id = generate_run_id(config["experiment"], seed)
    logger = JSONLLogger(run_id)
    git_sha = get_git_sha()

    total_epochs_config = train_cfg["epochs"]
    tpt_multiple = train_cfg.get("tpt_multiple", 5)
    batch_size = train_cfg["batch_size"]

    n_train = x_train.shape[0]
    e0: int | None = None
    target_epochs = total_epochs_config
    epoch = 0
    start_time = time.time()

    def _compute_qfi_rank() -> int:
        """QFI rank at the current weights, per SPEC2 §12.2. Restricted to
        epoch 0 / final only (caller's responsibility) since this is O(batch
        * dim * n_params) autograd calls."""
        x_qfi = x_train[: min(16, x_train.shape[0])]
        psi_fn = lambda: model.state(x_qfi)
        g = qfi_matrix(psi_fn, model.weights)
        return qfi_rank(g)

    def _log_split(
        epoch_num: int, split: str, x: torch.Tensor, y: torch.Tensor, qfi_rank_value: int | None = None
    ) -> None:
        with torch.no_grad():
            logits = model(x)
            acc = (logits.argmax(dim=1) == y).float().mean().item()
            loss_val = criterion(logits, y).item()
            psi = model.state(x).detach().cpu().numpy()
        rhos = statevectors_to_rhos(psi)
        assert_physical_rhos(rhos)
        logger.log({
            "run_id": run_id,
            "epoch": epoch_num,
            "split": split,
            "loss": loss_val,
            "acc": acc,
            "beta": model.beta.item() if model.beta is not None else None,
            "tpt_reached": e0 is not None,
            "e0_epoch": e0,
            "seed": seed,
            "git_sha": git_sha,
            "wall_time_s": time.time() - start_time if epoch_num > 0 else 0.0,
            "n_params": n_params,
            "m_over_mc": m_over_mc_value,
            "qfi_rank": qfi_rank_value,
            "loss_type": loss_type,
            "dataset": dataset_name,
            "encoding": encoding,
            "readout_family": model.readout_family,
            "m_params_measurement": model.m_params_measurement,
            "label_smoothing": label_smoothing,
            **_qnc_record_fields(rhos, y.detach().cpu().numpy(), n_qubits),
        })

    # Epoch 0 (pre-training) baseline, mirrors the classical hook.
    model.eval()
    qfi_rank_epoch0 = _compute_qfi_rank()
    if "train" in compute_on:
        _log_split(0, "train", x_train, y_train, qfi_rank_value=qfi_rank_epoch0)
    if "test" in compute_on:
        _log_split(0, "test", x_test, y_test, qfi_rank_value=qfi_rank_epoch0)

    while epoch < target_epochs:
        epoch += 1
        for g in optimizer.param_groups:
            g["lr"] = _lr_for_epoch(epoch, total_epochs_config, train_cfg["lr"], lr_schedule)
        model.train()
        if optimizer_name == "lbfgs":
            def closure():
                optimizer.zero_grad()
                logits = model(x_train)
                loss = criterion(logits, y_train) + weight_decay * torch.sum(model.weights**2)
                loss.backward()
                return loss

            optimizer.step(closure)
            if model.beta is not None:
                with torch.no_grad():
                    model.beta.clamp_(-beta_clamp, beta_clamp)
        else:
            perm = torch.randperm(n_train, device=device)
            for start in range(0, n_train, batch_size):
                idx = perm[start : start + batch_size]
                xb, yb = x_train[idx], y_train[idx]
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb) + weight_decay * torch.sum(model.weights**2)
                loss.backward()
                optimizer.step()  # type: ignore[call-arg]
                if model.beta is not None:
                    with torch.no_grad():
                        model.beta.clamp_(-beta_clamp, beta_clamp)

        model.eval()
        with torch.no_grad():
            train_logits = model(x_train)
            train_acc = (train_logits.argmax(dim=1) == y_train).float().mean().item()

        if e0 is None and is_e0(train_acc):
            e0 = epoch
        target_epochs = compute_target_epochs(e0, total_epochs_config, tpt_multiple)

        is_final = epoch == target_epochs
        if should_log(epoch, log_cfg["dense_until"], log_cfg["log_interval_dense"], log_cfg["log_interval_sparse"]) or is_final:
            qfi_rank_value = _compute_qfi_rank() if is_final else None
            if "train" in compute_on:
                _log_split(epoch, "train", x_train, y_train, qfi_rank_value=qfi_rank_value)
            if "test" in compute_on:
                _log_split(epoch, "test", x_test, y_test, qfi_rank_value=qfi_rank_value)

            if log_cfg.get("checkpoint", True):
                ckpt_dir = logger.run_dir / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:04d}.pt")

    return run_id
