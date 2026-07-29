"""Data loading and preprocessing.

Stage 1: implements SPEC.md §5 data pipeline (MNIST subset loader,
deterministic class/sample selection by seed).
Stage 2: extends with PCA + angle-encoding scaling per SPEC.md §2.
"""

import gzip
import itertools
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FASHION_MNIST_BASE_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
IDX_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
MNIST_FILES = IDX_FILES  # backward-compat alias
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache" / "mnist"
DEFAULT_FASHION_CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache" / "fashion_mnist"


def _download_and_cache(filename: str, cache_dir: Path, base_url: str = MNIST_BASE_URL) -> Path:
    """Downloads a canonical idx-ubyte.gz file (MNIST or Fashion-MNIST,
    same format, different mirror) if not already cached locally."""
    path = cache_dir / filename
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        url = base_url + filename
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    return path


def _parse_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        buf = f.read(rows * cols * num)
        return np.frombuffer(buf, dtype=np.uint8).reshape(num, rows * cols)


def _parse_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _, num = struct.unpack(">II", f.read(8))
        buf = f.read(num)
        return np.frombuffer(buf, dtype=np.uint8)


def _load_full_idx_dataset(cache_dir: Path, base_url: str) -> tuple[np.ndarray, np.ndarray]:
    train_x = _parse_idx_images(_download_and_cache(IDX_FILES["train_images"], cache_dir, base_url))
    train_y = _parse_idx_labels(_download_and_cache(IDX_FILES["train_labels"], cache_dir, base_url))
    test_x = _parse_idx_images(_download_and_cache(IDX_FILES["test_images"], cache_dir, base_url))
    test_y = _parse_idx_labels(_download_and_cache(IDX_FILES["test_labels"], cache_dir, base_url))

    x_all = np.concatenate([train_x, test_x], axis=0)
    y_all = np.concatenate([train_y, test_y], axis=0).astype(np.int64)
    return x_all, y_all


def _subset_from_full(
    x_all: np.ndarray,
    y_all: np.ndarray,
    classes: list[int],
    samples_per_class: int,
    test_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared deterministic class/sample selection + split, per SPEC.md §5.

    Labels are remapped to 0..C-1 in sorted class order. Pixels scaled to
    [0, 1]. Selection and train/test split are fully determined by `seed`.
    """
    rng = np.random.default_rng(seed)
    classes_sorted = sorted(classes)

    x_parts, y_parts = [], []
    for new_label, c in enumerate(classes_sorted):
        idx = np.where(y_all == c)[0]
        rng.shuffle(idx)
        idx = idx[:samples_per_class]
        x_parts.append(x_all[idx])
        y_parts.append(np.full(len(idx), new_label, dtype=np.int64))

    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    perm = rng.permutation(len(x))
    x, y = x[perm], y[perm]

    n_test = int(len(x) * test_fraction)
    x_test, y_test = x[:n_test], y[:n_test]
    x_train, y_train = x[n_test:], y[n_test:]

    return (
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
        torch.from_numpy(x_test),
        torch.from_numpy(y_test),
    )


def load_mnist_subset(
    classes: list[int],
    samples_per_class: int,
    test_fraction: float,
    seed: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic MNIST subset per SPEC.md §5. See `_subset_from_full`."""
    x_all_uint8, y_all = _load_full_idx_dataset(cache_dir, MNIST_BASE_URL)
    x_all = x_all_uint8.astype(np.float32) / 255.0
    return _subset_from_full(x_all, y_all, classes, samples_per_class, test_fraction, seed)


def load_fashion_mnist_subset(
    classes: list[int],
    samples_per_class: int,
    test_fraction: float,
    seed: int,
    cache_dir: Path = DEFAULT_FASHION_CACHE_DIR,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic Fashion-MNIST subset, same recipe as `load_mnist_subset`
    (Stage 19 Arm V1). Original Fashion-MNIST label 0/1/2 = T-shirt/top,
    Trouser, Pullover."""
    x_all_uint8, y_all = _load_full_idx_dataset(cache_dir, FASHION_MNIST_BASE_URL)
    x_all = x_all_uint8.astype(np.float32) / 255.0
    return _subset_from_full(x_all, y_all, classes, samples_per_class, test_fraction, seed)


def load_two_moons(
    n_samples: int,
    noise: float,
    test_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic two-moons synthetic dataset (Stage 2 smoke test).

    `sklearn.datasets.make_moons` seeded by `seed`; split is a seeded
    permutation, mirroring `load_mnist_subset`.

    Returns (X_train, y_train, X_test, y_test) as torch float32 / int64
    tensors, X raw (unscaled) 2-d coordinates.
    """
    x, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    y = y.astype(np.int64)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x))
    x, y = x[perm], y[perm]

    n_test = int(len(x) * test_fraction)
    x_test, y_test = x[:n_test], y[:n_test]
    x_train, y_train = x[n_test:], y[n_test:]

    return (
        torch.from_numpy(x_train.astype(np.float32)),
        torch.from_numpy(y_train),
        torch.from_numpy(x_test.astype(np.float32)),
        torch.from_numpy(y_test),
    )


def _seeded_split(
    x: np.ndarray, y: np.ndarray, test_fraction: float, rng: np.random.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    perm = rng.permutation(len(x))
    x, y = x[perm], y[perm]
    n_test = int(len(x) * test_fraction)
    x_test, y_test = x[:n_test], y[:n_test]
    x_train, y_train = x[n_test:], y[n_test:]
    return (
        torch.from_numpy(x_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.int64)),
        torch.from_numpy(x_test.astype(np.float32)),
        torch.from_numpy(y_test.astype(np.int64)),
    )


def load_blobs3(
    samples_per_class: int,
    blob_dist: float,
    blob_std: float,
    test_fraction: float,
    seed: int,
    ambient_dim: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """3 isotropic Gaussian blobs in R^`ambient_dim`, per SPEC2_ADDENDUM.md
    §13.1 (default R^4); lifted to R^6 for SPEC3_ADDENDUM.md §21's n=6
    replication.

    Centers fixed at `blob_dist * e_c` for `c = 0, 1, 2` (standard basis
    vectors in R^`ambient_dim`), so pairwise center distance is a controlled
    constant `blob_dist * sqrt(2)`. Deterministic per seed; balanced classes.
    """
    rng = np.random.default_rng(seed)
    centers = blob_dist * np.eye(3, ambient_dim)

    x_parts, y_parts = [], []
    for label, center in enumerate(centers):
        x_parts.append(rng.normal(loc=center, scale=blob_std, size=(samples_per_class, ambient_dim)))
        y_parts.append(np.full(samples_per_class, label))
    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    return _seeded_split(x, y, test_fraction, rng)


_BARS_STRIPES_PATTERNS = [
    p for p in itertools.product([0, 1], repeat=4) if len(set(p)) > 1
]


def load_bars_stripes_4x4(
    samples_per_class: int,
    noise_std: float,
    test_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """4x4 binary bars-and-stripes with Gaussian pixel noise, C=2, per
    SPEC2_ADDENDUM.md §13.2 (following Bowles, Ahmed, Schuld 2024).

    Class 0 (bars): each row is uniformly 0 or 1. Class 1 (stripes): each
    column is uniformly 0 or 1. The 2 degenerate all-0/all-1 images (shared
    by both definitions) are excluded from the pattern pool to keep labels
    unambiguous. Flattened to a 16-d vector with Gaussian noise added.
    """
    rng = np.random.default_rng(seed)
    patterns = _BARS_STRIPES_PATTERNS

    x_parts, y_parts = [], []
    for label in (0, 1):
        idxs = rng.integers(0, len(patterns), size=samples_per_class)
        images = []
        for idx in idxs:
            bits = np.array(patterns[idx])
            image = np.tile(bits.reshape(4, 1), (1, 4)) if label == 0 else np.tile(bits.reshape(1, 4), (4, 1))
            images.append(image.flatten())
        images = np.stack(images).astype(np.float64)
        images += rng.normal(scale=noise_std, size=images.shape)
        x_parts.append(images)
        y_parts.append(np.full(samples_per_class, label))
    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    return _seeded_split(x, y, test_fraction, rng)


def load_linearly_separable_4d(
    samples_per_class: int,
    margin: float,
    test_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bowles et al.'s "fruit fly": uniform hypercube samples in [-1, 1]^4
    split by the hyperplane orthogonal to (1,1,1,1), margin band excluded,
    C=2, per SPEC2_ADDENDUM.md §13.3.
    """
    rng = np.random.default_rng(seed)

    def _sample_class(label: int, count: int) -> np.ndarray:
        collected = []
        while len(collected) < count:
            batch = rng.uniform(-1.0, 1.0, size=(count * 2, 4))
            s = batch.sum(axis=1)
            mask = s > margin if label == 1 else s < -margin
            collected.extend(batch[mask])
        return np.array(collected[:count])

    x_parts = [_sample_class(label, samples_per_class) for label in (0, 1)]
    y_parts = [np.full(samples_per_class, label) for label in (0, 1)]
    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    return _seeded_split(x, y, test_fraction, rng)


def pca_encode(
    x_train_raw: torch.Tensor,
    x_test_raw: torch.Tensor,
    n_qubits: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PCA -> standardize -> scale to [-pi, pi], per SPEC.md §2.

    Fits PCA (`n_components = min(n_qubits, raw_dim)`) and the standardizer
    on the train split only; train min/max (post-standardization) define
    the [-pi, pi] scaling applied to both splits. `svd_solver="full"` makes
    PCA deterministic (no RNG dependence beyond the input ordering, which is
    itself seeded upstream).

    If `raw_dim < n_qubits` (e.g. two-moons), only the first `raw_dim`
    qubits receive angle-encoded features; the caller's VQC encodes exactly
    `x.shape[-1]` qubits, leaving the rest idle except for variational and
    entangling gates.

    Returns (X_train, X_test) scaled to [-pi, pi], shape (N, min(n_qubits, raw_dim)).
    """
    x_train_np = x_train_raw.detach().cpu().numpy().astype(np.float64)
    x_test_np = x_test_raw.detach().cpu().numpy().astype(np.float64)
    raw_dim = x_train_np.shape[1]
    n_components = min(n_qubits, raw_dim)

    pca = PCA(n_components=n_components, svd_solver="full", random_state=seed)
    z_train = pca.fit_transform(x_train_np)
    z_test = pca.transform(x_test_np)

    scaler = StandardScaler()
    z_train = scaler.fit_transform(z_train)
    z_test = scaler.transform(z_test)

    train_min = z_train.min(axis=0)
    train_max = z_train.max(axis=0)
    span = train_max - train_min

    def _scale(z: np.ndarray) -> np.ndarray:
        return (z - train_min) / span * 2 * np.pi - np.pi

    return (
        torch.from_numpy(_scale(z_train).astype(np.float32)),
        torch.from_numpy(_scale(z_test).astype(np.float32)),
    )


def pca_encode_amplitude(
    x_train_raw: torch.Tensor,
    x_test_raw: torch.Tensor,
    n_components: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PCA -> standardize, WITHOUT the [-pi, pi] angle rescale, for amplitude
    encoding (Stage 20 Arm W4, SPEC3_ADDENDUM.md section 24).

    `qml.AmplitudeEmbedding(normalize=True)` L2-normalizes the raw feature
    vector itself, so unlike `pca_encode` (built for RY rotation angles) this
    must NOT remap components to [-pi, pi] -- that per-component affine
    remap would distort the relative norms AmplitudeEmbedding depends on.
    Fits PCA (`n_components = min(n_components, raw_dim)`) and the
    standardizer on the train split only, same determinism guarantees as
    pca_encode (`svd_solver="full"`, `random_state=seed`).

    Returns (X_train, X_test), shape (N, min(n_components, raw_dim)), to be
    passed directly to a model configured with `encoding: amplitude`.
    """
    x_train_np = x_train_raw.detach().cpu().numpy().astype(np.float64)
    x_test_np = x_test_raw.detach().cpu().numpy().astype(np.float64)
    raw_dim = x_train_np.shape[1]
    n_comp = min(n_components, raw_dim)

    pca = PCA(n_components=n_comp, svd_solver="full", random_state=seed)
    z_train = pca.fit_transform(x_train_np)
    z_test = pca.transform(x_test_np)

    scaler = StandardScaler()
    z_train = scaler.fit_transform(z_train)
    z_test = scaler.transform(z_test)

    return (
        torch.from_numpy(z_train.astype(np.float32)),
        torch.from_numpy(z_test.astype(np.float32)),
    )
