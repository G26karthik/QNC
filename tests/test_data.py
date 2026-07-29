"""Unit tests for SPEC2_ADDENDUM.md §13 dataset generators in src/qnc/data.py.
Shape, class balance, and seeded determinism -- no network dependency."""

import numpy as np
import torch

from qnc.data import (
    _subset_from_full,
    load_blobs3,
    load_bars_stripes_4x4,
    load_linearly_separable_4d,
    pca_encode_amplitude,
)


def _combined_labels(y_train: torch.Tensor, y_test: torch.Tensor) -> np.ndarray:
    return np.concatenate([y_train.numpy(), y_test.numpy()])


class TestSubsetFromFull:
    """Stage 19: MNIST and Fashion-MNIST subsetting share this pure function
    (`_load_full_idx_dataset` differs only by download URL); tested here
    without any network dependency."""

    def _synthetic_full(self, n_per_class: int = 50, n_classes: int = 5, dim: int = 8):
        rng = np.random.default_rng(0)
        x = rng.integers(0, 256, size=(n_per_class * n_classes, dim)).astype(np.uint8)
        y = np.repeat(np.arange(n_classes), n_per_class)
        return x, y

    def test_class_balance_and_relabeling(self):
        x_all, y_all = self._synthetic_full()
        x_train, y_train, x_test, y_test = _subset_from_full(
            x_all, y_all, classes=[3, 1, 0], samples_per_class=20, test_fraction=0.2, seed=0
        )
        labels = _combined_labels(y_train, y_test)
        assert set(labels.tolist()) == {0, 1, 2}
        assert len(labels) == 60
        for label in (0, 1, 2):
            assert (labels == label).sum() == 20

    def test_deterministic_for_same_seed(self):
        x_all, y_all = self._synthetic_full()
        r1 = _subset_from_full(x_all, y_all, classes=[0, 1], samples_per_class=10, test_fraction=0.2, seed=7)
        r2 = _subset_from_full(x_all, y_all, classes=[0, 1], samples_per_class=10, test_fraction=0.2, seed=7)
        for a, b in zip(r1, r2):
            assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        x_all, y_all = self._synthetic_full()
        x_train_a, *_ = _subset_from_full(x_all, y_all, classes=[0, 1], samples_per_class=10, test_fraction=0.2, seed=0)
        x_train_b, *_ = _subset_from_full(x_all, y_all, classes=[0, 1], samples_per_class=10, test_fraction=0.2, seed=1)
        assert not torch.equal(x_train_a, x_train_b)


class TestLoadBlobs3:
    def test_shape_is_r4(self):
        x_train, y_train, x_test, y_test = load_blobs3(
            samples_per_class=20, blob_dist=3.0, blob_std=0.8, test_fraction=0.2, seed=0
        )
        assert x_train.shape[1] == 4
        assert x_test.shape[1] == 4
        assert x_train.dtype == torch.float32
        assert y_train.dtype == torch.int64

    def test_class_balance(self):
        x_train, y_train, x_test, y_test = load_blobs3(
            samples_per_class=20, blob_dist=3.0, blob_std=0.8, test_fraction=0.2, seed=0
        )
        labels = _combined_labels(y_train, y_test)
        assert len(labels) == 60
        counts = np.bincount(labels)
        np.testing.assert_array_equal(counts, [20, 20, 20])

    def test_deterministic_for_same_seed(self):
        run1 = load_blobs3(samples_per_class=10, blob_dist=3.0, blob_std=0.8, test_fraction=0.2, seed=5)
        run2 = load_blobs3(samples_per_class=10, blob_dist=3.0, blob_std=0.8, test_fraction=0.2, seed=5)
        for a, b in zip(run1, run2):
            torch.testing.assert_close(a, b)

    def test_different_seeds_differ(self):
        run1 = load_blobs3(samples_per_class=10, blob_dist=3.0, blob_std=0.8, test_fraction=0.2, seed=1)
        run2 = load_blobs3(samples_per_class=10, blob_dist=3.0, blob_std=0.8, test_fraction=0.2, seed=2)
        assert not torch.equal(run1[0], run2[0])

    def test_ambient_dim_6_lifts_shape(self):
        x_train, y_train, x_test, y_test = load_blobs3(
            samples_per_class=20, blob_dist=3.0, blob_std=0.8, test_fraction=0.2, seed=0, ambient_dim=6
        )
        assert x_train.shape[1] == 6
        assert x_test.shape[1] == 6
        labels = _combined_labels(y_train, y_test)
        counts = np.bincount(labels)
        np.testing.assert_array_equal(counts, [20, 20, 20])

    def test_ambient_dim_6_center_distance(self):
        # Large sample count to concentrate the empirical class means near the
        # true generating centers (blob_dist * e_c in R^6, pairwise
        # blob_dist*sqrt(2) apart), decoupled from the small-n sweep configs.
        x_train, y_train, x_test, y_test = load_blobs3(
            samples_per_class=2000, blob_dist=3.0, blob_std=0.8, test_fraction=0.2, seed=0, ambient_dim=6
        )
        labels = _combined_labels(y_train, y_test)
        x_all = np.concatenate([x_train.numpy(), x_test.numpy()], axis=0)
        class_means = np.stack([x_all[labels == c].mean(axis=0) for c in range(3)])
        for c1 in range(3):
            for c2 in range(c1 + 1, 3):
                dist = np.linalg.norm(class_means[c1] - class_means[c2])
                np.testing.assert_allclose(dist, 3.0 * np.sqrt(2), rtol=0.05)


class TestLoadBarsStripes4x4:
    def test_shape_is_16d(self):
        x_train, y_train, x_test, y_test = load_bars_stripes_4x4(
            samples_per_class=20, noise_std=0.5, test_fraction=0.2, seed=0
        )
        assert x_train.shape[1] == 16
        assert x_test.shape[1] == 16

    def test_class_balance(self):
        x_train, y_train, x_test, y_test = load_bars_stripes_4x4(
            samples_per_class=20, noise_std=0.5, test_fraction=0.2, seed=0
        )
        labels = _combined_labels(y_train, y_test)
        assert len(labels) == 40
        counts = np.bincount(labels)
        np.testing.assert_array_equal(counts, [20, 20])

    def test_deterministic_for_same_seed(self):
        run1 = load_bars_stripes_4x4(samples_per_class=10, noise_std=0.5, test_fraction=0.2, seed=7)
        run2 = load_bars_stripes_4x4(samples_per_class=10, noise_std=0.5, test_fraction=0.2, seed=7)
        for a, b in zip(run1, run2):
            torch.testing.assert_close(a, b)

    def test_different_seeds_differ(self):
        run1 = load_bars_stripes_4x4(samples_per_class=10, noise_std=0.5, test_fraction=0.2, seed=1)
        run2 = load_bars_stripes_4x4(samples_per_class=10, noise_std=0.5, test_fraction=0.2, seed=2)
        assert not torch.equal(run1[0], run2[0])


class TestLoadLinearlySeparable4d:
    def test_shape_is_r4(self):
        x_train, y_train, x_test, y_test = load_linearly_separable_4d(
            samples_per_class=20, margin=0.1, test_fraction=0.2, seed=0
        )
        assert x_train.shape[1] == 4
        assert x_test.shape[1] == 4

    def test_class_balance(self):
        x_train, y_train, x_test, y_test = load_linearly_separable_4d(
            samples_per_class=20, margin=0.1, test_fraction=0.2, seed=0
        )
        labels = _combined_labels(y_train, y_test)
        assert len(labels) == 40
        counts = np.bincount(labels)
        np.testing.assert_array_equal(counts, [20, 20])

    def test_labels_respect_margin(self):
        x_train, y_train, x_test, y_test = load_linearly_separable_4d(
            samples_per_class=30, margin=0.1, test_fraction=0.0, seed=0
        )
        x_np = x_train.numpy()
        y_np = y_train.numpy()
        s = x_np.sum(axis=1)
        assert np.all(s[y_np == 1] > 0.1)
        assert np.all(s[y_np == 0] < -0.1)

    def test_deterministic_for_same_seed(self):
        run1 = load_linearly_separable_4d(samples_per_class=10, margin=0.1, test_fraction=0.2, seed=3)
        run2 = load_linearly_separable_4d(samples_per_class=10, margin=0.1, test_fraction=0.2, seed=3)
        for a, b in zip(run1, run2):
            torch.testing.assert_close(a, b)

    def test_different_seeds_differ(self):
        run1 = load_linearly_separable_4d(samples_per_class=10, margin=0.1, test_fraction=0.2, seed=1)
        run2 = load_linearly_separable_4d(samples_per_class=10, margin=0.1, test_fraction=0.2, seed=2)
        assert not torch.equal(run1[0], run2[0])


class TestPcaEncodeAmplitude:
    """Stage 20 Arm W4: PCA path for amplitude encoding must NOT rescale to
    [-pi, pi] (SPEC3_ADDENDUM.md section 24) -- unlike pca_encode, which is
    built for RY rotation angles."""

    def _raw(self, n=60, dim=30, seed=0):
        rng = np.random.default_rng(seed)
        return torch.from_numpy(rng.normal(size=(n, dim)).astype(np.float32))

    def test_shape_matches_requested_n_components(self):
        x_train_raw, x_test_raw = self._raw(60, 30), self._raw(20, 30, seed=1)
        x_train, x_test = pca_encode_amplitude(x_train_raw, x_test_raw, n_components=16, seed=0)
        assert x_train.shape == (60, 16)
        assert x_test.shape == (20, 16)

    def test_output_not_rescaled_to_pi_range(self):
        # Standardized PCA output has no fixed [-pi, pi] bound; with 60
        # standard-normal-derived samples it will almost certainly exceed pi
        # in at least one component, unlike pca_encode's guaranteed [-pi,pi].
        x_train_raw, x_test_raw = self._raw(60, 30), self._raw(20, 30, seed=1)
        x_train, _ = pca_encode_amplitude(x_train_raw, x_test_raw, n_components=16, seed=0)
        assert x_train.abs().max().item() > np.pi

    def test_n_components_capped_at_raw_dim(self):
        x_train_raw, x_test_raw = self._raw(60, 4), self._raw(20, 4, seed=1)
        x_train, x_test = pca_encode_amplitude(x_train_raw, x_test_raw, n_components=16, seed=0)
        assert x_train.shape == (60, 4)
        assert x_test.shape == (20, 4)

    def test_deterministic_for_same_seed(self):
        x_train_raw, x_test_raw = self._raw(60, 30), self._raw(20, 30, seed=1)
        run1 = pca_encode_amplitude(x_train_raw, x_test_raw, n_components=16, seed=3)
        run2 = pca_encode_amplitude(x_train_raw, x_test_raw, n_components=16, seed=3)
        for a, b in zip(run1, run2):
            torch.testing.assert_close(a, b)
