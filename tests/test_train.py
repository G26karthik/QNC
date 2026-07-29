"""Unit tests for the pure-logic helpers in src/qnc/train.py: E0 detection,
TPT target-epoch computation, and the log-interval schedule (SPEC §3, §7).
Per CLAUDE.md invariant 1, mandatory test-first applies only to metrics.py;
the stochastic training loop itself is not unit-tested here."""

import numpy as np
import pytest
import torch

from qnc.train import (
    _build_criterion,
    _lr_for_epoch,
    assert_physical_rhos,
    compute_target_epochs,
    is_e0,
    m_over_mc,
    should_log,
    shuffle_labels,
    statevectors_to_rhos,
)


class TestIsE0:
    def test_exact_one_is_e0(self):
        assert is_e0(1.0) is True

    def test_below_one_is_not_e0(self):
        assert is_e0(0.999) is False

    def test_within_tolerance_counts_as_e0(self):
        assert is_e0(1.0 - 1e-10) is True


class TestComputeTargetEpochs:
    def test_no_e0_yet_uses_total_epochs(self):
        assert compute_target_epochs(e0=None, total_epochs=350, tpt_multiple=5) == 350

    def test_e0_early_extends_past_total_epochs(self):
        # E0 at epoch 100, tpt_multiple=5 -> 500 > total_epochs=350
        assert compute_target_epochs(e0=100, total_epochs=350, tpt_multiple=5) == 500

    def test_e0_late_keeps_total_epochs(self):
        # E0 at epoch 340, 5*340=1700 > 350 -> still extends
        assert compute_target_epochs(e0=340, total_epochs=350, tpt_multiple=5) == 1700

    def test_e0_at_one_uses_total_epochs_if_larger(self):
        assert compute_target_epochs(e0=1, total_epochs=350, tpt_multiple=5) == 350


class TestShouldLog:
    def test_logs_every_epoch_within_dense_window(self):
        for epoch in range(1, 51):
            assert should_log(epoch, dense_until=50, log_interval_dense=1, log_interval_sparse=5)

    def test_sparse_interval_after_dense_window(self):
        assert should_log(55, dense_until=50, log_interval_dense=1, log_interval_sparse=5) is True
        assert should_log(53, dense_until=50, log_interval_dense=1, log_interval_sparse=5) is False

    def test_boundary_epoch_at_dense_until(self):
        assert should_log(50, dense_until=50, log_interval_dense=1, log_interval_sparse=5) is True
        assert should_log(51, dense_until=50, log_interval_dense=1, log_interval_sparse=5) is False


class TestStatevectorsToRhos:
    def test_single_qubit_zero_state(self):
        psi = np.array([[1.0 + 0j, 0.0 + 0j]])
        rhos = statevectors_to_rhos(psi)
        np.testing.assert_allclose(rhos[0], [[1, 0], [0, 0]], atol=1e-12)

    def test_batch_trace_one(self):
        psi = np.array([[1, 0], [0, 1], [1 / 2**0.5, 1 / 2**0.5]], dtype=complex)
        rhos = statevectors_to_rhos(psi)
        traces = np.trace(rhos, axis1=1, axis2=2)
        np.testing.assert_allclose(traces, [1, 1, 1], atol=1e-12)


class TestAssertPhysicalRhos:
    def test_valid_pure_states_pass(self):
        psi = np.array([[1, 0], [1 / 2**0.5, 1 / 2**0.5]], dtype=complex)
        rhos = statevectors_to_rhos(psi)
        assert_physical_rhos(rhos)  # must not raise

    def test_non_hermitian_raises(self):
        bad = np.zeros((1, 2, 2), dtype=complex)
        bad[0] = [[1, 0.5j], [0, 0]]
        with pytest.raises(AssertionError, match="Hermitian"):
            assert_physical_rhos(bad)

    def test_wrong_trace_raises(self):
        bad = np.zeros((1, 2, 2), dtype=complex)
        bad[0] = [[2, 0], [0, 0]]
        with pytest.raises(AssertionError, match="trace"):
            assert_physical_rhos(bad)


class TestLrForEpoch:
    def test_constant_schedule_is_flat(self):
        assert _lr_for_epoch(1, 100, 0.05, "constant") == 0.05
        assert _lr_for_epoch(100, 100, 0.05, "constant") == 0.05

    def test_cosine_last_half_flat_before_midpoint(self):
        assert _lr_for_epoch(40, 100, 0.05, "cosine_last_half") == 0.05

    def test_cosine_last_half_decays_to_floor_at_end(self):
        lr_end = _lr_for_epoch(100, 100, 0.05, "cosine_last_half")
        assert lr_end == pytest.approx(0.05 * 0.005, abs=1e-6)

    def test_cosine_last_half_holds_floor_past_extension(self):
        lr_far = _lr_for_epoch(500, 100, 0.05, "cosine_last_half")
        assert lr_far == pytest.approx(0.05 * 0.005, abs=1e-6)

    def test_unknown_schedule_raises(self):
        with pytest.raises(ValueError):
            _lr_for_epoch(1, 100, 0.05, "bogus")


class TestMOverMc:
    """SPEC2_ADDENDUM.md §12.1: M_c_bound = 4^n - 1; m_over_mc = M / M_c_bound.
    PLAYBOOK2.md Stage 10: the non-entangling ansatz has DLA dim 3n regardless
    of depth, so its corrected bound is 3n (=12 at n=4)."""

    def test_l2_on_4_qubits(self):
        assert m_over_mc(24, 4) == pytest.approx(24 / 255, abs=1e-9)

    def test_full_bound_is_one(self):
        assert m_over_mc(255, 4) == pytest.approx(1.0, abs=1e-9)

    def test_non_entangling_bound_is_3n(self):
        assert m_over_mc(24, 4, entangling=False) == pytest.approx(24 / 12, abs=1e-9)

    def test_entangling_default_matches_explicit_true(self):
        assert m_over_mc(96, 4) == m_over_mc(96, 4, entangling=True)


class TestShuffleLabels:
    """PLAYBOOK Stage 5 shuffled-labels control: labels must be permuted
    (not resampled), seeded/deterministic, and applied independently of x."""

    def test_deterministic_for_same_seed(self):
        y = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
        out1 = shuffle_labels(y, seed=7)
        out2 = shuffle_labels(y, seed=7)
        assert torch.equal(out1, out2)

    def test_different_seeds_can_differ(self):
        y = torch.tensor(list(range(2)) * 10)
        out_a = shuffle_labels(y, seed=1)
        out_b = shuffle_labels(y, seed=2)
        assert not torch.equal(out_a, out_b)

    def test_preserves_class_counts(self):
        y = torch.tensor([0, 0, 0, 1, 1, 2])
        out = shuffle_labels(y, seed=3)
        assert sorted(out.tolist()) == sorted(y.tolist())

    def test_actually_permutes_not_identity(self):
        y = torch.tensor(list(range(2)) * 20)  # 40 samples, balanced 0/1
        out = shuffle_labels(y, seed=0)
        assert not torch.equal(out, y)


class TestBuildCriterion:
    """SPEC3_ADDENDUM.md §20 arm 3 / §23: label_smoothing is a ce-loss-only
    config knob threaded into nn.CrossEntropyLoss."""

    def test_ce_zero_smoothing_matches_plain_cross_entropy(self):
        criterion = _build_criterion("ce", 0.0, num_classes=3)
        logits = torch.tensor([[2.0, 0.5, 0.1], [0.1, 0.2, 3.0]])
        y = torch.tensor([0, 2])
        expected = torch.nn.CrossEntropyLoss()(logits, y)
        assert criterion(logits, y).item() == pytest.approx(expected.item(), abs=1e-9)

    def test_ce_label_smoothing_changes_loss_value(self):
        logits = torch.tensor([[2.0, 0.5, 0.1], [0.1, 0.2, 3.0]])
        y = torch.tensor([0, 2])
        plain = _build_criterion("ce", 0.0, num_classes=3)(logits, y)
        smoothed = _build_criterion("ce", 0.1, num_classes=3)(logits, y)
        assert smoothed.item() != pytest.approx(plain.item(), abs=1e-9)

    def test_mse_ignores_label_smoothing(self):
        logits = torch.tensor([[1.0, 0.0, 0.0]])
        y = torch.tensor([0])
        criterion = _build_criterion("mse", 0.1, num_classes=3)
        assert criterion(logits, y).item() == pytest.approx(0.0, abs=1e-9)

    def test_unknown_loss_raises(self):
        with pytest.raises(ValueError):
            _build_criterion("bogus", 0.0, num_classes=3)
