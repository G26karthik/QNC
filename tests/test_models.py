"""Unit tests for src/qnc/models.py VQC shots plumbing (PLAYBOOK Stage 5).

SPEC.md notes shots affect TRAINING only: the expval/logit circuit may be
evaluated with finite shots, but the state circuit used for SPEC §4 NC
metrics must remain exact (shots=None) regardless of the training config.
These tests pin that separation so a future change can't silently make
shots leak into the metrics path.
"""

import numpy as np
import torch

from qnc.models import ClassicalMLP, VQC


class TestClassicalMLP:
    """Stage 19 Arm V2: ClassicalMLP's hidden_dims/input_dim parametrization
    must default to the Stage 1 control's 784-512-512-64-C architecture
    exactly, and support the PCA-4 comparative arm's 4-64-16-C."""

    def test_default_matches_stage1_control(self):
        model = ClassicalMLP(num_classes=3)
        x = torch.randn(5, 784)
        logits, features = model(x, return_features=True)
        assert logits.shape == (5, 3)
        assert features.shape == (5, 64)

    def test_pca4_arch(self):
        model = ClassicalMLP(num_classes=3, input_dim=4, hidden_dims=(64, 16))
        x = torch.randn(5, 4)
        logits, features = model(x, return_features=True)
        assert logits.shape == (5, 3)
        assert features.shape == (5, 16)

    def test_forward_without_return_features(self):
        model = ClassicalMLP(num_classes=2, input_dim=4, hidden_dims=(64, 16))
        x = torch.randn(3, 4)
        logits = model(x)
        assert logits.shape == (3, 2)


def _tiny_model(shots: int | None, seed: int = 0) -> VQC:
    torch.manual_seed(seed)
    return VQC(
        n_qubits=2,
        num_classes=2,
        n_layers=1,
        reupload=False,
        entangling=True,
        beta_0=1.0,
        beta_learnable=True,
        shots=shots,
    )


class TestShotsNone:
    def test_forward_is_deterministic_across_calls(self):
        model = _tiny_model(shots=None)
        x = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
        out1 = model(x).detach().numpy()
        out2 = model(x).detach().numpy()
        np.testing.assert_allclose(out1, out2, atol=1e-12)

    def test_state_is_deterministic(self):
        model = _tiny_model(shots=None)
        x = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
        psi1 = model.state(x).detach().numpy()
        psi2 = model.state(x).detach().numpy()
        np.testing.assert_allclose(psi1, psi2, atol=1e-12)


class TestShotsSet:
    def test_forward_is_stochastic_across_calls(self):
        model = _tiny_model(shots=50)
        x = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
        outs = [model(x).detach().numpy().copy() for _ in range(5)]
        # With finite shots, repeated evaluations at fixed weights must not
        # all be bit-identical -- that would mean shots aren't wired in.
        assert not all(np.allclose(outs[0], o, atol=1e-9) for o in outs[1:])

    def test_state_stays_exact_even_when_shots_set(self):
        model = _tiny_model(shots=50)
        x = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
        psi1 = model.state(x).detach().numpy()
        psi2 = model.state(x).detach().numpy()
        # state() must always hit the exact (shots=None) device.
        np.testing.assert_allclose(psi1, psi2, atol=1e-12)
        norm = np.sum(np.abs(psi1) ** 2, axis=-1)
        np.testing.assert_allclose(norm, 1.0, atol=1e-8)

    def test_gradients_flow_with_shots_set(self):
        model = _tiny_model(shots=100)
        x = torch.tensor([[0.3, -0.2], [0.1, 0.4]], dtype=torch.float64)
        y = torch.tensor([0, 1])
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        assert model.weights.grad is not None
        assert torch.all(torch.isfinite(model.weights.grad))


class TestNParams:
    """SPEC2_ADDENDUM.md §12.1: M = 12L for SEL on 4 qubits (3 rotation
    params x 4 qubits per layer), excluding the learnable beta scalar."""

    def test_n_params_strongly_entangling(self):
        model = VQC(n_qubits=4, num_classes=2, n_layers=2, ansatz="strongly_entangling")
        assert model.n_params == 24

    def test_n_params_excludes_beta(self):
        model = VQC(n_qubits=4, num_classes=2, n_layers=2, ansatz="strongly_entangling")
        assert model.n_params == model.weights.numel()


class TestAmplitudeEncoding:
    """SPEC2_ADDENDUM.md §13.2: amplitude encoding option gated by config
    `encoding`, via qml.AmplitudeEmbedding(normalize=True)."""

    def test_forward_pass_produces_finite_logits(self):
        torch.manual_seed(0)
        model = VQC(n_qubits=4, num_classes=2, n_layers=1, encoding="amplitude")
        x = torch.rand(3, 16, dtype=torch.float64)  # unnormalized; normalize=True handles it
        logits = model(x)
        assert logits.shape == (3, 2)
        assert torch.all(torch.isfinite(logits))

    def test_state_is_normalized(self):
        torch.manual_seed(0)
        model = VQC(n_qubits=4, num_classes=2, n_layers=1, encoding="amplitude")
        x = torch.rand(2, 16, dtype=torch.float64)
        psi = model.state(x).detach().numpy()
        norm = np.sum(np.abs(psi) ** 2, axis=-1)
        np.testing.assert_allclose(norm, 1.0, atol=1e-8)

    def test_angle_is_still_default(self):
        torch.manual_seed(0)
        model = VQC(n_qubits=2, num_classes=2, n_layers=1)
        assert model.encoding == "angle"


class TestReadoutFamilies:
    """SPEC3_ADDENDUM.md section 19 (Stage 13 expressivity sweep): R0-R3
    readout families. n_qubits=4, num_classes=3 throughout to match the
    Stage 13 matched-M table (n_features = 3C=9 for R1, 3n+n(n-1)/2=18 for R2)."""

    def test_r0_uses_beta_no_head(self):
        model = VQC(n_qubits=4, num_classes=3, n_layers=1, readout_family="R0")
        assert model.beta is not None
        assert model.head is None
        assert model.m_params_measurement == 0

    def test_r1_forward_shape_and_head_param_count(self):
        torch.manual_seed(0)
        model = VQC(n_qubits=4, num_classes=3, n_layers=1, readout_family="R1")
        assert model.beta is None
        assert model.n_features == 9
        assert model.m_params_measurement == 9 * 3 + 3  # 30
        x = torch.rand(2, 4, dtype=torch.float64)
        logits = model(x)
        assert logits.shape == (2, 3)
        assert torch.all(torch.isfinite(logits))

    def test_r2_forward_shape_and_head_param_count(self):
        torch.manual_seed(0)
        model = VQC(n_qubits=4, num_classes=3, n_layers=1, readout_family="R2")
        assert model.beta is None
        assert model.n_features == 18  # 3*4 + 4*3/2
        assert model.m_params_measurement == 18 * 3 + 3  # 57
        x = torch.rand(2, 4, dtype=torch.float64)
        logits = model(x)
        assert logits.shape == (2, 3)
        assert torch.all(torch.isfinite(logits))

    def test_r3_forward_shape_and_measurement_param_count(self):
        torch.manual_seed(0)
        model = VQC(n_qubits=4, num_classes=3, n_layers=1, readout_family="R3", measurement_layers=2)
        assert model.beta is not None
        assert model.head is None
        assert model.m_params_measurement == 2 * 4 * 3  # 24
        x = torch.rand(2, 4, dtype=torch.float64)
        logits = model(x)
        assert logits.shape == (2, 3)
        assert torch.all(torch.isfinite(logits))

    def test_state_unaffected_by_readout_family(self):
        # Same seed -> identical main-circuit weights -> identical state(),
        # regardless of readout_family (measurement transform excluded).
        x = torch.rand(2, 4, dtype=torch.float64)
        torch.manual_seed(1)
        model_r0 = VQC(n_qubits=4, num_classes=3, n_layers=1, readout_family="R0")
        torch.manual_seed(1)
        model_r3 = VQC(n_qubits=4, num_classes=3, n_layers=1, readout_family="R3")
        np.testing.assert_allclose(
            model_r0.state(x).detach().numpy(), model_r3.state(x).detach().numpy(), atol=1e-12
        )

    def test_gradients_flow_for_r1_r2_r3(self):
        x = torch.rand(3, 4, dtype=torch.float64)
        y = torch.tensor([0, 1, 2])
        for family in ("R1", "R2", "R3"):
            torch.manual_seed(0)
            model = VQC(n_qubits=4, num_classes=3, n_layers=1, readout_family=family)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            assert model.weights.grad is not None
            assert torch.all(torch.isfinite(model.weights.grad))

    def test_unknown_readout_family_raises(self):
        try:
            VQC(n_qubits=4, num_classes=3, n_layers=1, readout_family="R4")
            assert False, "expected ValueError"
        except ValueError:
            pass
