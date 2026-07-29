"""Model definitions.

Stage 1: implements SPEC.md §5 classical MLP control model.
Stage 2: implements SPEC.md §2 VQC (angle encoding, StronglyEntanglingLayers,
Pauli-Z readout, learnable inverse-temperature beta).
"""

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn


class ClassicalMLP(nn.Module):
    """input_dim -> hidden_dims... -> C, ReLU. Implements SPEC.md §5.

    Defaults (784 -> 512 -> 512 -> 64 -> C) match the Stage 1 classical
    control exactly; `hidden_dims` is overridden for Stage 19 Arm V2's
    capacity-matched PCA-4 classical arm (4 -> 64 -> 16 -> C).

    `forward(x, return_features=True)` also returns the penultimate
    (last hidden_dims entry) layer activations used for the classical NC
    epoch hook.
    """

    def __init__(self, num_classes: int, input_dim: int = 784, hidden_dims: tuple[int, ...] = (512, 512, 64)):
        super().__init__()
        blocks = []
        prev_dim = input_dim
        for dim in hidden_dims:
            blocks.append(nn.Sequential(nn.Linear(prev_dim, dim), nn.ReLU()))
            prev_dim = dim
        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = x
        for block in self.blocks:
            features = block(features)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


class VQC(nn.Module):
    """Variational quantum classifier per SPEC.md §2.

    Angle encoding (RY per qubit, optional re-upload before each layer),
    `qml.StronglyEntanglingLayers` variational blocks, Pauli-Z readout on
    the first `num_classes` qubits, learnable inverse-temperature `beta`.
    The `entangling=False` ablation removes entangling gates by swapping
    the layer's `imprimitive` from CNOT to Identity -- same rotations,
    gates literally removed, per SPEC.md §2.

    `forward(x)` returns logits `(batch, num_classes)`.
    `state(x)` returns the pre-measurement statevector `(batch, 2**n_qubits)`
    (complex128) for the epoch-hook NC metrics extraction (SPEC.md §4);
    it does not depend on `beta`, which only scales the post-measurement
    logits.

    `shots` (PLAYBOOK.md Stage 5 falsification check) only affects the
    expval circuit used by `forward()` (training signal): when set, that
    circuit runs on its own finite-shots device with `diff_method=
    "parameter-shift"` (backprop requires an exact, shots=None device).
    `state()` always uses a separate, permanently exact device, so SPEC §4
    metrics are unaffected by shots -- "shots affect TRAINING only".

    `readout_family` (SPEC3_ADDENDUM.md section 19, Stage 13 expressivity
    sweep): "R0" (default) is the original fixed-Z_c + learnable-beta readout
    above. "R1" measures all 3 single-qubit Paulis on the first `num_classes`
    qubits (3*C features) through a trainable linear head (no beta). "R2"
    measures all weight-1 Paulis on all qubits plus all weight-2 ZZ pairs
    (3n + n(n-1)/2 features) through a trainable linear head (no beta). "R3"
    keeps the fixed Z_c + beta readout but inserts a trainable "measurement
    unitary" (`measurement_layers` StronglyEntanglingLayers, params in
    `measurement_weights`) between the main circuit and the measurement.
    `state()` always returns the state BEFORE any measurement-family-specific
    transform (main circuit output only) so full-space witnesses stay
    directly comparable across arms.
    """

    def __init__(
        self,
        n_qubits: int,
        num_classes: int,
        n_layers: int,
        reupload: bool = False,
        entangling: bool = True,
        beta_0: float = 5.0,
        beta_learnable: bool = True,
        ansatz: str = "strongly_entangling",
        shots: int | None = None,
        encoding: str = "angle",
        readout_family: str = "R0",
        measurement_layers: int = 2,
    ):
        super().__init__()
        if num_classes > n_qubits:
            raise ValueError("SPEC §2 requires C <= n (readout needs one qubit per class)")
        if encoding not in ("angle", "amplitude"):
            raise ValueError(f"unknown encoding: {encoding}")
        if readout_family not in ("R0", "R1", "R2", "R3"):
            raise ValueError(f"unknown readout_family: {readout_family}")
        self.n_qubits = n_qubits
        self.num_classes = num_classes
        self.n_layers = n_layers
        self.reupload = reupload
        self.ansatz = ansatz
        self.encoding = encoding
        self.readout_family = readout_family
        self.imprimitive = qml.CNOT if entangling else qml.Identity

        # float64 throughout: PennyLane's torch-interface state computation loses
        # precision silently under float32 (state dtype still reports complex128,
        # but trace/Hermiticity only hold to ~1e-7), which fails the 1e-8 physics
        # assertions required by SPEC §9.
        if ansatz == "strongly_entangling":
            weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        elif ansatz == "hardware_efficient":
            # Genuinely different functional form from StronglyEntanglingLayers:
            # RY+RZ per qubit per layer (2 params/qubit vs its 3), entangled by a
            # linear (open-chain) CNOT ladder instead of a ring with variable range.
            weight_shape = (n_layers, n_qubits, 2)
        else:
            raise ValueError(f"unknown ansatz: {ansatz}")
        self.weights = nn.Parameter(torch.rand(*weight_shape, dtype=torch.float64) * 2 * np.pi)

        # beta only applies to the fixed-Z_c readouts (R0/R3); R1/R2 route
        # through a trainable linear head instead (SPEC3 §19), which learns
        # its own scale.
        if readout_family in ("R0", "R3"):
            self.beta = nn.Parameter(torch.tensor(float(beta_0), dtype=torch.float64), requires_grad=beta_learnable)
        else:
            self.beta = None

        if readout_family == "R1":
            self.n_features = 3 * num_classes
            self.head = nn.Linear(self.n_features, num_classes, dtype=torch.float64)
        elif readout_family == "R2":
            self.n_features = 3 * n_qubits + n_qubits * (n_qubits - 1) // 2
            self.head = nn.Linear(self.n_features, num_classes, dtype=torch.float64)
        else:
            self.n_features = None
            self.head = None

        if readout_family == "R3":
            self.measurement_layers = measurement_layers
            meas_shape = qml.StronglyEntanglingLayers.shape(n_layers=measurement_layers, n_wires=n_qubits)
            self.measurement_weights = nn.Parameter(torch.rand(*meas_shape, dtype=torch.float64) * 2 * np.pi)
        else:
            self.measurement_layers = 0
            self.measurement_weights = None

        # Exact device: always shots=None, backprop. Used for state() (SPEC §4
        # metrics) unconditionally, and for forward() when shots is not set.
        self._device = qml.device("default.qubit", wires=n_qubits)
        self._state_qnode = qml.QNode(self._state_circuit, self._device, interface="torch", diff_method="backprop")

        if shots is None:
            self._expval_qnode = qml.QNode(self._expval_circuit, self._device, interface="torch", diff_method="backprop")
        else:
            # Backprop requires shots=None, so a finite-shots QNode needs a
            # stochastic-compatible differentiation rule instead. `set_shots`
            # (not device-level `shots=`) is the non-deprecated PennyLane API.
            base_qnode = qml.QNode(self._expval_circuit, self._device, interface="torch", diff_method="parameter-shift")
            self._expval_qnode = qml.set_shots(base_qnode, shots=shots)

    @property
    def n_params(self) -> int:
        """Main-circuit variational parameter count M (excludes beta and any
        readout-family head/measurement params), SPEC2 §12.1."""
        return self.weights.numel()

    @property
    def m_params_measurement(self) -> int:
        """Extra readout-family trainable params beyond the main circuit:
        linear-head weights+bias for R1/R2, measurement-unitary weights for
        R3, 0 for R0. SPEC3_ADDENDUM.md section 19/23."""
        if self.readout_family == "R3":
            return self.measurement_weights.numel()
        if self.head is not None:
            return sum(p.numel() for p in self.head.parameters())
        return 0

    def _encode(self, x: torch.Tensor) -> None:
        if self.encoding == "amplitude":
            qml.AmplitudeEmbedding(x, wires=range(self.n_qubits), normalize=True)
            return
        encode_dim = x.shape[-1]
        for j in range(encode_dim):
            qml.RY(x[..., j], wires=j)

    def _hardware_efficient_layer(self, weights_layer: torch.Tensor) -> None:
        for q in range(self.n_qubits):
            qml.RY(weights_layer[..., q, 0], wires=q)
            qml.RZ(weights_layer[..., q, 1], wires=q)
        if self.imprimitive is not qml.Identity:
            for q in range(self.n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])

    def _variational_block(self, weights: torch.Tensor) -> None:
        if self.ansatz == "strongly_entangling":
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits), imprimitive=self.imprimitive)
        else:
            for l in range(weights.shape[0]):
                self._hardware_efficient_layer(weights[l])

    def _circuit_body(self, x: torch.Tensor, weights: torch.Tensor) -> None:
        if self.reupload:
            for l in range(self.n_layers):
                self._encode(x)
                self._variational_block(weights[l : l + 1])
        else:
            self._encode(x)
            self._variational_block(weights)

    def _expval_circuit(self, x: torch.Tensor, weights: torch.Tensor, measurement_weights: torch.Tensor | None = None):
        self._circuit_body(x, weights)
        if self.readout_family == "R0":
            return [qml.expval(qml.PauliZ(c)) for c in range(self.num_classes)]
        if self.readout_family == "R1":
            obs = []
            for c in range(self.num_classes):
                obs.append(qml.expval(qml.PauliX(c)))
                obs.append(qml.expval(qml.PauliY(c)))
                obs.append(qml.expval(qml.PauliZ(c)))
            return obs
        if self.readout_family == "R2":
            obs = []
            for q in range(self.n_qubits):
                obs.append(qml.expval(qml.PauliX(q)))
                obs.append(qml.expval(qml.PauliY(q)))
                obs.append(qml.expval(qml.PauliZ(q)))
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    obs.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)))
            return obs
        # R3: trainable measurement unitary block, then fixed Z_c readout.
        qml.StronglyEntanglingLayers(measurement_weights, wires=range(self.n_qubits))
        return [qml.expval(qml.PauliZ(c)) for c in range(self.num_classes)]

    def _state_circuit(self, x: torch.Tensor, weights: torch.Tensor):
        # Always the main-circuit-only state (no measurement-family
        # transform applied), so full-space witnesses stay comparable
        # across R0-R3 arms.
        self._circuit_body(x, weights)
        return qml.state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x64 = x.to(torch.float64)
        if self.readout_family == "R3":
            raw = torch.stack(self._expval_qnode(x64, self.weights, self.measurement_weights), dim=-1)
            return self.beta * raw
        raw = torch.stack(self._expval_qnode(x64, self.weights), dim=-1)
        if self.readout_family == "R0":
            return self.beta * raw
        return self.head(raw)

    def state(self, x: torch.Tensor) -> torch.Tensor:
        return self._state_qnode(x.to(torch.float64), self.weights)
