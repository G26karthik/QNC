"""Stage 18 arm B4: translation gate.

`qnc.stage18_noisy_analysis.noisy_z` reuses the trained VQC's own
`_circuit_body` inside a `default.mixed` + `qml.insert(..., position="all")`
noisy QNode (PREDICTIONS.md "Stage 18 preregistration", arm B4). Hard gate
per that preregistration: with p=0 (no depolarizing channel), the noisy
pipeline must reproduce the exact noiseless raw-Z_c expectations to numerical
precision before any noisy-inference probe or job is treated as
scientifically meaningful, mirroring Stage 16's translation-gate discipline
(`tests/test_stage16_translation_gate.py`).
"""

import numpy as np
import torch

from qnc.stage16_hardware_analysis import _load_reference_data_and_model
from qnc.stage18_noisy_analysis import noisy_z


def test_zero_noise_matches_exact_statevector():
    x_all, _y_all, weights_np = _load_reference_data_and_model()

    from qnc.models import VQC

    model = VQC(
        n_qubits=4, num_classes=3, n_layers=8,
        reupload=True, entangling=True, encoding="angle", readout_family="R0",
    )
    model.weights.data = torch.from_numpy(weights_np)

    # Raw (pre-beta) exact expectations from the model's own exact QNode --
    # ground truth, same convention as qnc.hardware.statevector_z.
    exact = np.stack(
        [torch.stack(model._expval_qnode(torch.from_numpy(x_all[i]), model.weights), dim=-1).detach().numpy()
         for i in range(x_all.shape[0])]
    )

    noisy_p0 = noisy_z(x_all, weights_np, model, p=0.0)

    # 1e-6, matching Stage 16's own translation-gate tolerance
    # (tests/test_stage16_translation_gate.py) -- default.mixed's
    # density-matrix representation accumulates more floating-point error
    # than the statevector simulator over an 8-layer reuploading circuit,
    # even with an identity (p=0) channel.
    assert np.max(np.abs(noisy_p0 - exact)) < 1e-6


def test_nonzero_noise_contracts_magnitude():
    """Sanity check (not a scored prediction): depolarizing noise should
    not increase |z| beyond its noiseless magnitude, on average, for a
    handful of samples -- a cheap smoke test that the channel is actually
    being applied, before spending probe time on the full 180-sample set."""
    x_all, _y_all, weights_np = _load_reference_data_and_model()

    from qnc.models import VQC

    model = VQC(
        n_qubits=4, num_classes=3, n_layers=8,
        reupload=True, entangling=True, encoding="angle", readout_family="R0",
    )
    model.weights.data = torch.from_numpy(weights_np)

    x_small = x_all[:5]
    z_exact = noisy_z(x_small, weights_np, model, p=0.0)
    z_noisy = noisy_z(x_small, weights_np, model, p=0.05)

    assert np.mean(np.abs(z_noisy)) <= np.mean(np.abs(z_exact)) + 1e-9
    assert not np.allclose(z_noisy, z_exact)
