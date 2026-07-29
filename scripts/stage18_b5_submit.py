"""Stage 18 arm B5: hardware second-backend replication (ibm_kingston).

Same protocol as Stage 16/16b (SPEC3_ADDENDUM.md section 22): the trained
n=4/L8/R0/reupload=true anchor checkpoint, 180 blobs3 samples, one batched
SamplerV2 job at 2048 shots/circuit. Only the backend changes (ibm_kingston
instead of ibm_marrakesh) -- this is a cross-hardware replication of HW1,
not a new experiment design.

Two-phase by design: `build_only()` transpiles and reports circuit stats
with NO hardware call, so the cost estimate can be sanity-checked before
`submit()` is ever invoked.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

from qnc.data import load_blobs3, pca_encode
from qnc.hardware import build_circuit
from qnc.models import VQC

N_QUBITS, NUM_CLASSES, N_LAYERS, SEED = 4, 3, 8, 0
CKPT_PATH = "results/transition_L_L8_s0_20260712-194651/checkpoints/epoch_0600.pt"
BACKEND_NAME = "ibm_kingston"
SHOTS = 2048
OUT_DIR = Path("results/stage18_hardware")


def _load_anchor():
    x_train_raw, y_train, x_test_raw, y_test = load_blobs3(
        samples_per_class=60, blob_dist=3.0, blob_std=0.8, test_fraction=0.2,
        seed=SEED, ambient_dim=N_QUBITS,
    )
    x_train, x_test = pca_encode(x_train_raw, x_test_raw, n_qubits=N_QUBITS, seed=SEED)
    x_all = torch.cat([x_train, x_test], dim=0).numpy()

    model = VQC(
        n_qubits=N_QUBITS, num_classes=NUM_CLASSES, n_layers=N_LAYERS,
        reupload=True, entangling=True, encoding="angle", readout_family="R0",
    )
    model.load_state_dict(torch.load(CKPT_PATH, weights_only=True))
    model.eval()
    return x_all, model.weights.detach().numpy()


def build_only() -> dict:
    """Builds + transpiles all 180 circuits against ibm_kingston's target,
    no hardware call. Returns stats for the pre-submission cost sanity check."""
    x_all, weights = _load_anchor()
    service = QiskitRuntimeService()
    backend = service.backend(BACKEND_NAME)

    circuits = [
        build_circuit(x_all[i], weights, N_QUBITS, NUM_CLASSES, measure=True)
        for i in range(x_all.shape[0])
    ]
    transpiled = transpile(circuits, backend=backend, optimization_level=1)
    depths = [c.depth() for c in transpiled]
    return {
        "backend": BACKEND_NAME,
        "backend_num_qubits": backend.num_qubits,
        "pending_jobs": backend.status().pending_jobs,
        "n_circuits": len(transpiled),
        "shots_per_circuit": SHOTS,
        "mean_depth": float(np.mean(depths)),
        "max_depth": int(np.max(depths)),
    }


def submit() -> str:
    """Submits the one batched job. Only call after explicit user go-ahead."""
    x_all, weights = _load_anchor()
    service = QiskitRuntimeService()
    backend = service.backend(BACKEND_NAME)

    circuits = [
        build_circuit(x_all[i], weights, N_QUBITS, NUM_CLASSES, measure=True)
        for i in range(x_all.shape[0])
    ]
    transpiled = transpile(circuits, backend=backend, optimization_level=1)

    sampler = SamplerV2(mode=backend)
    job = sampler.run(transpiled, shots=SHOTS)
    job_id = job.job_id()

    out_dir = OUT_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "stage18_b5_x_all.npy", x_all)
    with open(out_dir / "stage18_b5_job_id.txt", "w") as f:
        f.write(f"{job_id}\n{BACKEND_NAME}\n")
    print(f"submitted: {job_id} on {BACKEND_NAME}")
    return job_id


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "submit":
        submit()
    else:
        print(json.dumps(build_only(), indent=2))
