"""Stage 19 Arm V3: fiber_fraction vs qubit count (PREDICTIONS.md "Stage 19
preregistration", Arm V3). Free -- reuses already-reported/scored numbers
(Stage 12 P4, Stage 15 HN2, Stage 18 B3), no new runs, so CLAUDE.md
invariant 4 (figures as a pure function of metrics.jsonl) does not apply:
this is a meta-analysis figure built from findings-file numbers, analogous
to `stage18_schematic.py`.

n=4 has no reupload-arm split reported anywhere (Stage 12's P4 predates the
true/false dichotomy introduced at Stage 14B) -- plotted as a single point,
not fabricated as either arm's value; the gap is real and shown honestly.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qnc.metrics import isotropic_fiber_null

# (n_qubits, fiber_fraction) -- verbatim from findings files, not recomputed.
N4_REFERENCE = (4, 0.9934)  # STAGE12_FINDINGS.md P4, L=8 anchor, reupload=true
N6_TRUE = (6, 0.99963)  # STAGE15_FINDINGS.md HN2
N6_FALSE = (6, 0.99917)  # STAGE15_FINDINGS.md HN2
N8_TRUE = (8, 0.999977)  # STAGE18_FINDINGS.md B3, 5-seed mean
N8_FALSE = (8, 0.999847)  # STAGE18_FINDINGS.md B3, 5-seed mean


def make_scaling_figure(out_path: str | Path = "figs/fig18_stage19_scaling.png") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_true = [N4_REFERENCE[0], N6_TRUE[0], N8_TRUE[0]]
    fiber_true = [N4_REFERENCE[1], N6_TRUE[1], N8_TRUE[1]]
    n_false = [N6_FALSE[0], N8_FALSE[0]]
    fiber_false = [N6_FALSE[1], N8_FALSE[1]]

    n_dense = np.arange(4, 9)
    isotropic = [isotropic_fiber_null(int(n), measured_dim=3) for n in n_dense]

    fig, ax = plt.subplots(figsize=(3.5, 3.0))  # column-sized

    ax.plot(n_dense, isotropic, linestyle="--", color="grey", marker="x",
             label="isotropic null (1-3/(4^n-1))", zorder=1)
    ax.plot(n_true, fiber_true, marker="o", color="#1b9e77",
             label="reupload=true", zorder=2)
    ax.plot(n_false, fiber_false, marker="s", color="#d95f02",
             label="reupload=false", zorder=2)
    ax.scatter([N4_REFERENCE[0]], [N4_REFERENCE[1]], marker="o",
               facecolors="none", edgecolors="#1b9e77", s=80, zorder=3)
    ax.annotate("n=4: single reference\n(no arm split reported)",
                xy=N4_REFERENCE, xytext=(4.3, 0.9895),
                fontsize=6, color="#333333")

    ax.set_xlabel("qubit count n")
    ax.set_ylabel("fiber_fraction (final epoch)")
    ax.set_xticks([4, 6, 8])
    ax.set_ylim(0.985, 1.0005)
    ax.legend(fontsize=6, loc="lower right")
    ax.set_title("Fiber-fraction scaling with qubit count", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    path = make_scaling_figure()
    print(f"wrote {path}")
