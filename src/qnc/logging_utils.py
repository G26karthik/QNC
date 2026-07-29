"""Append-only JSONL logging.

Implements SPEC.md §7 logging schema and CLAUDE.md invariant 3 (append-only
logging, one JSONL line per logged epoch, run_id generation).
"""

import json
from datetime import datetime
from pathlib import Path

REQUIRED_FIELDS = frozenset(
    {
        "run_id",
        "epoch",
        "split",
        "loss",
        "acc",
        "beta",
        "qnc1_trace",
        "qnc1_hs",
        "qnc1_rel",
        "equinorm_cv",
        "equiangle_dev",
        "equiangle_std",
        "gram_offdiag_cos",
        "class_mean_norms",
        "entropy_mean",
        "entropy_class",
        "entropy_classmean",
        "tpt_reached",
        "e0_epoch",
        "seed",
        "git_sha",
        "wall_time_s",
        # SPEC2_ADDENDUM.md §15
        "qnc1_fisher",
        "qnc1_ratio",
        "overlap_offdiag",
        "purity_class",
        "purity_mean",
        "n_params",
        "m_over_mc",
        "qfi_rank",
        "loss_type",
        "dataset",
        "encoding",
    }
)


def generate_run_id(config_name: str, seed: int, timestamp: datetime | None = None) -> str:
    """run_id = <config_name>_s<seed>_<YYYYmmdd-HHMMSS>, per CLAUDE.md."""
    ts = timestamp if timestamp is not None else datetime.now()
    return f"{config_name}_s{seed}_{ts.strftime('%Y%m%d-%H%M%S')}"


class JSONLLogger:
    """Appends one JSON line per logged epoch to results/<run_id>/metrics.jsonl.

    Never overwrites an existing results dir (CLAUDE.md invariant 3).
    """

    def __init__(self, run_id: str, results_dir: str | Path = "results"):
        self.run_id = run_id
        self.run_dir = Path(results_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.path = self.run_dir / "metrics.jsonl"

    def log(self, record: dict) -> None:
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(f"record missing required SPEC §7 fields: {sorted(missing)}")
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
