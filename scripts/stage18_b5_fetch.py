"""Fetches Stage 18 B5's job result once complete, saves counts.json +
job_log.json in the same format as Stage 16 (results/stage16_hardware/*),
so qnc.stage16_hardware_analysis.run_analysis can be reused unchanged."""

import json
import sys
from pathlib import Path

from qiskit_ibm_runtime import QiskitRuntimeService

JOB_ID = sys.argv[1] if len(sys.argv) > 1 else open("results/stage18_hardware").read()
OUT_DIR = Path("results/stage18_hardware") / JOB_ID


def main():
    service = QiskitRuntimeService()
    job = service.job(JOB_ID)
    status = job.status()
    print("status:", status)
    if status != "DONE":
        return

    result = job.result()
    counts_list = [pub_result.data.c.get_counts() for pub_result in result]

    metrics = job.metrics()
    job_log = {
        "job_id": JOB_ID,
        "backend": job.backend().name,
        "final_status": str(status),
        "metrics": metrics,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "counts.json", "w") as f:
        json.dump(counts_list, f)
    with open(OUT_DIR / "job_log.json", "w") as f:
        json.dump(job_log, f, indent=2, default=str)
    print("saved:", OUT_DIR)


if __name__ == "__main__":
    main()
