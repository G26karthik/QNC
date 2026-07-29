"""Tests for src/qnc/logging_utils.py: JSONLLogger (SPEC §7 schema) and
run_id generation (CLAUDE.md invariant 3)."""

import json
import re
from datetime import datetime

import pytest

from qnc.logging_utils import JSONLLogger, REQUIRED_FIELDS, generate_run_id


def make_record(**overrides):
    record = {
        "run_id": "stage0_smoke_s0_20260101-000000",
        "epoch": 0,
        "split": "train",
        "loss": 1.0,
        "acc": 0.5,
        "beta": 5.0,
        "qnc1_trace": 0.1,
        "qnc1_hs": 0.1,
        "qnc1_rel": 0.1,
        "equinorm_cv": 0.0,
        "equiangle_dev": 0.0,
        "equiangle_std": 0.0,
        "gram_offdiag_cos": [-0.5, -0.5, -0.5],
        "class_mean_norms": [1.0, 1.0, 1.0],
        "entropy_mean": 0.5,
        "entropy_class": [0.5, 0.5, 0.5],
        "entropy_classmean": [0.5, 0.5, 0.5],
        "tpt_reached": False,
        "e0_epoch": None,
        "seed": 0,
        "git_sha": "abc1234",
        "wall_time_s": 1.23,
        "qnc1_fisher": 1.0,
        "qnc1_ratio": 0.5,
        "overlap_offdiag": 0.1,
        "purity_class": [0.9, 0.9, 0.9],
        "purity_mean": 0.9,
        "n_params": 24,
        "m_over_mc": 0.0941,
        "qfi_rank": None,
        "loss_type": "ce",
        "dataset": "blobs3",
        "encoding": "angle",
    }
    record.update(overrides)
    return record


class TestGenerateRunId:
    def test_format_matches_config_seed_timestamp(self):
        ts = datetime(2026, 1, 2, 3, 4, 5)
        run_id = generate_run_id("stage0_smoke", seed=0, timestamp=ts)
        assert run_id == "stage0_smoke_s0_20260102-030405"

    def test_matches_pattern_without_explicit_timestamp(self):
        run_id = generate_run_id("stage1_classical", seed=2)
        assert re.match(r"^stage1_classical_s2_\d{8}-\d{6}$", run_id)


class TestJSONLLogger:
    def test_log_writes_one_json_line(self, tmp_path):
        logger = JSONLLogger("run_a", results_dir=tmp_path)
        logger.log(make_record())
        lines = logger.path.read_text().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == make_record()

    def test_log_is_append_only(self, tmp_path):
        logger = JSONLLogger("run_b", results_dir=tmp_path)
        logger.log(make_record(epoch=0))
        logger.log(make_record(epoch=1))
        lines = logger.path.read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["epoch"] == 0
        assert json.loads(lines[1])["epoch"] == 1

    def test_log_rejects_record_missing_required_field(self, tmp_path):
        logger = JSONLLogger("run_c", results_dir=tmp_path)
        record = make_record()
        del record["beta"]
        with pytest.raises(ValueError):
            logger.log(record)

    def test_refuses_to_overwrite_existing_results_dir(self, tmp_path):
        JSONLLogger("run_d", results_dir=tmp_path)
        with pytest.raises(FileExistsError):
            JSONLLogger("run_d", results_dir=tmp_path)

    def test_required_fields_matches_spec_schema(self):
        assert REQUIRED_FIELDS == frozenset(make_record().keys())
