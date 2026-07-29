"""Unit tests for src/qnc/sweep.py (PLAYBOOK Stage 4): dotted-path config
override merging and the variant x seed dispatch loop. dispatch_train and
make_figures are monkeypatched so these tests don't run real training (same
policy as test_train.py: the stochastic training loop itself isn't unit
tested, per CLAUDE.md invariant 1).
"""

import copy
import json

from qnc.sweep import _set_by_dotted_path, merge_overrides, run_sweep


class TestSetByDottedPath:
    def test_sets_nested_key(self):
        config = {"model": {"entangling": True}}
        _set_by_dotted_path(config, "model.entangling", False)
        assert config["model"]["entangling"] is False

    def test_sets_top_level_key(self):
        config = {"seed": 0}
        _set_by_dotted_path(config, "seed", 5)
        assert config["seed"] == 5


class TestMergeOverrides:
    def test_does_not_mutate_base(self):
        base = {"model": {"entangling": True}}
        merged = merge_overrides(base, {"model.entangling": False})
        assert base["model"]["entangling"] is True
        assert merged["model"]["entangling"] is False

    def test_applies_multiple_overrides_leaves_others_untouched(self):
        base = {"model": {"n_qubits": 6, "layers": 4}, "train": {"weight_decay": 0.0}}
        merged = merge_overrides(base, {"model.n_qubits": 8, "train.weight_decay": 1e-4})
        assert merged["model"]["n_qubits"] == 8
        assert merged["train"]["weight_decay"] == 1e-4
        assert merged["model"]["layers"] == 4


def _patch_dispatch_and_figures(monkeypatch, calls=None, seen_configs=None):
    def fake_dispatch(config, seed_override=None):
        run_id = f"{config['experiment']}_s{seed_override}_fake"
        if calls is not None:
            calls.append(run_id)
        if seen_configs is not None:
            seen_configs.append(copy.deepcopy(config))
        return run_id

    monkeypatch.setattr("qnc.sweep.dispatch_train", fake_dispatch)
    monkeypatch.setattr("qnc.sweep.make_figures", lambda *a, **k: None)
    monkeypatch.setattr("qnc.sweep.build_cross_config_table", lambda manifest, results_dir=None: "fake report")
    comparison_calls = []
    monkeypatch.setattr(
        "qnc.sweep.make_sweep_comparison_figure",
        lambda sweep_config, manifest, out_path, results_dir=None: comparison_calls.append(out_path),
    )
    return comparison_calls


class TestRunSweep:
    def test_dispatches_each_variant_x_seed_and_writes_manifest(self, tmp_path, monkeypatch):
        calls = []
        _patch_dispatch_and_figures(monkeypatch, calls=calls)

        base_config = {"experiment": "base", "seed": 0, "model": {"entangling": True}}
        sweep_config = {
            "name": "ablation_entangling",
            "seeds": [0, 1],
            "variants": [
                {"label": "true", "overrides": {}},
                {"label": "false", "overrides": {"model.entangling": False}},
            ],
        }

        manifest, manifest_path = run_sweep(sweep_config, base_config, results_dir=tmp_path)

        assert set(manifest.keys()) == {"true", "false"}
        assert set(manifest["true"].keys()) == {"0", "1"}
        assert len(calls) == 4  # 2 variants x 2 seeds
        assert manifest_path.exists()
        with open(manifest_path) as f:
            assert json.load(f) == manifest

    def test_variant_overrides_do_not_leak_across_variants(self, tmp_path, monkeypatch):
        seen_configs = []
        _patch_dispatch_and_figures(monkeypatch, seen_configs=seen_configs)

        base_config = {"experiment": "base", "seed": 0, "model": {"entangling": True}}
        sweep_config = {
            "name": "abl",
            "seeds": [0],
            "variants": [
                {"label": "true", "overrides": {}},
                {"label": "false", "overrides": {"model.entangling": False}},
            ],
        }
        run_sweep(sweep_config, base_config, results_dir=tmp_path)

        assert seen_configs[0]["model"]["entangling"] is True
        assert seen_configs[1]["model"]["entangling"] is False

    def test_writes_comparison_figure_and_cross_config_report(self, tmp_path, monkeypatch):
        comparison_calls = _patch_dispatch_and_figures(monkeypatch)

        base_config = {"experiment": "base", "seed": 0, "model": {"n_qubits": 6}}
        sweep_config = {
            "name": "sweep_qubits",
            "seeds": [0],
            "variants": [{"label": "n6", "overrides": {}, "value": 6}],
        }
        manifest, manifest_path = run_sweep(sweep_config, base_config, results_dir=tmp_path)

        sweep_dir = manifest_path.parent
        assert len(comparison_calls) == 1
        assert comparison_calls[0] == sweep_dir / "comparison.png"
        assert (sweep_dir / "report.md").read_text() == "fake report"
