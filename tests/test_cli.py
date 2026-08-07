import pytest

import cbir.eval.results as results_module
from cbir.cli import _format_params, _render, results_cmd, sweep_configs
from cbir.configs.classic import BoWConfig, VLADConfig
from cbir.eval.results import RunRecord, append


@pytest.fixture
def runs_path(tmp_path, monkeypatch):
    path = tmp_path / "runs.jsonl"
    monkeypatch.setattr(results_module, "RESULTS_PATH", path)
    return path


def _record(technique="bow", k=5000, dataset="roxford5k", map_=0.1484, **kwargs) -> RunRecord:
    # The held-out set is always the *other* dataset -- see holdout.py. Getting this
    # right in the fixture matters: with both columns set to the same name, a filter
    # test cannot tell which column it actually matched.
    return RunRecord(
        dataset=dataset,
        held_out_dataset="rparis6k" if dataset == "roxford5k" else "roxford5k",
        technique=technique,
        params={"k": k, "seed": 0},
        metrics={
            "easy": {"map": 0.1775, "mp_at_k": 0.3211, "k": 10, "num_queries": 68, "num_excluded": 2},
            "medium": {"map": map_, "mp_at_k": 0.3050, "k": 10, "num_queries": 70, "num_excluded": 0},
            "hard": {"map": 0.0570, "mp_at_k": 0.0855, "k": 10, "num_queries": 70, "num_excluded": 0},
        },
        commit="abc1234",
        **kwargs,
    )


def test_sweep_expands_k_into_one_config_each():
    configs = sweep_configs(BoWConfig(k=5000, seed=3), "roxford5k", 10, (1000, 20000))

    assert [config.descriptor.k for config in configs] == [1000, 20000]
    # Everything else must survive the expansion -- a sweep that silently reset the
    # seed would produce rows that look comparable and are not.
    assert {config.descriptor.seed for config in configs} == {3}
    assert {config.dataset for config in configs} == {"roxford5k"}


def test_sweep_preserves_technique_specific_knobs():
    configs = sweep_configs(VLADConfig(k=64, intra_norm=False, power=0.5), "rparis6k", 5, (16, 32))

    assert [config.descriptor.k for config in configs] == [16, 32]
    assert all(config.descriptor.intra_norm is False for config in configs)
    assert all(config.descriptor.power == 0.5 for config in configs)
    assert all(config.mp_at_k == 5 for config in configs)


def test_no_sweep_runs_the_configured_k_once():
    configs = sweep_configs(BoWConfig(k=5000), "roxford5k", 10, ())
    assert [config.descriptor.k for config in configs] == [5000]


def test_format_params_is_sorted_so_columns_line_up():
    assert _format_params({"seed": 0, "k": 5000}) == "k=5000 seed=0"


def test_render_pads_columns_to_content_width():
    table = _render([["a", "longer"], ["bbbb", "x"]], ["h1", "h2"])
    lines = table.splitlines()
    assert lines[0].startswith("h1  ")
    # Every row's second column starts at the same offset.
    assert lines[0].index("h2") == lines[2].index("longer") == lines[3].index("x")


def test_results_prints_a_row_per_run(runs_path, capsys):
    append(_record(k=5000), runs_path)
    append(_record(k=20000, map_=0.1800), runs_path)

    results_cmd()

    out = capsys.readouterr().out
    assert "k=5000 seed=0" in out
    assert "k=20000 seed=0" in out
    assert "0.1484" in out and "0.1800" in out


def test_results_reports_an_empty_store_distinctly_from_an_empty_filter(runs_path, capsys):
    # Saying "nothing recorded" when rows exist would send you looking in the wrong
    # place, so the two cases must not share a message.
    results_cmd()
    assert "no runs recorded yet" in capsys.readouterr().out

    append(_record(technique="bow"), runs_path)
    results_cmd(technique="vlad")
    out = capsys.readouterr().out
    assert "no runs match that filter" in out
    assert "1 recorded" in out


def test_results_filters_by_dataset_and_technique(runs_path, capsys):
    append(_record(technique="bow", dataset="roxford5k"), runs_path)
    append(_record(technique="vlad", dataset="roxford5k"), runs_path)
    append(_record(technique="bow", dataset="rparis6k"), runs_path)

    results_cmd(technique="bow")
    out = capsys.readouterr().out
    assert out.count("bow") >= 2
    assert "vlad" not in out

    results_cmd(dataset="rparis6k")
    # Check the dataset column specifically: "roxford5k" still appears on this row as
    # the *held-out* set, so a substring check over the whole output proves nothing.
    data_rows = capsys.readouterr().out.splitlines()[2:]
    assert [row.split()[0] for row in data_rows] == ["rparis6k"]
    assert [row.split()[1] for row in data_rows] == ["roxford5k"]


def test_results_metric_switch_tabulates_mp_at_k(runs_path, capsys):
    append(_record(), runs_path)

    results_cmd(metric="mp_at_k")
    out = capsys.readouterr().out
    assert "0.3211" in out  # easy mp_at_k
    assert "0.1775" not in out  # easy map must not leak in


def test_results_current_only_collapses_reruns(runs_path, capsys):
    append(_record(k=5000, map_=0.1000, recorded_at="2026-01-01T00:00:00+00:00"), runs_path)
    append(_record(k=5000, map_=0.1484, recorded_at="2026-06-01T00:00:00+00:00"), runs_path)

    results_cmd()
    assert capsys.readouterr().out.count("k=5000") == 2

    results_cmd(current_only=True)
    out = capsys.readouterr().out
    assert out.count("k=5000") == 1
    assert "0.1484" in out and "0.1000" not in out


def test_results_shows_a_dash_for_a_missing_protocol(runs_path, capsys):
    record = RunRecord(
        dataset="roxford5k",
        held_out_dataset="rparis6k",
        technique="bow",
        params={"k": 5000},
        metrics={"medium": {"map": 0.1484}},  # easy/hard absent
    )
    append(record, runs_path)

    results_cmd()
    # `"-" in out` would be vacuous -- the header separator is all dashes. Check the
    # protocol cells of the data row itself.
    cells = capsys.readouterr().out.splitlines()[2].split()
    easy, medium, hard = cells[4], cells[5], cells[6]
    assert (easy, medium, hard) == ("-", "0.1484", "-")
